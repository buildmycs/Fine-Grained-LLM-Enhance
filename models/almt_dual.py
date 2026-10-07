"""
Dual-text ALMT with a shared BERT and original-anchored gated cross-attention.
"""

import torch
from einops import repeat
from torch import nn

from .almt_layer import CrossTransformer, HhyperLearningEncoder, Transformer
from .bert import BertTextEncoder
from .dual_text_fusion import GatedCrossTextFusion
from .intensity_heads import IntensityProjectionHead, MonotonicOrdinalHead


class DualTextALMT(nn.Module):
    def __init__(self, args):
        super().__init__()
        args = args.model

        self.text_fusion_mode = getattr(
            args, "dual_fusion_mode", "gated_cross"
        ).lower()
        self.use_intensity_objective = getattr(
            args, "use_intensity_objective", False
        )
        self.ordinal_prediction_weight = float(
            getattr(args, "ordinal_prediction_weight", 0.0)
        )
        if not 0.0 <= self.ordinal_prediction_weight <= 1.0:
            raise ValueError("ordinal_prediction_weight must be in [0, 1]")
        supported_modes = {"gated_cross", "mean", "original", "enhanced"}
        if self.text_fusion_mode not in supported_modes:
            raise ValueError(
                f"dual_fusion_mode must be one of {sorted(supported_modes)}, "
                f"got '{self.text_fusion_mode}'"
            )

        self.h_hyper = nn.Parameter(
            torch.ones(1, args.token_len, args.token_dim)
        )

        # Both branches use this exact same BERT instance.
        self.bertmodel = BertTextEncoder(
            use_finetune=True,
            transformers="bert",
            pretrained=args.bert_pretrained,
        )

        # The language projection (including its learned tokens) is also shared.
        self.proj_l = nn.Sequential(
            nn.Linear(args.l_input_dim, args.l_proj_dst_dim),
            Transformer(
                num_frames=args.l_input_length,
                save_hidden=False,
                token_len=args.token_length,
                dim=args.proj_input_dim,
                depth=args.proj_depth,
                heads=args.proj_heads,
                mlp_dim=args.proj_mlp_dim,
            ),
        )
        self.proj_a = nn.Sequential(
            nn.Linear(args.a_input_dim, args.a_proj_dst_dim),
            Transformer(
                num_frames=args.a_input_length,
                save_hidden=False,
                token_len=args.token_length,
                dim=args.proj_input_dim,
                depth=args.proj_depth,
                heads=args.proj_heads,
                mlp_dim=args.proj_mlp_dim,
            ),
        )
        self.proj_v = nn.Sequential(
            nn.Linear(args.v_input_dim, args.v_proj_dst_dim),
            Transformer(
                num_frames=args.v_input_length,
                save_hidden=False,
                token_len=args.token_length,
                dim=args.proj_input_dim,
                depth=args.proj_depth,
                heads=args.proj_heads,
                mlp_dim=args.proj_mlp_dim,
            ),
        )

        if self.text_fusion_mode == "gated_cross":
            self.dual_text_fusion = GatedCrossTextFusion(
                dim=args.proj_input_dim,
                heads=getattr(args, "dual_cross_heads", args.proj_heads),
                dropout=getattr(args, "dual_cross_dropout", 0.1),
                scale_logit_init=getattr(args, "dual_scale_logit_init", -2.0),
            )
        else:
            self.dual_text_fusion = None

        self.l_encoder = Transformer(
            num_frames=args.token_length,
            save_hidden=True,
            token_len=None,
            dim=args.proj_input_dim,
            depth=args.AHL_depth - 1,
            heads=args.l_enc_heads,
            mlp_dim=args.l_enc_mlp_dim,
        )
        self.h_hyper_layer = HhyperLearningEncoder(
            dim=args.token_dim,
            depth=args.AHL_depth,
            heads=args.ahl_heads,
            dim_head=args.ahl_dim_head,
            dropout=args.ahl_droup,
        )
        self.fusion_layer = CrossTransformer(
            source_num_frames=args.token_len,
            tgt_num_frames=args.token_len,
            dim=args.proj_input_dim,
            depth=args.fusion_layer_depth,
            heads=args.fusion_heads,
            mlp_dim=args.fusion_mlp_dim,
        )
        self.regression_layer = nn.Linear(args.token_dim, 1)
        if self.use_intensity_objective:
            self.ordinal_head = MonotonicOrdinalHead(args.token_dim)
            self.intensity_projection = IntensityProjectionHead(
                input_dim=args.token_dim,
                projection_dim=getattr(
                    args, "contrastive_projection_dim", 64
                ),
                dropout=getattr(args, "contrastive_projection_dropout", 0.1),
            )
        else:
            self.ordinal_head = None
            self.intensity_projection = None

    def _encode_text_pair(self, original_text, enhanced_text):
        if original_text.shape != enhanced_text.shape:
            raise ValueError(
                "original and enhanced BERT inputs must have identical shapes, "
                f"got {tuple(original_text.shape)} and "
                f"{tuple(enhanced_text.shape)}"
            )

        batch_size = original_text.size(0)

        # Concatenating on the batch dimension proves parameter sharing and is
        # more efficient than calling the same BERT module twice.
        text_pair = torch.cat((original_text, enhanced_text), dim=0)
        bert_pair = self.bertmodel(text_pair)
        projected_pair = self.proj_l(bert_pair)
        original_hidden, enhanced_hidden = projected_pair.split(
            batch_size, dim=0
        )

        token_count = self.h_hyper.shape[1]
        return (
            original_hidden[:, :token_count],
            enhanced_hidden[:, :token_count],
        )

    def _fuse_text(self, original_hidden, enhanced_hidden):
        if self.text_fusion_mode == "gated_cross":
            return self.dual_text_fusion(original_hidden, enhanced_hidden)
        if self.text_fusion_mode == "mean":
            return 0.5 * (original_hidden + enhanced_hidden)
        if self.text_fusion_mode == "enhanced":
            return enhanced_hidden
        return original_hidden

    def forward(
        self,
        x_visual,
        x_audio,
        x_text,
        x_text_llm,
        return_aux=False,
    ):
        batch_size = x_visual.size(0)
        if x_audio.size(0) != batch_size or x_text.size(0) != batch_size:
            raise ValueError("visual, audio and text batch sizes must match")
        if x_text_llm.size(0) != batch_size:
            raise ValueError("enhanced text batch size must match other modalities")

        h_hyper = repeat(
            self.h_hyper, "1 n d -> b n d", b=batch_size
        )
        h_l_original, h_l_enhanced = self._encode_text_pair(
            x_text, x_text_llm
        )
        h_l = self._fuse_text(h_l_original, h_l_enhanced)

        token_count = self.h_hyper.shape[1]
        h_v = self.proj_v(x_visual)[:, :token_count]
        h_a = self.proj_a(x_audio)[:, :token_count]

        h_t_list = self.l_encoder(h_l)
        h_hyper = self.h_hyper_layer(
            h_t_list, h_a, h_v, h_hyper
        )
        feat = self.fusion_layer(h_hyper, h_t_list[-1])[:, 0]
        regression_prediction = self.regression_layer(feat)

        if not self.use_intensity_objective:
            return regression_prediction

        ordinal_logits, ordinal_prediction = self.ordinal_head(feat)
        weight = self.ordinal_prediction_weight
        prediction = (
            (1.0 - weight) * regression_prediction
            + weight * ordinal_prediction
        )
        if not return_aux:
            return prediction

        return {
            "prediction": prediction,
            "regression_prediction": regression_prediction,
            "ordinal_prediction": ordinal_prediction,
            "ordinal_logits": ordinal_logits,
            "contrastive_features": self.intensity_projection(feat),
        }

    def get_dual_text_stats(self):
        if self.dual_text_fusion is None:
            return {}
        return self.dual_text_fusion.get_last_stats()

    def get_ordinal_thresholds(self):
        if self.ordinal_head is None:
            return None
        return self.ordinal_head.thresholds().detach().cpu()


def build_model(args):
    return DualTextALMT(args)
