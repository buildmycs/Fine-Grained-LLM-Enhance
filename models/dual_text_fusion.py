"""
Original-anchored fusion for raw and LLM-enhanced text representations.
"""

import math

import torch
from torch import nn


class GatedCrossTextFusion(nn.Module):
    """
    Let the original text query the LLM-enhanced text and gate the update.

    The original branch is always kept as a residual anchor:

        C = MHA(Q=H_original, K=H_enhanced, V=H_enhanced)
        G = sigmoid(W[H_original; C; |H_original-C|; H_original*C])
        H = LayerNorm(H_original + sigmoid(alpha) * G * C)
    """

    def __init__(
        self,
        dim,
        heads=8,
        dropout=0.1,
        scale_logit_init=-2.0,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(
                f"dual text dimension ({dim}) must be divisible by heads ({heads})"
            )

        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Linear(dim * 4, dim)
        self.update_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(dim)

        # sigmoid(-2) ~= 0.119. Starting close to the reproduced baseline makes
        # optimization less sensitive to noisy or over-explicit LLM rewrites.
        self.scale_logit = nn.Parameter(
            torch.tensor(float(scale_logit_init), dtype=torch.float32)
        )
        self._last_stats = {}

    def forward(self, original, enhanced):
        if original.shape != enhanced.shape:
            raise ValueError(
                "original and enhanced text representations must have the same "
                f"shape, got {tuple(original.shape)} and {tuple(enhanced.shape)}"
            )

        query = self.query_norm(original)
        key_value = self.context_norm(enhanced)
        context, attention = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=True,
            average_attn_weights=True,
        )

        gate_input = torch.cat(
            [
                original,
                context,
                torch.abs(original - context),
                original * context,
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate(gate_input))
        residual_scale = torch.sigmoid(self.scale_logit)
        fused = self.output_norm(
            original
            + residual_scale * self.update_dropout(gate * context)
        )

        # Detached diagnostics are exposed for TensorBoard and do not retain the
        # autograd graph.
        with torch.no_grad():
            self._last_stats = {
                "gate_mean": gate.mean().item(),
                "gate_std": gate.std(unbiased=False).item(),
                "residual_scale": residual_scale.item(),
                "attention_entropy": self._attention_entropy(attention).item(),
            }

        return fused

    @staticmethod
    def _attention_entropy(attention):
        probability = attention.clamp_min(torch.finfo(attention.dtype).tiny)
        entropy = -(probability * probability.log()).sum(dim=-1)
        normalizer = math.log(max(attention.size(-1), 2))
        return entropy.mean() / normalizer

    def get_last_stats(self):
        return dict(self._last_stats)
