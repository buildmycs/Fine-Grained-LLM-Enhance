"""Regression, balanced ordinal, and continuous intensity contrastive losses."""

import math
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def sentiment_class_ids(labels):
    """Match ALMT Acc-7: clip to [-3, 3], round, then shift to [0, 6]."""
    return torch.round(labels.view(-1).clamp(-3.0, 3.0)).long() + 3


def ordinal_targets(labels):
    class_ids = sentiment_class_ids(labels)
    thresholds = torch.arange(6, device=labels.device).view(1, -1)
    return (class_ids.view(-1, 1) > thresholds).to(labels.dtype)


def _renormalize_binary_weights(
    positive, negative, positive_count, negative_count, total
):
    denominator = positive * positive_count + negative * negative_count
    if denominator <= 0:
        return 1.0, 1.0
    scale = total / denominator
    return positive * scale, negative * scale


def compute_balance_weights(labels, max_weight=5.0):
    """Compute train-only weights for ordinal thresholds and Acc-7 classes."""
    if max_weight < 1.0:
        raise ValueError("max_weight must be at least 1.0")
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    class_ids = np.round(np.clip(labels, -3.0, 3.0)).astype(np.int64) + 3
    total = len(class_ids)
    if total == 0:
        raise ValueError("Cannot compute balance weights from an empty dataset")

    positive_weights, negative_weights = [], []
    for threshold in range(6):
        positive_count = int(np.sum(class_ids > threshold))
        negative_count = total - positive_count
        positive = total / (2.0 * max(positive_count, 1))
        negative = total / (2.0 * max(negative_count, 1))
        positive = min(positive, max_weight)
        negative = min(negative, max_weight)
        positive, negative = _renormalize_binary_weights(
            positive,
            negative,
            positive_count,
            negative_count,
            total,
        )
        positive = min(positive, max_weight)
        negative = min(negative, max_weight)
        positive_weights.append(positive)
        negative_weights.append(negative)

    class_counts = np.bincount(class_ids, minlength=7).astype(np.float64)
    nonzero_counts = class_counts[class_counts > 0]
    largest_count = float(nonzero_counts.max())
    class_weights = np.ones(7, dtype=np.float64)
    present = class_counts > 0
    class_weights[present] = np.sqrt(largest_count / class_counts[present])
    class_weights = np.minimum(class_weights, max_weight)
    average_weight = np.sum(class_counts * class_weights) / total
    class_weights /= max(average_weight, 1e-12)

    return {
        "ordinal_positive": np.asarray(positive_weights, dtype=np.float32),
        "ordinal_negative": np.asarray(negative_weights, dtype=np.float32),
        "class_weights": class_weights.astype(np.float32),
        "class_counts": class_counts.astype(np.int64),
    }


class ContinuousIntensityContrastiveLoss(nn.Module):
    """
    Match representation similarities to continuous label-distance similarities.

    For every anchor, the target distribution over the other samples is
    proportional to exp(-|y_i-y_j| / label_temperature).
    """

    def __init__(self, temperature=0.1, label_temperature=0.5):
        super().__init__()
        if temperature <= 0 or label_temperature <= 0:
            raise ValueError("contrastive temperatures must be positive")
        self.temperature = temperature
        self.label_temperature = label_temperature

    def forward(self, features, labels, anchor_weights=None):
        batch_size = features.size(0)
        if batch_size < 2:
            return features.sum() * 0.0

        features = F.normalize(features, dim=-1)
        similarities = torch.matmul(features, features.transpose(0, 1))
        similarities = similarities / self.temperature

        labels = labels.view(-1)
        label_distances = torch.abs(labels[:, None] - labels[None, :])
        target_logits = -label_distances / self.label_temperature

        diagonal = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        similarities = similarities.masked_fill(diagonal, float("-inf"))
        target_logits = target_logits.masked_fill(diagonal, float("-inf"))

        log_probabilities = F.log_softmax(similarities, dim=1)
        target_probabilities = F.softmax(target_logits, dim=1)
        # Avoid the undefined 0 * -inf product on the excluded diagonal.
        log_probabilities = log_probabilities.masked_fill(diagonal, 0.0)
        per_anchor = -(target_probabilities * log_probabilities).sum(dim=1)

        if anchor_weights is None:
            return per_anchor.mean()
        anchor_weights = anchor_weights.view(-1)
        weighted_sum = torch.sum(per_anchor * anchor_weights)
        return weighted_sum / anchor_weights.sum().clamp_min(1e-8)


class SentimentIntensityObjective(nn.Module):
    def __init__(
        self,
        balance_weights,
        regression_weight=1.0,
        ordinal_weight=0.2,
        contrastive_weight=0.05,
        contrastive_temperature=0.1,
        contrastive_label_temperature=0.5,
        auxiliary_warmup_epochs=5,
        ordinal_decay_start_epoch=None,
        ordinal_decay_end_epoch=None,
        ordinal_decay_final_scale=0.0,
    ):
        super().__init__()
        self.regression_weight = float(regression_weight)
        self.ordinal_weight = float(ordinal_weight)
        self.contrastive_weight = float(contrastive_weight)
        if min(
            self.regression_weight,
            self.ordinal_weight,
            self.contrastive_weight,
        ) < 0:
            raise ValueError("objective weights must be non-negative")
        self.auxiliary_warmup_epochs = int(auxiliary_warmup_epochs)
        if self.auxiliary_warmup_epochs < 0:
            raise ValueError("auxiliary_warmup_epochs must be non-negative")

        decay_epochs = (ordinal_decay_start_epoch, ordinal_decay_end_epoch)
        if (decay_epochs[0] is None) != (decay_epochs[1] is None):
            raise ValueError(
                "ordinal_decay_start_epoch and ordinal_decay_end_epoch "
                "must either both be set or both be omitted"
            )
        self.ordinal_decay_start_epoch = (
            None
            if ordinal_decay_start_epoch is None
            else int(ordinal_decay_start_epoch)
        )
        self.ordinal_decay_end_epoch = (
            None
            if ordinal_decay_end_epoch is None
            else int(ordinal_decay_end_epoch)
        )
        self.ordinal_decay_final_scale = float(ordinal_decay_final_scale)
        if not 0.0 <= self.ordinal_decay_final_scale <= 1.0:
            raise ValueError("ordinal_decay_final_scale must be in [0, 1]")
        if self.ordinal_decay_start_epoch is not None:
            if self.ordinal_decay_start_epoch < self.auxiliary_warmup_epochs:
                raise ValueError(
                    "ordinal_decay_start_epoch must not precede the end of "
                    "auxiliary warm-up"
                )
            if self.ordinal_decay_end_epoch <= self.ordinal_decay_start_epoch:
                raise ValueError(
                    "ordinal_decay_end_epoch must be greater than "
                    "ordinal_decay_start_epoch"
                )
        self.current_epoch = 1
        self.requires_auxiliary_outputs = (
            self.ordinal_weight > 0 or self.contrastive_weight > 0
        )

        self.register_buffer(
            "ordinal_positive_weights",
            torch.as_tensor(balance_weights["ordinal_positive"]),
        )
        self.register_buffer(
            "ordinal_negative_weights",
            torch.as_tensor(balance_weights["ordinal_negative"]),
        )
        self.register_buffer(
            "class_weights",
            torch.as_tensor(balance_weights["class_weights"]),
        )
        self.register_buffer(
            "class_counts",
            torch.as_tensor(balance_weights["class_counts"]),
        )
        self.contrastive_loss = ContinuousIntensityContrastiveLoss(
            temperature=contrastive_temperature,
            label_temperature=contrastive_label_temperature,
        )

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def auxiliary_scale(self):
        if self.auxiliary_warmup_epochs <= 0:
            return 1.0
        return min(self.current_epoch / self.auxiliary_warmup_epochs, 1.0)

    def ordinal_scale(self):
        """Return warm-up followed by an optional cosine decay multiplier."""
        warmup_scale = self.auxiliary_scale()
        if self.ordinal_decay_start_epoch is None:
            return warmup_scale
        if self.current_epoch <= self.ordinal_decay_start_epoch:
            return warmup_scale
        if self.current_epoch >= self.ordinal_decay_end_epoch:
            return self.ordinal_decay_final_scale

        decay_progress = (
            (self.current_epoch - self.ordinal_decay_start_epoch)
            / (self.ordinal_decay_end_epoch - self.ordinal_decay_start_epoch)
        )
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return self.ordinal_decay_final_scale + (
            1.0 - self.ordinal_decay_final_scale
        ) * cosine_scale

    def _balanced_ordinal_loss(self, logits, labels):
        targets = ordinal_targets(labels)
        element_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        weights = (
            targets * self.ordinal_positive_weights.view(1, -1)
            + (1.0 - targets) * self.ordinal_negative_weights.view(1, -1)
        )
        return torch.mean(element_loss * weights)

    def forward(self, outputs, labels):
        if torch.is_tensor(outputs):
            outputs = {"prediction": outputs}

        prediction = outputs["prediction"]
        regression_loss = F.mse_loss(prediction, labels)
        zero = regression_loss.new_zeros(())

        ordinal_loss = zero
        if self.ordinal_weight > 0:
            if "ordinal_logits" not in outputs:
                raise KeyError("ordinal_logits are required by ordinal_weight")
            ordinal_loss = self._balanced_ordinal_loss(
                outputs["ordinal_logits"], labels
            )

        contrastive_loss = zero
        if self.contrastive_weight > 0:
            if "contrastive_features" not in outputs:
                raise KeyError(
                    "contrastive_features are required by contrastive_weight"
                )
            class_ids = sentiment_class_ids(labels)
            anchor_weights = self.class_weights[class_ids]
            contrastive_loss = self.contrastive_loss(
                outputs["contrastive_features"], labels, anchor_weights
            )

        auxiliary_scale = self.auxiliary_scale()
        ordinal_scale = self.ordinal_scale()
        ordinal_effective_weight = self.ordinal_weight * ordinal_scale
        total_loss = (
            self.regression_weight * regression_loss
            + ordinal_effective_weight * ordinal_loss
            + auxiliary_scale * self.contrastive_weight * contrastive_loss
        )
        values = {
            "total": total_loss.detach().item(),
            "regression": regression_loss.detach().item(),
            "ordinal": ordinal_loss.detach().item(),
            "contrastive": contrastive_loss.detach().item(),
            "auxiliary_scale": auxiliary_scale,
            "ordinal_scale": ordinal_scale,
            "ordinal_effective_weight": ordinal_effective_weight,
        }
        return total_loss, values

    def describe(self):
        return {
            "regression_weight": self.regression_weight,
            "ordinal_weight": self.ordinal_weight,
            "contrastive_weight": self.contrastive_weight,
            "auxiliary_warmup_epochs": self.auxiliary_warmup_epochs,
            "ordinal_decay_start_epoch": self.ordinal_decay_start_epoch,
            "ordinal_decay_end_epoch": self.ordinal_decay_end_epoch,
            "ordinal_decay_final_scale": self.ordinal_decay_final_scale,
            "class_counts_-3_to_+3": self.class_counts.tolist(),
            "contrastive_class_weights": self.class_weights.tolist(),
            "ordinal_positive_weights": self.ordinal_positive_weights.tolist(),
            "ordinal_negative_weights": self.ordinal_negative_weights.tolist(),
        }


def build_sentiment_objective(args, train_labels):
    enabled = getattr(args.model, "use_intensity_objective", False)
    objective = getattr(args, "objective", SimpleNamespace())
    max_weight = getattr(objective, "max_balance_weight", 5.0)
    balance_weights = compute_balance_weights(train_labels, max_weight=max_weight)

    return SentimentIntensityObjective(
        balance_weights=balance_weights,
        regression_weight=getattr(objective, "regression_weight", 1.0),
        ordinal_weight=(
            getattr(objective, "ordinal_weight", 0.2) if enabled else 0.0
        ),
        contrastive_weight=(
            getattr(objective, "contrastive_weight", 0.05) if enabled else 0.0
        ),
        contrastive_temperature=getattr(
            objective, "contrastive_temperature", 0.1
        ),
        contrastive_label_temperature=getattr(
            objective, "contrastive_label_temperature", 0.5
        ),
        auxiliary_warmup_epochs=getattr(
            objective, "auxiliary_warmup_epochs", 5
        ),
        ordinal_decay_start_epoch=getattr(
            objective, "ordinal_decay_start_epoch", None
        ),
        ordinal_decay_end_epoch=getattr(
            objective, "ordinal_decay_end_epoch", None
        ),
        ordinal_decay_final_scale=getattr(
            objective, "ordinal_decay_final_scale", 0.0
        ),
    )
