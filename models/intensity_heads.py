"""Prediction heads for fine-grained ordinal sentiment intensity."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _inverse_softplus(value):
    return math.log(math.expm1(value))


class MonotonicOrdinalHead(nn.Module):
    """
    Six cumulative decisions for seven ordered classes (-3, ..., +3).

    The head estimates P(class > k) for k=0,...,5. Threshold increments are
    parameterized with softplus, so their order cannot be violated during
    optimization.
    """

    def __init__(self, input_dim, num_classes=7):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        self.num_classes = num_classes
        self.score = nn.Linear(input_dim, 1)

        # The initial thresholds match the seven ALMT rounding intervals.
        self.threshold_start = nn.Parameter(torch.tensor(-2.5))
        delta_count = num_classes - 2
        initial_delta = _inverse_softplus(1.0)
        self.threshold_delta_raw = nn.Parameter(
            torch.full((delta_count,), initial_delta)
        )

    def thresholds(self):
        if self.threshold_delta_raw.numel() == 0:
            return self.threshold_start.view(1)
        positive_deltas = F.softplus(self.threshold_delta_raw) + 1e-4
        remaining = self.threshold_start + torch.cumsum(positive_deltas, dim=0)
        return torch.cat((self.threshold_start.view(1), remaining), dim=0)

    def forward(self, features):
        score = self.score(features)
        logits = score - self.thresholds().view(1, -1)
        probabilities = torch.sigmoid(logits)
        expected_class = probabilities.sum(dim=-1, keepdim=True)
        expected_sentiment = expected_class - (self.num_classes // 2)
        return logits, expected_sentiment


class IntensityProjectionHead(nn.Module):
    """Projection used only by the continuous intensity contrastive loss."""

    def __init__(self, input_dim, projection_dim=64, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, projection_dim),
        )

    def forward(self, features):
        return F.normalize(self.network(features), dim=-1)
