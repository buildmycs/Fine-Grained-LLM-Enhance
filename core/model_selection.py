"""Validation-only checkpoint selection for multimodal sentiment models."""

import math

import torch


def exact_metric_value(epoch_result, metric_name):
    """Return unrounded values for metrics whose tensors are available."""
    predictions = epoch_result["predictions"].view(-1)
    labels = epoch_result["labels"].view(-1)

    if metric_name == "MAE":
        return torch.mean(torch.abs(predictions - labels)).item()
    if metric_name == "Mult_acc_7":
        predicted_classes = torch.round(predictions.clamp(-3.0, 3.0))
        true_classes = torch.round(labels.clamp(-3.0, 3.0))
        return torch.mean((predicted_classes == true_classes).float()).item()

    results = epoch_result["results"]
    if metric_name not in results:
        raise KeyError(
            f"Selection metric '{metric_name}' is unavailable. "
            f"Available metrics: {sorted(results.keys())}"
        )
    return float(results[metric_name])


class ValidationMetricSelector:
    """Select one epoch using a primary metric and an optional tie-breaker."""

    def __init__(
        self,
        primary_metric="MAE",
        primary_mode="min",
        secondary_metric=None,
        secondary_mode="min",
        tie_tolerance=1e-12,
    ):
        self.primary_metric = primary_metric
        self.primary_mode = self._validate_mode(primary_mode)
        self.secondary_metric = secondary_metric
        self.secondary_mode = self._validate_mode(secondary_mode)
        self.tie_tolerance = float(tie_tolerance)

        self.best_primary = None
        self.best_secondary = None
        self.selected_epoch = None
        self.selected_validation_results = None

    @staticmethod
    def _validate_mode(mode):
        mode = str(mode).lower()
        if mode not in {"min", "max"}:
            raise ValueError("selection mode must be 'min' or 'max'")
        return mode

    def _is_better(self, current, best, mode):
        if best is None:
            return True
        if mode == "max":
            return current > best + self.tie_tolerance
        return current < best - self.tie_tolerance

    def _is_tied(self, current, best):
        return best is not None and math.isclose(
            current,
            best,
            rel_tol=0.0,
            abs_tol=self.tie_tolerance,
        )

    def consider(self, epoch, validation_result):
        primary = exact_metric_value(validation_result, self.primary_metric)
        secondary = (
            exact_metric_value(validation_result, self.secondary_metric)
            if self.secondary_metric is not None
            else None
        )

        improved = self._is_better(
            primary, self.best_primary, self.primary_mode
        )
        if (
            not improved
            and self.secondary_metric is not None
            and self._is_tied(primary, self.best_primary)
        ):
            improved = self._is_better(
                secondary, self.best_secondary, self.secondary_mode
            )

        if improved:
            self.best_primary = primary
            self.best_secondary = secondary
            self.selected_epoch = int(epoch)
            self.selected_validation_results = dict(
                validation_result["results"]
            )
        return improved

    def as_dict(self):
        if self.selected_epoch is None:
            raise RuntimeError("No validation epoch has been selected")
        return {
            "selected_epoch": self.selected_epoch,
            "primary_metric": self.primary_metric,
            "primary_mode": self.primary_mode,
            "primary_value": self.best_primary,
            "secondary_metric": self.secondary_metric,
            "secondary_mode": (
                self.secondary_mode
                if self.secondary_metric is not None
                else None
            ),
            "secondary_value": self.best_secondary,
            "validation_results": self.selected_validation_results,
        }
