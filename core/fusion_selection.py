"""Select inference fusion weights using validation outputs, never test data."""

import math

import torch

from core.model_selection import ValidationMetricSelector


def validate_rho_candidates(candidates):
    """None preserves the original fixed-rho checkpoint-selection protocol."""
    if candidates is None:
        return None
    if not isinstance(candidates, (list, tuple)) or not candidates:
        raise ValueError("validation_rho_candidates must be a non-empty list")
    values = [float(value) for value in candidates]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError(
            "validation_rho_candidates must contain finite values in [0, 1]"
        )
    if len(set(values)) != len(values):
        raise ValueError("validation_rho_candidates must not contain duplicates")
    # Exact ties within an epoch prefer the smaller rho, independent of YAML order.
    return sorted(values)


def select_validation_fusion(validation_result, candidates, metrics_fn, selector):
    """Return the epoch's best inference result without mutating training state.

    Use the same unrounded primary/secondary metrics as checkpoint selection.
    Loss recorders, if present, still describe the configured training rho.
    """
    candidates = validate_rho_candidates(candidates)
    if candidates is None:
        return validation_result
    required = ("regression_predictions", "ordinal_predictions", "labels")
    for key in required:
        if key not in validation_result:
            raise KeyError(f"{key} is required for validation rho selection")
    regression, ordinal, labels = (validation_result[key] for key in required)
    if regression.numel() == 0 or not (
        regression.shape == ordinal.shape == labels.shape
    ):
        raise ValueError(
            "Validation head predictions and labels need identical non-empty shapes"
        )
    if not all(
        torch.isfinite(values).all().item()
        for values in (regression, ordinal, labels)
    ):
        raise ValueError("Validation head predictions and labels must be finite")

    local_selector = ValidationMetricSelector(
        primary_metric=selector.primary_metric,
        primary_mode=selector.primary_mode,
        secondary_metric=selector.secondary_metric,
        secondary_mode=selector.secondary_mode,
        tie_tolerance=selector.tie_tolerance,
    )
    best_result = None
    with torch.no_grad():
        for rho in candidates:
            predictions = (1.0 - rho) * regression + rho * ordinal
            candidate = {
                **validation_result,
                "predictions": predictions,
                "results": metrics_fn(predictions, labels),
                "inference_rho": rho,
            }
            if local_selector.consider(0, candidate):
                best_result = candidate
    return best_result


def resolve_inference_rho(configured_rho, selection=None, override=None):
    """Explicit override > checkpoint's validation choice > legacy YAML rho."""
    selected_rho = (selection or {}).get("inference_rho", configured_rho)
    rho = float(selected_rho if override is None else override)
    if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise ValueError("inference rho must be finite and in [0, 1]")
    return rho
