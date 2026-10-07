import unittest

import torch

from core.fusion_selection import (
    resolve_inference_rho,
    select_validation_fusion,
    validate_rho_candidates,
)
from core.model_selection import ValidationMetricSelector, exact_metric_value


def metrics(predictions, labels):
    result = {"predictions": predictions, "labels": labels}
    return {
        name: round(exact_metric_value(result, name), 4)
        for name in ("Mult_acc_7", "MAE")
    }


def result(regression, ordinal, labels, rho=0.3):
    regression, ordinal, labels = (
        torch.tensor(values, dtype=torch.float32).view(-1, 1)
        for values in (regression, ordinal, labels)
    )
    predictions = (1 - rho) * regression + rho * ordinal
    return {
        "regression_predictions": regression,
        "ordinal_predictions": ordinal,
        "predictions": predictions,
        "labels": labels,
        "results": metrics(predictions, labels),
    }


def selector():
    return ValidationMetricSelector("Mult_acc_7", "max", "MAE", "min")


class FusionSelectionTest(unittest.TestCase):
    def test_disabled_selection_returns_original_object(self):
        original = result([0.0], [2.0], [0.0])
        self.assertIs(select_validation_fusion(original, None, metrics, selector()), original)

    def test_regression_only_can_win_without_mutating_raw_result_or_selector(self):
        original = result([0.0, 1.0], [2.0, 3.0], [0.0, 1.0])
        saved_prediction = original["predictions"].clone()
        saved_metrics = dict(original["results"])
        global_selector = selector()
        selected = select_validation_fusion(original, [0.0, 0.1, 0.2, 0.3], metrics, global_selector)
        self.assertEqual(selected["inference_rho"], 0.0)
        self.assertEqual(selected["results"]["Mult_acc_7"], 1.0)
        torch.testing.assert_close(selected["predictions"], original["regression_predictions"])
        torch.testing.assert_close(original["predictions"], saved_prediction)
        self.assertEqual(original["results"], saved_metrics)
        self.assertIsNone(global_selector.selected_epoch)
        self.assertNotIn("inference_rho", original)

    def test_nonzero_fusion_can_win(self):
        original = result([0.6, -0.6], [-1.4, 1.4], [0.0, 0.0])
        selected = select_validation_fusion(original, [0.0, 0.1, 0.2, 0.3], metrics, selector())
        self.assertEqual(selected["inference_rho"], 0.3)
        self.assertEqual(selected["results"]["Mult_acc_7"], 1.0)
        self.assertFalse(selected["predictions"].requires_grad)

    def test_mae_breaks_acc7_ties_using_unrounded_values(self):
        original = result([0.100002], [0.099998], [0.0])
        selected = select_validation_fusion(original, [0.0, 1.0], metrics, selector())
        self.assertEqual(selected["inference_rho"], 1.0)

    def test_exact_tie_prefers_smaller_rho(self):
        original = result([1.0], [1.0], [1.0])
        selected = select_validation_fusion(original, [0.3, 0.0, 0.1], metrics, selector())
        self.assertEqual(selected["inference_rho"], 0.0)

    def test_global_checkpoint_can_keep_earlier_regression_winner(self):
        early = result([0.0, 1.0], [2.0, 3.0], [0.0, 1.0])
        later = result([0.6, 1.0], [0.6, 1.0], [0.0, 1.0])
        fixed_selector = selector()
        self.assertTrue(fixed_selector.consider(21, early))
        self.assertTrue(fixed_selector.consider(70, later))
        global_selector = selector()
        early_selected = select_validation_fusion(early, [0.0, 0.3], metrics, global_selector)
        later_selected = select_validation_fusion(later, [0.0, 0.3], metrics, global_selector)
        self.assertTrue(global_selector.consider(21, early_selected))
        self.assertFalse(global_selector.consider(70, later_selected))
        self.assertEqual(global_selector.selected_epoch, 21)

    def test_missing_heads_rejected(self):
        original = result([0.0], [1.0], [0.0])
        del original["ordinal_predictions"]
        with self.assertRaisesRegex(KeyError, "ordinal_predictions"):
            select_validation_fusion(original, [0.0], metrics, selector())

    def test_invalid_arrays_rejected(self):
        for invalid in (float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                original = result([0.0], [0.0], [0.0])
                original["ordinal_predictions"][0] = invalid
                with self.assertRaisesRegex(ValueError, "finite"):
                    select_validation_fusion(original, [0.0], metrics, selector())
        original = result([0.0], [0.0], [0.0])
        original["labels"] = torch.zeros(2, 1)
        with self.assertRaisesRegex(ValueError, "shapes"):
            select_validation_fusion(original, [0.0], metrics, selector())

    def test_candidate_validation(self):
        self.assertIsNone(validate_rho_candidates(None))
        self.assertEqual(validate_rho_candidates([0.3, 0.0]), [0.0, 0.3])
        for invalid in ([], 0.3, [0.0, 0.0], [-0.1], [1.1], [float("nan")], [float("inf")]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_rho_candidates(invalid)

    def test_rho_resolution_precedence_and_legacy_fallback(self):
        self.assertEqual(resolve_inference_rho(0.3), 0.3)
        self.assertEqual(resolve_inference_rho(0.3, {"selected_epoch": 70}), 0.3)
        self.assertEqual(resolve_inference_rho(0.3, {"inference_rho": 0.0}), 0.0)
        self.assertEqual(resolve_inference_rho(0.3, {"inference_rho": 0.1}, 0.0), 0.0)
        for invalid in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                resolve_inference_rho(0.3, {"inference_rho": invalid})


if __name__ == "__main__":
    unittest.main()
