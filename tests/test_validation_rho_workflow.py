"""Exercise real checkpoint/artifact I/O without BERT, a GPU, or MOSEI data."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import yaml

from core.model_selection import exact_metric_value
from scripts import evaluate_selected_test


ROOT = Path(__file__).resolve().parents[1]


def metrics(predictions, labels):
    result = {"predictions": predictions, "labels": labels}
    return {
        name: round(exact_metric_value(result, name), 4)
        for name in ("Mult_acc_7", "MAE")
    }


def epoch_result(epoch, rho):
    regression = torch.tensor([[0.0], [1.0]]) if epoch == 1 else torch.tensor([[0.6], [1.0]])
    ordinal = torch.tensor([[2.0], [3.0]]) if epoch == 1 else regression.clone()
    labels = torch.tensor([[0.0], [1.0]])
    predictions = (1 - rho) * regression + rho * ordinal
    return {
        "predictions": predictions,
        "regression_predictions": regression,
        "ordinal_predictions": ordinal,
        "labels": labels,
        "results": metrics(predictions, labels),
        "loss_recorder": SimpleNamespace(value_avg=0.0),
        "loss_components": {},
        "fusion_stats": {},
    }


class TinyModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.tensor(0.0))
        self.ordinal_prediction_weight = args.model.ordinal_prediction_weight
        self.use_intensity_objective = True

    def get_ordinal_thresholds(self):
        return None


class ValidationRhoWorkflowTest(unittest.TestCase):
    def run_workflow(
        self, candidates, expected_epoch, expected_rho, use_default_config=False
    ):
        dataset = SimpleNamespace(
            labels={"M": np.array([0.0, 1.0])},
            ids=["sample-a", "sample-b"],
            raw_text=["a", "b"],
        )
        loaders = {
            split: SimpleNamespace(dataset=dataset, split=split)
            for split in ("train", "valid", "test")
        }
        stubs = {
            "tensorboardX": SimpleNamespace(SummaryWriter=MagicMock()),
            "core.dataset_dual": SimpleNamespace(
                DualTextMMDataLoader=lambda *args, **kwargs: loaders,
                DualTextMMDataset=lambda *args, **kwargs: dataset,
            ),
            "models.almt_dual": SimpleNamespace(build_model=TinyModel),
            "core.metric": SimpleNamespace(
                MetricsTop=lambda: SimpleNamespace(getMetics=lambda name: metrics)
            ),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(ROOT / "configs/mosei_dual_c4_intensity.yaml", encoding="utf-8") as file:
                config = yaml.safe_load(file)
            config["base"].update(ckpt_root=temp_dir, project_name="smoke", n_epochs=2)
            if not use_default_config:
                if candidates is None:
                    config["base"].pop("validation_rho_candidates", None)
                else:
                    config["base"]["validation_rho_candidates"] = candidates
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            spec = importlib.util.spec_from_file_location("_test_train_dual", ROOT / "train_dual.py")
            training = importlib.util.module_from_spec(spec)
            with (
                patch.dict(sys.modules, stubs),
                patch.object(sys, "argv", ["train_dual.py", "--config_file", str(config_path)]),
                patch.dict("os.environ"),
                patch("torch.cuda.is_available", return_value=False),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.chdir(temp_dir),
            ):
                spec.loader.exec_module(training)
                calls = []
                epoch = 0

                def fake_run_epoch(model, loader, loss_fn, metrics_fn, optimizer=None, collect_components=False):
                    nonlocal epoch
                    if optimizer is not None:
                        epoch += 1
                        with torch.no_grad():
                            model.marker.fill_(epoch)
                    calls.append((loader.split, model.ordinal_prediction_weight))
                    if loader.split == "test":
                        self.assertEqual(epoch, 2)
                        self.assertEqual(int(model.marker.item()), expected_epoch)
                        self.assertEqual(model.ordinal_prediction_weight, expected_rho)
                    return epoch_result(int(model.marker.item()), model.ordinal_prediction_weight)

                objective = SimpleNamespace(describe=lambda: {}, set_epoch=lambda epoch: None)
                with (
                    patch.object(training, "run_epoch", side_effect=fake_run_epoch),
                    patch.object(training, "build_sentiment_objective", return_value=SimpleNamespace(to=lambda device: objective)),
                    patch.object(training, "get_scheduler", return_value=SimpleNamespace(step=lambda: None)),
                ):
                    training.main()

                self.assertEqual([rho for split, rho in calls if split == "train"], [0.3, 0.3])
                self.assertEqual([split for split, _ in calls], ["train", "valid", "train", "valid", "test"])
                run_dir = Path(temp_dir) / "smoke"
                checkpoint = torch.load(run_dir / "best_validation_model.pth", weights_only=False)
                self.assertEqual(checkpoint["epoch"], expected_epoch)
                self.assertEqual(checkpoint["selection"]["selected_epoch"], expected_epoch)
                if candidates is not None:
                    self.assertEqual(checkpoint["selection"]["inference_rho"], expected_rho)
                    self.assertEqual(checkpoint["selection"]["training_rho"], 0.3)
                else:
                    self.assertNotIn("inference_rho", checkpoint["selection"])
                expected = epoch_result(expected_epoch, expected_rho)["predictions"].view(-1).numpy()
                for split in ("validation", "test"):
                    with np.load(run_dir / f"best_{split}_predictions.npz") as data:
                        self.assertEqual(int(data["epoch"]), expected_epoch)
                        np.testing.assert_allclose(data["predictions"], expected)
                        if candidates is not None:
                            self.assertEqual(float(data["inference_rho"]), expected_rho)
                report = json.loads((run_dir / "best_validation_selection.json").read_text(encoding="utf-8"))
                self.assertEqual(report["validation_results"], checkpoint["validation_results"])
                self.assertEqual(report["test_results"], metrics(torch.from_numpy(expected), torch.tensor([0.0, 1.0])))

                # Independent test entry point must also use saved rho, even when it is zero.
                def fake_evaluate(model, *args):
                    self.assertEqual(model.ordinal_prediction_weight, expected_rho)
                    self.assertEqual(int(model.marker.item()), expected_epoch)
                    return epoch_result(expected_epoch, model.ordinal_prediction_weight)

                with (
                    patch.object(sys, "argv", ["evaluate_selected_test.py", "--config_file", str(config_path)]),
                    patch.object(evaluate_selected_test, "DataLoader", return_value=loaders["test"]),
                    patch.object(evaluate_selected_test, "evaluate", side_effect=fake_evaluate) as evaluation,
                ):
                    evaluate_selected_test.main()
                self.assertEqual(evaluation.call_count, 1)
                report = json.loads((run_dir / "selected_test_results.json").read_text(encoding="utf-8"))
                self.assertEqual(report["inference_ordinal_prediction_weight"], expected_rho)

    def test_validation_grid_saves_and_reloads_earlier_rho_zero_checkpoint(self):
        self.run_workflow([0.0, 0.1, 0.2, 0.3], expected_epoch=1, expected_rho=0.0)

    def test_legacy_fixed_rho_behavior_is_preserved(self):
        self.run_workflow(None, expected_epoch=2, expected_rho=0.3)

    def test_default_mosei_config_uses_fixed_rho_without_grid(self):
        self.run_workflow(
            None, expected_epoch=2, expected_rho=0.3, use_default_config=True
        )

    def test_rollback_preserves_training_settings_and_uses_a_distinct_project(self):
        configs = []
        for filename in (
            "mosei_dual_c4_intensity_rho030_ord020_fixedselect_lr2e-5.yaml",
            "mosei_dual_c4_intensity_rho030_lr2e-5.yaml",
            "mosei_dual_c4_intensity_rho030_valrho_lr2e-5.yaml",
        ):
            with open(ROOT / "configs" / filename, encoding="utf-8") as file:
                configs.append(yaml.safe_load(file))
        current, fixed, grid = configs
        self.assertIsNone(current["base"]["validation_rho_candidates"])
        self.assertEqual(grid["base"]["validation_rho_candidates"], [0.0, 0.1, 0.2, 0.3])
        names = [config["base"].pop("project_name") for config in configs]
        self.assertEqual(len(set(names)), 3)
        for config in configs:
            config["base"].pop("validation_rho_candidates", None)
        self.assertEqual(current, fixed)
        self.assertEqual(current, grid)

    def test_ordinal015_trial_changes_only_loss_weight_and_project_name(self):
        configs = []
        for filename in (
            "mosei_dual_c4_intensity.yaml",
            "mosei_dual_c4_intensity_rho030_ord020_fixedselect_lr2e-5.yaml",
        ):
            with open(ROOT / "configs" / filename, encoding="utf-8") as file:
                configs.append(yaml.safe_load(file))
        trial, baseline = configs
        self.assertIsNone(trial["base"]["validation_rho_candidates"])
        self.assertEqual(trial["objective"]["ordinal_weight"], 0.15)
        self.assertEqual(baseline["objective"]["ordinal_weight"], 0.2)
        self.assertNotEqual(
            trial["base"].pop("project_name"),
            baseline["base"].pop("project_name"),
        )
        trial["objective"]["ordinal_weight"] = 0.2
        self.assertEqual(trial, baseline)


if __name__ == "__main__":
    unittest.main()
