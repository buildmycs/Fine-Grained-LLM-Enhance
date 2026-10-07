"""
Training entry point for the dual-text ALMT variant.
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter

from core.dataset_dual import DualTextMMDataLoader
from core.fusion_selection import (
    resolve_inference_rho,
    select_validation_fusion,
    validate_rho_candidates,
)
from core.intensity_objective import build_sentiment_objective
from core.metric import MetricsTop
from core.model_selection import ValidationMetricSelector
from core.scheduler import get_scheduler
from core.utils import (
    AverageMeter,
    dict_to_namespace,
    results_recorder,
    setup_seed,
)
from models.almt_dual import build_model


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_file",
    type=str,
    default="configs/mosi_dual_c4.yaml",
)
parser.add_argument("--seed", type=int, default=-1)
parser.add_argument("--gpu_id", type=int, default=-1)
opt = parser.parse_args()
print(opt)

with open(opt.config_file, encoding="utf-8") as file:
    args = yaml.load(file, Loader=yaml.FullLoader)
args = dict_to_namespace(args)
print(args)

seed = args.base.seed if opt.seed == -1 else opt.seed
gpu_id = args.base.gpu_id if opt.gpu_id == -1 else opt.gpu_id

print("-----------------args-----------------")
print(args)
print("-------------------------------------")

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({gpu_id})")


def _update_fusion_meters(model, meters):
    for name, value in model.get_dual_text_stats().items():
        meters.setdefault(name, AverageMeter()).update(value, 1)


def _average_fusion_stats(meters):
    return {name: meter.value_avg for name, meter in meters.items()}


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _save_best_checkpoint(
    model,
    optimizer,
    epoch,
    validation_ret,
    selection_info,
    save_path,
):
    checkpoint_path = os.path.join(save_path, "best_validation_model.pth")
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "validation_results": validation_ret["results"],
            "selection": selection_info,
        },
        checkpoint_path,
    )
    print(f"Saved validation-selected checkpoint: {checkpoint_path}")
    return checkpoint_path


def _save_prediction_artifacts(
    epoch_ret,
    dataset,
    epoch,
    save_path,
    split_name,
):
    predictions = epoch_ret["predictions"].view(-1).numpy()
    labels = epoch_ret["labels"].view(-1).numpy()
    prediction_path = os.path.join(
        save_path, f"best_{split_name}_predictions.npz"
    )
    prediction_payload = {
        "predictions": predictions,
        "labels": labels,
        "epoch": np.asarray(epoch, dtype=np.int64),
    }
    if "inference_rho" in epoch_ret:
        prediction_payload["inference_rho"] = np.asarray(
            epoch_ret["inference_rho"], dtype=np.float64
        )
    for key in ("regression_predictions", "ordinal_predictions"):
        if key in epoch_ret:
            prediction_payload[key] = epoch_ret[key].view(-1).numpy()
    np.savez_compressed(prediction_path, **prediction_payload)

    csv_path = os.path.join(
        save_path, f"best_{split_name}_predictions.csv"
    )
    raw_text_llm = getattr(dataset, "raw_text_llm", None)
    regression_predictions = prediction_payload.get("regression_predictions")
    ordinal_predictions = prediction_payload.get("ordinal_predictions")
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "index",
                "id",
                "label",
                "prediction",
                "regression_prediction",
                "ordinal_prediction",
                "raw_text",
                "raw_text_llm",
            ]
        )
        for index, (label, prediction) in enumerate(zip(labels, predictions)):
            enhanced_text = "" if raw_text_llm is None else raw_text_llm[index]
            regression_prediction = (
                ""
                if regression_predictions is None
                else float(regression_predictions[index])
            )
            ordinal_prediction = (
                ""
                if ordinal_predictions is None
                else float(ordinal_predictions[index])
            )
            writer.writerow(
                [
                    index,
                    dataset.ids[index],
                    float(label),
                    float(prediction),
                    regression_prediction,
                    ordinal_prediction,
                    dataset.raw_text[index],
                    enhanced_text,
                ]
            )

    print(f"Saved best-{split_name} predictions: {prediction_path}")
    print(f"Saved readable best-{split_name} predictions: {csv_path}")
    return prediction_path, csv_path


def _save_selected_test_artifacts(
    test_ret,
    test_dataset,
    selection_info,
    save_path,
):
    epoch = selection_info["selected_epoch"]
    _save_prediction_artifacts(
        epoch_ret=test_ret,
        dataset=test_dataset,
        epoch=epoch,
        save_path=save_path,
        split_name="test",
    )

    selection_report = dict(selection_info)
    selection_report["test_results"] = test_ret["results"]
    report_path = os.path.join(save_path, "best_validation_selection.json")
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(
            _json_ready(selection_report),
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved selection report: {report_path}")


def _save_legacy_test_acc7_checkpoint(
    model,
    optimizer,
    epoch,
    test_ret,
    test_dataset,
    save_path,
):
    checkpoint_path = os.path.join(
        save_path, "best_test_acc7_oracle_model.pth"
    )
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "test_results": test_ret["results"],
            "evaluation_protocol": "legacy_test_oracle",
        },
        checkpoint_path,
    )
    _save_prediction_artifacts(
        epoch_ret=test_ret,
        dataset=test_dataset,
        epoch=epoch,
        save_path=save_path,
        split_name="test_acc7_oracle",
    )
    print(f"Saved legacy test-Acc-7 checkpoint: {checkpoint_path}")
    return checkpoint_path


def _save_legacy_test_summary(
    test_results_recorder,
    validation_results_recorder,
    best_test_acc7_epoch,
    best_test_acc7_results,
    validation_selection,
    save_path,
):
    report = {
        "evaluation_protocol": "legacy_test_oracle",
        "warning": (
            "Test was evaluated every epoch. This reproduces the original "
            "ALMT reporting logic and is not an unbiased held-out estimate."
        ),
        "best_test_acc7_epoch": best_test_acc7_epoch,
        "best_test_acc7_epoch_results": best_test_acc7_results,
        "best_test_results_one_epoch_by_test_mae": (
            test_results_recorder.best_results_one_epoch
        ),
        "best_test_results_across_all_epochs": (
            test_results_recorder.best_results_all_epochs
        ),
        "best_validation_results_one_epoch_by_validation_mae": (
            validation_results_recorder.best_results_one_epoch
        ),
        "best_validation_results_across_all_epochs": (
            validation_results_recorder.best_results_all_epochs
        ),
        "validation_selected_checkpoint": validation_selection,
    }
    report_path = os.path.join(save_path, "legacy_test_oracle_summary.json")
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(_json_ready(report), file, ensure_ascii=False, indent=2)
    print(f"Saved legacy test-oracle report: {report_path}")
    return report_path


def run_epoch(
    model, data_loader, loss_fn, metrics_fn, optimizer=None, collect_components=False
):
    training = optimizer is not None
    model.train(training)

    loss_recorder = AverageMeter()
    loss_meters = {}
    fusion_meters = {}
    predictions, labels = [], []
    component_predictions = {
        "regression_predictions": [],
        "ordinal_predictions": [],
    }

    for data in data_loader:
        visual = data["vision"].to(device)
        audio = data["audio"].to(device)
        text = data["text"].to(device)
        text_llm = data["text_llm"].to(device)
        label = data["labels"]["M"].to(device).view(-1, 1)

        if training:
            optimizer.zero_grad()
            model_output = model(
                visual,
                audio,
                text,
                text_llm,
                return_aux=loss_fn.requires_auxiliary_outputs or collect_components,
            )
            loss, loss_values = loss_fn(model_output, label)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                model_output = model(
                    visual,
                    audio,
                    text,
                    text_llm,
                    return_aux=loss_fn.requires_auxiliary_outputs or collect_components,
                )
                loss, loss_values = loss_fn(model_output, label)

        output = (
            model_output["prediction"]
            if isinstance(model_output, dict)
            else model_output
        )

        batch_size = visual.size(0)
        loss_recorder.update(loss.item(), batch_size)
        for name, value in loss_values.items():
            loss_meters.setdefault(name, AverageMeter()).update(value, batch_size)
        _update_fusion_meters(model, fusion_meters)
        predictions.append(output.detach().cpu())
        labels.append(label.detach().cpu())
        if isinstance(model_output, dict):
            component_predictions["regression_predictions"].append(
                model_output["regression_prediction"].detach().cpu()
            )
            component_predictions["ordinal_predictions"].append(
                model_output["ordinal_prediction"].detach().cpu()
            )

    prediction = torch.cat(predictions)
    label = torch.cat(labels)
    epoch_result = {
        "results": metrics_fn(prediction, label),
        "loss_recorder": loss_recorder,
        "loss_components": _average_fusion_stats(loss_meters),
        "fusion_stats": _average_fusion_stats(fusion_meters),
        "predictions": prediction,
        "labels": label,
    }
    for name, values in component_predictions.items():
        if values:
            epoch_result[name] = torch.cat(values)
    return epoch_result


def main():
    setup_seed(seed)
    log_path = os.path.join(".", "log", args.base.project_name)
    save_path = os.path.join(args.base.ckpt_root, args.base.project_name)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    model = build_model(args).to(device)
    data_loader = DualTextMMDataLoader(args, seed=seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.base.lr,
        weight_decay=args.base.weight_decay,
    )
    scheduler_warmup = get_scheduler(optimizer, args)
    loss_fn = build_sentiment_objective(
        args, data_loader["train"].dataset.labels["M"]
    ).to(device)
    print("-----------------objective-----------------")
    print(loss_fn.describe())
    print("-------------------------------------------")
    metrics_fn = MetricsTop().getMetics(args.dataset.datasetName)

    writer = SummaryWriter(logdir=log_path)

    evaluation_protocol = getattr(
        args.base, "evaluation_protocol", "validation_selected"
    ).lower()
    supported_protocols = {"validation_selected", "legacy_test_oracle"}
    if evaluation_protocol not in supported_protocols:
        raise ValueError(
            "evaluation_protocol must be one of "
            f"{sorted(supported_protocols)}, got '{evaluation_protocol}'"
        )
    legacy_test_oracle = evaluation_protocol == "legacy_test_oracle"
    rho_candidates = validate_rho_candidates(
        getattr(args.base, "validation_rho_candidates", None)
    )
    if rho_candidates is not None:
        if legacy_test_oracle:
            raise ValueError(
                "validation_rho_candidates requires validation_selected protocol"
            )
        if not model.use_intensity_objective:
            raise ValueError(
                "validation_rho_candidates requires the ordinal prediction head"
            )
    training_rho = model.ordinal_prediction_weight
    selected_rho = training_rho
    training_results_recorder = results_recorder() if legacy_test_oracle else None
    validation_results_recorder = (
        results_recorder() if legacy_test_oracle else None
    )
    test_results_recorder = results_recorder() if legacy_test_oracle else None
    best_test_acc7 = None
    best_test_acc7_epoch = None
    best_test_acc7_results = None

    selection_metric = getattr(args.base, "selection_metric", "MAE")
    default_mode = "min" if selection_metric == "MAE" else "max"
    selector = ValidationMetricSelector(
        primary_metric=selection_metric,
        primary_mode=getattr(args.base, "selection_mode", default_mode),
        secondary_metric=getattr(
            args.base, "selection_secondary_metric", None
        ),
        secondary_mode=getattr(
            args.base, "selection_secondary_mode", "min"
        ),
    )
    print("-----------------selection-----------------")
    if rho_candidates is not None:
        print(f"Validation-only inference rho candidates: {rho_candidates}")
        print(f"Training rho remains fixed at {training_rho}")
    else:
        print(f"Checkpoint selection uses fixed training rho={training_rho}; no rho grid.")
    print(
        f"Primary: validation {selector.primary_metric} "
        f"({selector.primary_mode})"
    )
    if selector.secondary_metric is not None:
        print(
            f"Tie-breaker: validation {selector.secondary_metric} "
            f"({selector.secondary_mode})"
        )
    if legacy_test_oracle:
        print(
            "Protocol: legacy_test_oracle. Test is evaluated every epoch to "
            "reproduce the original ALMT reporting logic."
        )
    else:
        print(
            "Protocol: validation_selected. Test is evaluated once after "
            "training using the selected checkpoint."
        )
    print("-------------------------------------------")

    if args.base.n_epochs < 1:
        raise ValueError("n_epochs must be at least 1")
    checkpoint_path = os.path.join(save_path, "best_validation_model.pth")

    for epoch in range(1, args.base.n_epochs + 1):
        loss_fn.set_epoch(epoch)
        training_ret = run_epoch(
            model, data_loader["train"], loss_fn, metrics_fn, optimizer
        )
        validation_ret = run_epoch(
            model, data_loader["valid"], loss_fn, metrics_fn,
            collect_components=rho_candidates is not None,
        )
        test_ret = (
            run_epoch(model, data_loader["test"], loss_fn, metrics_fn)
            if legacy_test_oracle
            else None
        )

        selection_ret = select_validation_fusion(
            validation_ret, rho_candidates, metrics_fn, selector
        )
        if selector.consider(epoch, selection_ret):
            selection_info = selector.as_dict()
            if rho_candidates is not None:
                selected_rho = selection_ret["inference_rho"]
                selection_info.update(
                    {
                        "selection_split": "validation",
                        "training_rho": training_rho,
                        "inference_rho": selected_rho,
                        "validation_rho_candidates": rho_candidates,
                        "training_rho_validation_results": validation_ret["results"],
                    }
                )
            _save_best_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_ret=selection_ret,
                selection_info=selection_info,
                save_path=save_path,
            )
            _save_prediction_artifacts(
                epoch_ret=selection_ret,
                dataset=data_loader["valid"].dataset,
                epoch=epoch,
                save_path=save_path,
                split_name="validation",
            )

        if legacy_test_oracle:
            training_results_recorder.update(training_ret["results"], epoch)
            validation_results_recorder.update(
                validation_ret["results"], epoch
            )
            test_results_recorder.update(test_ret["results"], epoch)
            current_test_acc7 = test_ret["results"]["Mult_acc_7"]
            if best_test_acc7 is None or current_test_acc7 > best_test_acc7:
                best_test_acc7 = current_test_acc7
                best_test_acc7_epoch = epoch
                best_test_acc7_results = dict(test_ret["results"])
                _save_legacy_test_acc7_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    test_ret=test_ret,
                    test_dataset=data_loader["test"].dataset,
                    save_path=save_path,
                )

        print(f"\n----------------- Results Epoch {epoch} -----------------")
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]}')
        print(f'Training Results: {training_ret["results"]}')
        print(f'Validation Results: {validation_ret["results"]}')
        if rho_candidates is not None:
            print(
                f'Validation Inference Results (rho={selection_ret["inference_rho"]}): '
                f'{selection_ret["results"]}'
            )
            print(f"Selected inference rho: {selected_rho}")
        if legacy_test_oracle:
            best_validation_results = (
                validation_results_recorder.get_best_results()
            )
            best_test_results = test_results_recorder.get_best_results()
            print(f'Test Results: {test_ret["results"]}')
            print(
                "Best Validation Results across All Epochs: "
                f"{best_validation_results['best_results_all_epochs']}"
            )
            print(
                "Best Validation Results of One Epoch: "
                f"{best_validation_results['best_results_one_epoch']}"
            )
            print(
                "Best Test Results across All Epochs: "
                f"{best_test_results['best_results_all_epochs']}"
            )
            print(
                "Best Test Results of One Epoch: "
                f"{best_test_results['best_results_one_epoch']}"
            )
            print(
                f"Best Test Acc-7 Epoch: {best_test_acc7_epoch} | "
                f"Acc-7={best_test_acc7:.4f} | "
                f"Results={best_test_acc7_results}"
            )
        print(f'Training Objective: {training_ret["loss_components"]}')
        if training_ret["fusion_stats"]:
            print(f'Dual-text Fusion: {training_ret["fusion_stats"]}')
        selected = selector.as_dict()
        print(
            f"Selected Epoch: {selected['selected_epoch']} | "
            f"validation {selected['primary_metric']}="
            f"{selected['primary_value']:.6f}"
        )
        if selected["secondary_metric"] is not None:
            print(
                f"Tie-breaker validation {selected['secondary_metric']}="
                f"{selected['secondary_value']:.6f}"
            )
        print("----------------------------------------------------------\n")

        writer.add_scalar(
            "train/loss", training_ret["loss_recorder"].value_avg, epoch
        )
        writer.add_scalar(
            "valid/loss", validation_ret["loss_recorder"].value_avg, epoch
        )
        writer.add_scalar(
            "train/learning_rate", optimizer.param_groups[0]["lr"], epoch
        )
        for name, value in validation_ret["results"].items():
            writer.add_scalar(f"valid/{name}", value, epoch)
        if rho_candidates is not None:
            for name, value in selection_ret["results"].items():
                writer.add_scalar(f"valid_selection/{name}", value, epoch)
            writer.add_scalar(
                "valid_selection/rho", selection_ret["inference_rho"], epoch
            )
            writer.add_scalar("valid_selection/best_rho", selected_rho, epoch)
        for prediction_name in (
            "regression_predictions",
            "ordinal_predictions",
        ):
            if prediction_name not in validation_ret:
                continue
            component_name = prediction_name.removesuffix("_predictions")
            component_results = metrics_fn(
                validation_ret[prediction_name],
                validation_ret["labels"],
            )
            for name, value in component_results.items():
                writer.add_scalar(
                    f"valid_{component_name}/{name}", value, epoch
                )
        if legacy_test_oracle:
            writer.add_scalar(
                "test/loss", test_ret["loss_recorder"].value_avg, epoch
            )
            for name, value in test_ret["results"].items():
                writer.add_scalar(f"test/{name}", value, epoch)
        for name, value in training_ret["fusion_stats"].items():
            writer.add_scalar(f"dual_text/{name}", value, epoch)
        for name, value in training_ret["loss_components"].items():
            writer.add_scalar(f"objective/{name}", value, epoch)

        ordinal_thresholds = model.get_ordinal_thresholds()
        if ordinal_thresholds is not None:
            for index, value in enumerate(ordinal_thresholds.tolist()):
                writer.add_scalar(f"ordinal/threshold_{index}", value, epoch)

        scheduler_warmup.step()

    if legacy_test_oracle:
        _save_legacy_test_summary(
            test_results_recorder=test_results_recorder,
            validation_results_recorder=validation_results_recorder,
            best_test_acc7_epoch=best_test_acc7_epoch,
            best_test_acc7_results=best_test_acc7_results,
            validation_selection=selector.as_dict(),
            save_path=save_path,
        )
        print("\n---------------- Legacy ALMT Test Oracle ----------------")
        print(
            "This block reproduces the original train.py test-every-epoch "
            "reporting protocol."
        )
        print(f"Best Test Acc-7 Epoch: {best_test_acc7_epoch}")
        print(f"Best Test Acc-7 Epoch Results: {best_test_acc7_results}")
        print(
            "Best Test Results across All Epochs: "
            f"{test_results_recorder.best_results_all_epochs}"
        )
        print(
            "Best Test Results of One Epoch (minimum test MAE): "
            f"{test_results_recorder.best_results_one_epoch}"
        )
        print("---------------------------------------------------------\n")
        writer.close()
        return

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    selection_info = checkpoint["selection"]
    selected_epoch = selection_info["selected_epoch"]
    model.ordinal_prediction_weight = resolve_inference_rho(
        training_rho, selection_info
    )
    loss_fn.set_epoch(selected_epoch)
    test_ret = run_epoch(
        model, data_loader["test"], loss_fn, metrics_fn,
        collect_components=rho_candidates is not None,
    )
    if rho_candidates is not None:
        test_ret["inference_rho"] = model.ordinal_prediction_weight
    _save_selected_test_artifacts(
        test_ret=test_ret,
        test_dataset=data_loader["test"].dataset,
        selection_info=selection_info,
        save_path=save_path,
    )
    writer.add_scalar(
        "test_selected/loss", test_ret["loss_recorder"].value_avg, selected_epoch
    )
    for name, value in test_ret["results"].items():
        writer.add_scalar(f"test_selected/{name}", value, selected_epoch)

    print("\n----------------- Final Selected Test -----------------")
    print(
        f"Checkpoint selected by validation {selector.primary_metric} "
        f"at epoch {selected_epoch}"
    )
    print(f"Validation Results: {selector.selected_validation_results}")
    print(f"Inference rho: {model.ordinal_prediction_weight}")
    print(f'Test Results: {test_ret["results"]}')
    print("-------------------------------------------------------\n")

    writer.close()


if __name__ == "__main__":
    main()
