"""Search the regression/ordinal fusion weight on validation predictions."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Search rho on a saved validation NPZ. Use the selected rho once "
            "on test; do not search rho on test predictions."
        )
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="best_validation_predictions.npz produced by train_dual.py.",
    )
    parser.add_argument("--min-rho", type=float, default=0.0)
    parser.add_argument("--max-rho", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <prediction directory>/rho_search_validation.",
    )
    return parser.parse_args()


def load_predictions(path):
    with np.load(path, allow_pickle=False) as data:
        required = {"regression_predictions", "ordinal_predictions", "labels"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(
                f"{path} is missing {missing}; available keys: {data.files}"
            )
        regression = data["regression_predictions"].reshape(-1)
        ordinal = data["ordinal_predictions"].reshape(-1)
        labels = data["labels"].reshape(-1)

    if regression.shape != ordinal.shape or regression.shape != labels.shape:
        raise ValueError(
            "Regression, ordinal and label arrays must have identical shapes; "
            f"got {regression.shape}, {ordinal.shape}, {labels.shape}"
        )
    arrays = (regression, ordinal, labels)
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("Prediction arrays must contain only finite values")
    return regression, ordinal, labels


def search_rho(regression, ordinal, labels, min_rho, max_rho, step):
    if step <= 0:
        raise ValueError("--step must be positive")
    if not 0.0 <= min_rho <= max_rho <= 1.0:
        raise ValueError("rho search range must satisfy 0 <= min <= max <= 1")

    true_class = np.round(np.clip(labels, -3.0, 3.0)).astype(np.int64)
    count = int(np.floor((max_rho - min_rho) / step + 1e-9)) + 1
    rhos = min_rho + np.arange(count, dtype=np.float64) * step
    if rhos[-1] < max_rho - 1e-9:
        rhos = np.append(rhos, max_rho)

    results = []
    for rho in rhos:
        fused = (1.0 - rho) * regression + rho * ordinal
        predicted_class = np.round(
            np.clip(fused, -3.0, 3.0)
        ).astype(np.int64)
        recalls = []
        for class_value in (-3, 3):
            mask = true_class == class_value
            recalls.append(
                float(np.mean(predicted_class[mask] == class_value))
                if np.any(mask)
                else 0.0
            )
        results.append(
            {
                "rho": float(rho),
                "acc7": float(np.mean(predicted_class == true_class)),
                "extreme_macro_recall": float(np.mean(recalls)),
                "mae": float(np.mean(np.abs(fused - labels))),
            }
        )

    # Preserve the previous script's tie order: Acc-7, extreme recall,
    # lower MAE, then larger rho.
    return sorted(
        results,
        key=lambda item: (
            item["acc7"],
            item["extreme_macro_recall"],
            -item["mae"],
            item["rho"],
        ),
        reverse=True,
    )


def save_results(results, predictions_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "rho_search_all.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    report_path = output_dir / "rho_search_summary.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "selection_split": "validation",
                "predictions": str(predictions_path),
                "best": results[0],
                "candidate_count": len(results),
                "warning": (
                    "Apply the selected rho once on test; never search test rho."
                ),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    return csv_path, report_path


def main():
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    predictions_path = Path(args.predictions)
    if "validation" not in predictions_path.name.lower():
        print(
            "Warning: the filename does not contain 'validation'. Confirm that "
            "rho is being selected on validation rather than test."
        )
    regression, ordinal, labels = load_predictions(predictions_path)
    results = search_rho(
        regression,
        ordinal,
        labels,
        args.min_rho,
        args.max_rho,
        args.step,
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else predictions_path.parent / "rho_search_validation"
    )
    csv_path, report_path = save_results(results, predictions_path, output_dir)

    best = results[0]
    print(f"best rho: {best['rho']:.6f}")
    print(f"validation Acc-7: {best['acc7']:.6f}")
    print(f"extreme macro recall: {best['extreme_macro_recall']:.6f}")
    print(f"validation MAE: {best['mae']:.6f}")
    print(f"\nTop {min(args.top_k, len(results))}:")
    for item in results[: args.top_k]:
        print(
            f"rho={item['rho']:.2f}, Acc7={item['acc7']:.4f}, "
            f"extreme_recall={item['extreme_macro_recall']:.4f}, "
            f"MAE={item['mae']:.4f}"
        )
    print(f"\nAll candidates: {csv_path}")
    print(f"Selection report: {report_path}")


if __name__ == "__main__":
    main()
