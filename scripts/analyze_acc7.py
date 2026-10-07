"""
Generate the exact ALMT Acc-7 confusion matrix and per-class metrics.

The input can be the NPZ or CSV produced by train_dual.py. Acc-7 follows
core/metric.py exactly: clip continuous values to [-3, 3], then np.round.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CLASSES = np.arange(-3, 4, dtype=np.int64)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze seven-level sentiment predictions."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="NPZ or CSV containing predictions and labels.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <prediction_stem>_acc7_analysis.",
    )
    parser.add_argument(
        "--title",
        default="Acc-7 Confusion Matrix",
        help="Title used in the generated figure.",
    )
    parser.add_argument(
        "--prediction-key",
        default=None,
        help=(
            "Prediction array/column to analyze. Defaults to predictions for "
            "NPZ and prediction for CSV."
        ),
    )
    parser.add_argument(
        "--label-key",
        default=None,
        help="Label array/column. Usually inferred automatically.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _find_npz_key(data, candidates):
    for key in candidates:
        if key in data:
            return key
    raise KeyError(
        f"None of {candidates} were found. Available keys: {list(data.keys())}"
    )


def load_predictions(path, requested_prediction_key=None, requested_label_key=None):
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            prediction_key = requested_prediction_key or _find_npz_key(
                data, ("predictions", "prediction", "y_pred", "pred")
            )
            label_key = requested_label_key or _find_npz_key(
                data, ("labels", "label", "y_true", "truth")
            )
            if prediction_key not in data or label_key not in data:
                raise KeyError(
                    f"Requested keys {prediction_key!r}, {label_key!r}; "
                    f"available keys: {list(data.keys())}"
                )
            predictions = np.asarray(data[prediction_key], dtype=np.float64)
            labels = np.asarray(data[label_key], dtype=np.float64)
    elif suffix == ".csv":
        predictions, labels = [], []
        with open(path, encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames or []
            prediction_key = requested_prediction_key or next(
                (
                    key
                    for key in ("prediction", "predictions", "y_pred", "pred")
                    if key in fieldnames
                ),
                None,
            )
            label_key = requested_label_key or next(
                (
                    key
                    for key in ("label", "labels", "y_true", "truth")
                    if key in fieldnames
                ),
                None,
            )
            if prediction_key is None or label_key is None:
                raise KeyError(
                    "CSV must contain prediction and label columns; "
                    f"found {fieldnames}"
                )
            for row in reader:
                predictions.append(float(row[prediction_key]))
                labels.append(float(row[label_key]))
        predictions = np.asarray(predictions, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
    else:
        raise ValueError("--predictions must be an .npz or .csv file")

    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)
    if predictions.size == 0:
        raise ValueError("Prediction file is empty")
    if predictions.shape != labels.shape:
        raise ValueError(
            f"Predictions shape {predictions.shape} does not match "
            f"labels shape {labels.shape}"
        )
    if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(labels)):
        raise ValueError("Predictions and labels must contain only finite values")
    return predictions, labels


def to_acc7_class(values):
    # This intentionally uses np.round to stay identical to core/metric.py.
    return np.round(np.clip(values, -3.0, 3.0)).astype(np.int64)


def build_confusion_matrix(true_classes, predicted_classes):
    matrix = np.zeros((7, 7), dtype=np.int64)
    true_indices = true_classes + 3
    predicted_indices = predicted_classes + 3
    np.add.at(matrix, (true_indices, predicted_indices), 1)
    return matrix


def safe_divide(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def compute_metrics(matrix):
    true_support = matrix.sum(axis=1)
    predicted_support = matrix.sum(axis=0)
    correct = np.diag(matrix)
    recall = safe_divide(correct, true_support)
    precision = safe_divide(correct, predicted_support)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    total = matrix.sum()
    accuracy = float(correct.sum() / total)
    valid_classes = true_support > 0
    macro_recall = float(recall[valid_classes].mean())
    macro_f1 = float(f1[valid_classes].mean())
    return {
        "support": true_support,
        "predicted_support": predicted_support,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def save_matrix_csv(path, matrix, number_format="%d"):
    header = "true\\pred," + ",".join(str(value) for value in CLASSES)
    rows = np.column_stack((CLASSES, matrix))
    if np.issubdtype(matrix.dtype, np.floating):
        formats = ["%d"] + [number_format] * matrix.shape[1]
    else:
        formats = ["%d"] * (matrix.shape[1] + 1)
    np.savetxt(
        path,
        rows,
        delimiter=",",
        header=header,
        comments="",
        fmt=formats,
    )


def save_per_class_csv(path, metrics):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["class", "recall", "precision", "f1", "support", "predicted"]
        )
        for index, class_value in enumerate(CLASSES):
            writer.writerow(
                [
                    int(class_value),
                    float(metrics["recall"][index]),
                    float(metrics["precision"][index]),
                    float(metrics["f1"][index]),
                    int(metrics["support"][index]),
                    int(metrics["predicted_support"][index]),
                ]
            )


def save_sample_diagnostics(
    path, predictions, labels, predicted_classes, true_classes
):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "index",
                "label",
                "prediction",
                "true_class",
                "predicted_class",
                "correct_acc7",
                "absolute_error",
                "class_distance",
            ]
        )
        for index in range(len(labels)):
            writer.writerow(
                [
                    index,
                    float(labels[index]),
                    float(predictions[index]),
                    int(true_classes[index]),
                    int(predicted_classes[index]),
                    int(true_classes[index] == predicted_classes[index]),
                    float(abs(labels[index] - predictions[index])),
                    int(abs(true_classes[index] - predicted_classes[index])),
                ]
            )


def plot_confusion_matrices(path, counts, normalized, metrics, title, dpi):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Warning: matplotlib is not installed; CSV/JSON reports were saved "
            "but the PNG was skipped. Install it with: pip install matplotlib"
        )
        return False

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    count_maximum = max(float(counts.max()), 1.0)
    panels = (
        (counts, "Counts", "d", count_maximum),
        (normalized * 100.0, "Row-normalized (%)", ".1f", 100.0),
    )
    for axis, (matrix, subtitle, value_format, maximum) in zip(axes, panels):
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        axis.set_xticks(range(7), labels=CLASSES)
        axis.set_yticks(range(7), labels=CLASSES)
        axis.set_xlabel("Predicted sentiment level")
        axis.set_ylabel("True sentiment level")
        axis.set_title(subtitle)
        threshold = maximum * 0.5
        for row in range(7):
            for column in range(7):
                value = matrix[row, column]
                text = format(value, value_format)
                axis.text(
                    column,
                    row,
                    text,
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                    fontsize=9,
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.suptitle(
        f"{title} | Acc-7={metrics['accuracy']:.4f}, "
        f"Macro Recall={metrics['macro_recall']:.4f}",
        fontsize=14,
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return True


def main():
    args = parse_args()
    prediction_path = Path(args.predictions)
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else prediction_path.parent / f"{prediction_path.stem}_acc7_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions, labels = load_predictions(
        prediction_path,
        requested_prediction_key=args.prediction_key,
        requested_label_key=args.label_key,
    )
    predicted_classes = to_acc7_class(predictions)
    true_classes = to_acc7_class(labels)
    counts = build_confusion_matrix(true_classes, predicted_classes)
    row_support = counts.sum(axis=1, keepdims=True)
    normalized = safe_divide(counts, row_support)
    metrics = compute_metrics(counts)

    save_matrix_csv(output_dir / "confusion_counts.csv", counts)
    save_matrix_csv(
        output_dir / "confusion_row_normalized.csv",
        normalized,
        number_format="%.6f",
    )
    save_per_class_csv(output_dir / "per_class_metrics.csv", metrics)
    save_sample_diagnostics(
        output_dir / "sample_diagnostics.csv",
        predictions,
        labels,
        predicted_classes,
        true_classes,
    )

    correlation = (
        float(np.corrcoef(predictions, labels)[0, 1])
        if len(predictions) > 1
        else None
    )
    if correlation is not None and not np.isfinite(correlation):
        correlation = None
    summary = {
        "prediction_file": str(prediction_path),
        "sample_count": int(len(labels)),
        "acc7": metrics["accuracy"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "mae": float(np.mean(np.abs(predictions - labels))),
        "correlation": correlation,
        "rounding": "np.round(np.clip(value, -3, 3))",
        "per_class": {
            str(class_value): {
                "recall": float(metrics["recall"][index]),
                "precision": float(metrics["precision"][index]),
                "f1": float(metrics["f1"][index]),
                "support": int(metrics["support"][index]),
                "predicted": int(metrics["predicted_support"][index]),
            }
            for index, class_value in enumerate(CLASSES)
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    png_path = output_dir / "acc7_confusion_matrix.png"
    image_saved = plot_confusion_matrices(
        png_path, counts, normalized, metrics, args.title, args.dpi
    )

    print(f"Samples: {len(labels)}")
    print(f"Acc-7: {metrics['accuracy']:.4f}")
    print(f"Macro recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("Per-class recall:")
    for index, class_value in enumerate(CLASSES):
        print(
            f"  {class_value:+d}: {metrics['recall'][index]:.4f} "
            f"(support={metrics['support'][index]})"
        )
    print(f"Reports saved to: {output_dir}")
    if image_saved:
        print(f"Figure saved to: {png_path}")


if __name__ == "__main__":
    main()
