"""Evaluation utilities for classification models."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int], List[float]]:
    """Collect true labels, predicted labels, and tumor probabilities."""
    model.eval()
    true_labels: List[int] = []
    predicted_labels: List[int] = []
    tumor_probabilities: List[float] = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        true_labels.extend(labels.cpu().tolist())
        predicted_labels.extend(predictions.cpu().tolist())
        tumor_probabilities.extend(probabilities[:, 1].cpu().tolist())

    return true_labels, predicted_labels, tumor_probabilities


def confusion_counts(true_labels: Sequence[int], predicted_labels: Sequence[int]) -> Dict[str, int]:
    """Return binary confusion matrix counts using Tumor as the positive class."""
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have the same length.")

    true_positive = true_negative = false_positive = false_negative = 0

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if true_label == 1 and predicted_label == 1:
            true_positive += 1
        elif true_label == 0 and predicted_label == 0:
            true_negative += 1
        elif true_label == 0 and predicted_label == 1:
            false_positive += 1
        elif true_label == 1 and predicted_label == 0:
            false_negative += 1
        else:
            raise ValueError(f"Unexpected label pair: true={true_label}, predicted={predicted_label}")

    return {
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def safe_divide(numerator: float, denominator: float) -> float:
    """Avoid division-by-zero errors in metric calculations."""
    return numerator / denominator if denominator else 0.0


def binary_classification_metrics(true_labels: Sequence[int], predicted_labels: Sequence[int]) -> Dict[str, float]:
    """Compute accuracy, precision, recall, and F1 score for binary classification."""
    counts = confusion_counts(true_labels, predicted_labels)
    tp = counts["true_positive"]
    tn = counts["true_negative"]
    fp = counts["false_positive"]
    fn = counts["false_negative"]
    total = tp + tn + fp + fn

    accuracy = safe_divide(tp + tn, total)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1_score = safe_divide(2 * precision * recall, precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        **{key: float(value) for key, value in counts.items()},
    }


def save_metrics_csv(metrics: Dict[str, float], output_path: Path) -> None:
    """Save one row of evaluation metrics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def save_predictions_csv(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    tumor_probabilities: Sequence[float],
    output_path: Path,
) -> None:
    """Save per-image prediction results for later error analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["true_label", "predicted_label", "tumor_probability"])
        for true_label, predicted_label, probability in zip(true_labels, predicted_labels, tumor_probabilities):
            writer.writerow([true_label, predicted_label, probability])


def create_confusion_matrix_image(metrics: Dict[str, float], output_path: Path) -> Image.Image:
    """Create a simple confusion matrix image without extra plotting libraries."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cell_width = 210
    cell_height = 95
    left = 170
    top = 95
    width = left + 2 * cell_width + 40
    height = top + 2 * cell_height + 90

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.text((left, 20), "Confusion Matrix", fill="black")
    draw.text((left + 35, 60), "Predicted Normal", fill="black")
    draw.text((left + cell_width + 35, 60), "Predicted Tumor", fill="black")
    draw.text((25, top + 35), "Actual Normal", fill="black")
    draw.text((25, top + cell_height + 35), "Actual Tumor", fill="black")

    cells = [
        ("TN", int(metrics["true_negative"]), 0, 0, "#d9f2d9"),
        ("FP", int(metrics["false_positive"]), 1, 0, "#ffd9d9"),
        ("FN", int(metrics["false_negative"]), 0, 1, "#ffd9d9"),
        ("TP", int(metrics["true_positive"]), 1, 1, "#d9f2d9"),
    ]

    for label, value, column, row, color in cells:
        x1 = left + column * cell_width
        y1 = top + row * cell_height
        x2 = x1 + cell_width
        y2 = y1 + cell_height
        draw.rectangle((x1, y1, x2, y2), fill=color, outline="black")
        draw.text((x1 + 20, y1 + 22), label, fill="black")
        draw.text((x1 + 20, y1 + 50), str(value), fill="black")

    summary = (
        f"Accuracy: {metrics['accuracy']:.4f} | Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | F1: {metrics['f1_score']:.4f}"
    )
    draw.text((40, height - 45), summary, fill="black")

    image.save(output_path)
    return image


def compute_roc_curve(
    true_labels: Sequence[int],
    probabilities: Sequence[float],
    num_thresholds: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute True Positive Rate and False Positive Rate for multiple thresholds."""
    y_true = np.array(true_labels)
    y_prob = np.array(probabilities)
    thresholds = np.linspace(1.0, 0.0, num_thresholds)  # Scan from 1 to 0 for monotonic FPR increase
    
    tpr = []
    fpr = []
    
    pos_count = np.sum(y_true == 1)
    neg_count = np.sum(y_true == 0)
    
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        
        tpr.append(tp / pos_count if pos_count > 0 else 0.0)
        fpr.append(fp / neg_count if neg_count > 0 else 0.0)
        
    return np.array(fpr), np.array(tpr), thresholds


def compute_pr_curve(
    true_labels: Sequence[int],
    probabilities: Sequence[float],
    num_thresholds: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Precision and Recall for multiple thresholds."""
    y_true = np.array(true_labels)
    y_prob = np.array(probabilities)
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    
    precision = []
    recall = []
    
    pos_count = np.sum(y_true == 1)
    
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        
        rec = tp / pos_count if pos_count > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        
        precision.append(prec)
        recall.append(rec)
        
    return np.array(recall), np.array(precision), thresholds


def compute_auc(fpr: Sequence[float], tpr: Sequence[float]) -> float:
    """Compute Area Under the Curve using the trapezoidal rule."""
    # Ensure sorted by FPR
    sorted_indices = np.argsort(fpr)
    fpr_sorted = np.array(fpr)[sorted_indices]
    tpr_sorted = np.array(tpr)[sorted_indices]
    
    # Custom trapezoidal integration for compatibility with numpy 1.x and 2.x
    auc_val = 0.0
    for i in range(len(fpr_sorted) - 1):
        dx = fpr_sorted[i+1] - fpr_sorted[i]
        my = (tpr_sorted[i+1] + tpr_sorted[i]) / 2.0
        auc_val += dx * my
    return float(auc_val)

