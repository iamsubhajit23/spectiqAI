"""Training utilities for PyTorch classification models."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader


def set_seed(seed: int = 42) -> None:
    """Make training behavior more reproducible."""
    random.seed(seed)
    torch.manual_seed(seed)


def count_correct_predictions(logits: torch.Tensor, labels: torch.Tensor) -> int:
    """Count how many predictions match the true labels."""
    predictions = torch.argmax(logits, dim=1)
    return int((predictions == labels).sum().item())


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_index, (images, labels) in enumerate(data_loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += count_correct_predictions(logits, labels)
        total_examples += batch_size

        if batch_index % 25 == 0:
            print(f"  Trained {batch_index} batches...")

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate the model without updating weights."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_index, (images, labels) in enumerate(data_loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += count_correct_predictions(logits, labels)
        total_examples += batch_size

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_path: Path,
) -> None:
    """Save model weights and training state."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )


def save_history_csv(history: Iterable[Dict[str, float]], output_path: Path) -> None:
    """Save training history to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(history)
    if not rows:
        raise ValueError("Cannot save empty training history.")

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def create_history_plot(history: List[Dict[str, float]], output_path: Path) -> Image.Image:
    """Create a simple loss/accuracy curve image using Pillow."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 900, 420
    padding_left, padding_right = 70, 30
    padding_top, padding_bottom = 45, 60
    plot_width = width - padding_left - padding_right
    plot_height = height - padding_top - padding_bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.text((padding_left, 15), "Training and Validation Curves", fill="black")
    draw.rectangle(
        [
            (padding_left, padding_top),
            (padding_left + plot_width, padding_top + plot_height),
        ],
        outline="black",
    )

    epochs = [int(row["epoch"]) for row in history]
    if len(epochs) == 1:
        x_positions = [padding_left + plot_width // 2]
    else:
        x_positions = [
            padding_left + round(index * plot_width / (len(epochs) - 1))
            for index in range(len(epochs))
        ]

    metric_specs = [
        ("train_loss", "red"),
        ("val_loss", "orange"),
        ("train_accuracy", "blue"),
        ("val_accuracy", "green"),
    ]

    values = [float(row[key]) for row in history for key, _ in metric_specs]
    min_value = min(values)
    max_value = max(values)
    value_range = max(max_value - min_value, 1e-6)

    def y_for(value: float) -> int:
        scaled = (value - min_value) / value_range
        return padding_top + plot_height - round(scaled * plot_height)

    for metric_name, color in metric_specs:
        points = [
            (x_positions[index], y_for(float(row[metric_name])))
            for index, row in enumerate(history)
        ]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        else:
            draw.line(points, fill=color, width=3)
            for x, y in points:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)

    legend_x = padding_left + 10
    legend_y = padding_top + 10
    for index, (metric_name, color) in enumerate(metric_specs):
        y = legend_y + index * 22
        draw.rectangle((legend_x, y + 4, legend_x + 12, y + 16), fill=color)
        draw.text((legend_x + 18, y), metric_name, fill="black")

    draw.text((padding_left, height - 35), "Epoch", fill="black")
    draw.text((10, padding_top), f"max {max_value:.3f}", fill="black")
    draw.text((10, padding_top + plot_height - 15), f"min {min_value:.3f}", fill="black")

    image.save(output_path)
    return image
