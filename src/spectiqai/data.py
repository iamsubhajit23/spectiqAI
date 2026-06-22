"""Data utilities for the kidney tumor classification project."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_TO_LABEL = {"Normal": 0, "Tumor": 1}
LABEL_TO_CLASS = {label: class_name for class_name, label in CLASS_TO_LABEL.items()}
CLASS_DIRS = {
    "Normal": "Hyperspectral_normal_images",
    "Tumor": "Hyperspectral_tumor_images",
}


@dataclass(frozen=True)
class ImageRecord:
    """One image path with its human-readable class name and numeric label."""

    path: Path
    class_name: str
    label: int


def validate_dataset_dir(dataset_dir: Path) -> Dict[str, Path]:
    """Validate the expected dataset folder structure and return class paths."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    class_paths: Dict[str, Path] = {}
    for class_name, folder_name in CLASS_DIRS.items():
        class_path = dataset_dir / folder_name
        if not class_path.exists():
            raise FileNotFoundError(f"Missing {class_name} folder: {class_path}")
        class_paths[class_name] = class_path

    return class_paths


def collect_image_records(dataset_dir: Path) -> List[ImageRecord]:
    """Collect image file paths and labels without loading images into memory."""
    class_paths = validate_dataset_dir(dataset_dir)
    records: List[ImageRecord] = []

    for class_name, class_path in class_paths.items():
        image_paths = sorted(
            path
            for path in class_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in image_paths:
            records.append(
                ImageRecord(
                    path=image_path,
                    class_name=class_name,
                    label=CLASS_TO_LABEL[class_name],
                )
            )

    if not records:
        raise ValueError(f"No image files found in dataset directory: {dataset_dir}")

    return records


def verify_readable_images(records: Sequence[ImageRecord]) -> List[Tuple[ImageRecord, str]]:
    """Return unreadable images so they can be excluded before training."""
    unreadable: List[Tuple[ImageRecord, str]] = []

    for record in records:
        try:
            with Image.open(record.path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError) as error:
            unreadable.append((record, str(error)))

    return unreadable


def validate_kidney_image(image: Image.Image, ood_bounds: dict) -> Tuple[bool, str]:
    """Validate if the given PIL image fits the expected profile of a hyperspectral kidney scan."""
    try:
        rgb_img = image.convert("RGB")
    except Exception as e:
        return False, f"Failed to convert image to RGB: {e}"

    img_arr = np.array(rgb_img)
    if len(img_arr.shape) != 3 or img_arr.shape[2] != 3:
        return False, "Image must have 3 color channels (RGB)."

    # Calculate channel statistics
    mean = img_arr.mean(axis=(0, 1))
    std = img_arr.std(axis=(0, 1))

    # Check bounds
    r_mean_min, r_mean_max = ood_bounds["r_mean"]
    g_mean_min, g_mean_max = ood_bounds["g_mean"]
    b_mean_min, b_mean_max = ood_bounds["b_mean"]

    r_std_min, r_std_max = ood_bounds["r_std"]
    g_std_min, g_std_max = ood_bounds["g_std"]
    b_std_min, b_std_max = ood_bounds["b_std"]

    if not (r_mean_min <= mean[0] <= r_mean_max):
        return False, f"Red channel mean {mean[0]:.2f} is out of bounds [{r_mean_min}, {r_mean_max}]."
    if not (g_mean_min <= mean[1] <= g_mean_max):
        return False, f"Green channel mean {mean[1]:.2f} is out of bounds [{g_mean_min}, {g_mean_max}]."
    if not (b_mean_min <= mean[2] <= b_mean_max):
        return False, f"Blue channel mean {mean[2]:.2f} is out of bounds [{b_mean_min}, {b_mean_max}]."

    if not (r_std_min <= std[0] <= r_std_max):
        return False, f"Red channel std {std[0]:.2f} is out of bounds [{r_std_min}, {r_std_max}]."
    if not (g_std_min <= std[1] <= g_std_max):
        return False, f"Green channel std {std[1]:.2f} is out of bounds [{g_std_min}, {g_std_max}]."
    if not (b_std_min <= std[2] <= b_std_max):
        return False, f"Blue channel std {std[2]:.2f} is out of bounds [{b_std_min}, {b_std_max}]."

    return True, "Valid kidney image"


def stratified_split(
    records: Sequence[ImageRecord],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[ImageRecord], List[ImageRecord], List[ImageRecord]]:
    """Split records while keeping each class proportion similar in every split."""
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    grouped: Dict[int, List[ImageRecord]] = {}
    for record in records:
        grouped.setdefault(record.label, []).append(record)

    train_records: List[ImageRecord] = []
    val_records: List[ImageRecord] = []
    test_records: List[ImageRecord] = []
    random_generator = random.Random(seed)

    for label_records in grouped.values():
        shuffled_records = list(label_records)
        random_generator.shuffle(shuffled_records)

        total = len(shuffled_records)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_records.extend(shuffled_records[:train_end])
        val_records.extend(shuffled_records[train_end:val_end])
        test_records.extend(shuffled_records[val_end:])

    random_generator.shuffle(train_records)
    random_generator.shuffle(val_records)
    random_generator.shuffle(test_records)

    return train_records, val_records, test_records


def count_by_class(records: Sequence[ImageRecord]) -> Dict[str, int]:
    """Count records by class name."""
    counts = {class_name: 0 for class_name in CLASS_TO_LABEL}
    for record in records:
        counts[record.class_name] += 1
    return counts


def save_split_csv(
    split_records: Dict[str, Sequence[ImageRecord]],
    output_path: Path,
    project_root: Optional[Path] = None,
) -> None:
    """Save train/validation/test membership to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["split", "path", "class_name", "label"])

        for split_name, records in split_records.items():
            for record in records:
                image_path = record.path
                if project_root is not None:
                    image_path = image_path.relative_to(project_root)
                writer.writerow([split_name, str(image_path), record.class_name, record.label])


def load_split_csv(split_csv_path: Path, project_root: Path) -> Dict[str, List[ImageRecord]]:
    """Load train/validation/test records from a split CSV file."""
    if not split_csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv_path}")

    splits: Dict[str, List[ImageRecord]] = {"train": [], "validation": [], "test": []}

    with split_csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"split", "path", "class_name", "label"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"Split CSV must contain columns: {sorted(required_columns)}")

        for row in reader:
            split_name = row["split"]
            if split_name not in splits:
                raise ValueError(f"Unknown split name in CSV: {split_name}")

            image_path = project_root / row["path"]
            class_name = row["class_name"]
            label = int(row["label"])

            if class_name not in CLASS_TO_LABEL:
                raise ValueError(f"Unknown class name in CSV: {class_name}")
            if label != CLASS_TO_LABEL[class_name]:
                raise ValueError(f"Label mismatch for {image_path}: {class_name} should be {CLASS_TO_LABEL[class_name]}")

            splits[split_name].append(
                ImageRecord(path=image_path, class_name=class_name, label=label)
            )

    for split_name, records in splits.items():
        if not records:
            raise ValueError(f"Split '{split_name}' has zero records in {split_csv_path}")

    return splits


class KidneyImageDataset(Dataset):
    """PyTorch Dataset that loads one kidney image at a time."""

    def __init__(
        self,
        records: Sequence[ImageRecord],
        transform: Optional[Callable[[Image.Image], Tensor]] = None,
    ) -> None:
        self.records = list(records)
        self.transform = transform

        if not self.records:
            raise ValueError("KidneyImageDataset received zero records.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        record = self.records[index]

        try:
            with Image.open(record.path) as image:
                image = image.convert("RGB")
                if self.transform is not None:
                    image_tensor = self.transform(image)
                else:
                    raise ValueError("A transform is required to convert PIL images to tensors.")
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise RuntimeError(f"Could not load image: {record.path}") from error

        return image_tensor, record.label
