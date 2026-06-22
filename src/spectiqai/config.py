"""Configuration management for SpectiqAI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

# Resolve project root (SpectiqAI/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"

DEFAULT_CONFIG: Dict[str, Any] = {
    "model_checkpoint_path": "models/simple_cnn/simple_cnn_epoch_3.pt",
    "image_size": 224,
    "tumor_threshold": 0.85,
    "confidence_threshold": 0.90,
    "class_names": {
        "0": "Normal",
        "1": "Tumor"
    },
    "ood_bounds": {
        "r_mean": [110.0, 170.0],
        "g_mean": [38.0, 135.0],
        "b_mean": [14.0, 92.0],
        "r_std": [35.0, 75.0],
        "g_std": [80.0, 120.0],
        "b_std": [38.0, 105.0]
    }
}


def load_config(config_path: Path | None = None) -> Dict[str, Any]:
    """Load configuration from a JSON file, falling back to defaults."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    config = DEFAULT_CONFIG.copy()

    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                user_config = json.load(f)
            # Recursively update config with user values
            for key, val in user_config.items():
                if isinstance(val, dict) and key in config and isinstance(config[key], dict):
                    config[key].update(val)
                else:
                    config[key] = val
        except Exception as e:
            # Fall back to default config if reading fails
            print(f"Warning: Failed to load config from {config_path}: {e}. Using defaults.")

    return config


def save_config(config: Dict[str, Any], config_path: Path | None = None) -> None:
    """Save configuration dictionary to a JSON file."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        raise IOError(f"Failed to save config to {config_path}: {e}")
