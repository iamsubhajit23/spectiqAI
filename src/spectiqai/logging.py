"""Audit logging and monitoring utilities for SpectiqAI."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Resolve audit log path inside SpectiqAI/results/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_LOG_PATH = PROJECT_ROOT / "results" / "audit_log.jsonl"


def log_prediction(
    filename: str,
    prediction: str,
    confidence: float,
    tumor_prob: float,
    normal_prob: float,
    latency: float,
    model_name: str,
    is_valid: bool,
    validation_msg: str
) -> None:
    """
    Append diagnostic session metadata to results/audit_log.jsonl for clinical audit trails.

    Args:
        filename: Name of the uploaded scan image.
        prediction: Diagnostic state (Tumor, Normal, Uncertain Prediction, Invalid / Unsupported Image).
        confidence: Prediction confidence score.
        tumor_prob: Raw probability of tumor class.
        normal_prob: Raw probability of normal class.
        latency: Model execution time in seconds.
        model_name: Name of the classification architecture.
        is_valid: Boolean indicating if OOD check passed.
        validation_msg: Output message from the OOD validation check.
    """
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "prediction": prediction,
        "confidence": round(float(confidence), 4) if confidence is not None else None,
        "tumor_prob": round(float(tumor_prob), 4),
        "normal_prob": round(float(normal_prob), 4),
        "latency_sec": round(float(latency), 4),
        "model_name": model_name,
        "ood_validation_passed": is_valid,
        "validation_message": validation_msg
    }

    try:
        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Safe atomic append block
        with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Warning: Failed to append to audit log file: {e}")


def get_prediction_history(max_entries: int = 10) -> List[Dict[str, Any]]:
    """
    Load prediction history from the audit log, returning the latest logs first.

    Args:
        max_entries: Max number of history items to load.
    """
    if not AUDIT_LOG_PATH.exists():
        return []

    entries: List[Dict[str, Any]] = []
    try:
        with AUDIT_LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    try:
                        entries.append(json.loads(stripped))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Warning: Failed to read audit log file: {e}")
        return []

    # Reverse to show newest first, then slice
    return entries[::-1][:max_entries]
