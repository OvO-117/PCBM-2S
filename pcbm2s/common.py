"""Reproducible configuration, checkpoint and metric helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, accuracy_score, precision_recall_curve


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device="auto"):
    selected = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "auto":
        selected = device
    result = torch.device(selected)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return result


def read_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if set(value) != {"model", "training"}:
        raise ValueError("Config must contain 'model' and 'training' objects")
    return value


def write_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def classification_metrics(labels, probabilities, threshold=0.5):
    y = np.asarray(labels).reshape(-1)
    p = np.asarray(probabilities).reshape(-1)
    if y.size == 0 or y.shape != p.shape or not np.isfinite(p).all():
        raise ValueError("Empty, nonfinite or mismatched predictions")
    if not np.isin(y, [0, 1]).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected binary labels and probabilities in [0, 1]")
    both = np.unique(y).size == 2
    binary = (p >= threshold).astype(int)
    return {
        "n_samples": int(y.size), "auroc": float(roc_auc_score(y, p)) if both else None,
        "ap": float(average_precision_score(y, p)) if both else None,
        "f1": float(f1_score(y, binary, zero_division=0)),
        "accuracy": float(accuracy_score(y, binary)), "threshold": float(threshold),
    }


def validation_threshold(labels, probabilities):
    """Choose F1 threshold from VALIDATION labels, never test labels."""
    y, p = np.asarray(labels), np.asarray(probabilities)
    if np.unique(y).size != 2:
        raise ValueError("Validation split needs both classes for checkpoint selection")
    precision, recall, thresholds = precision_recall_curve(y, p)
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(thresholds[int(np.argmax(scores))])


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch).reshape(-1)
        if logits.numel() != batch.y.numel() or not torch.isfinite(logits).all():
            raise RuntimeError("Invalid model output; evaluation stopped")
        labels.extend(batch.y.reshape(-1).cpu().tolist())
        probabilities.extend(logits.sigmoid().cpu().tolist())
    if not labels:
        raise ValueError("Evaluation loader is empty")
    return labels, probabilities


def load_checkpoint(path, device="cpu"):
    """Load only trusted checkpoints; pickle can execute arbitrary code."""
    from .model import PCBM2S
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Expected a release checkpoint with schema_version=1")
    model = PCBM2S(checkpoint["descriptor_stats"], **checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint
