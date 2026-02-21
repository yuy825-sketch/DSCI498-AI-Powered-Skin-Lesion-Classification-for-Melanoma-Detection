from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@dataclass
class Metrics:
    accuracy: float
    macro_f1: float
    per_class_recall: dict[str, float]
    confusion_matrix: list[list[int]]


def evaluate(
    *,
    model: nn.Module,
    loader,
    idx_to_class: dict[int, str],
    device: torch.device,
) -> tuple[Metrics, dict[str, Any]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(y.numpy().tolist())

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    labels = list(range(len(idx_to_class)))
    recalls = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_class_recall = {idx_to_class[i]: float(recalls[i]) for i in labels}
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[idx_to_class[i] for i in labels],
        output_dict=True,
        zero_division=0,
    )
    return Metrics(accuracy=acc, macro_f1=macro_f1, per_class_recall=per_class_recall, confusion_matrix=cm), report


@torch.no_grad()
def predict_proba(
    *,
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      y_true: (N,)
      probs: (N, C)
    """
    model.eval()
    y_true: list[int] = []
    probs: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=1).detach().cpu().numpy()
        probs.append(p)
        y_true.extend(y.numpy().tolist())
    return np.asarray(y_true, dtype=np.int64), np.concatenate(probs, axis=0)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
