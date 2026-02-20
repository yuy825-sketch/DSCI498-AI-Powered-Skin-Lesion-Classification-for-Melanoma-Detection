from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from dsci498_skin.models.cnn import CnnConfig, build_model


@dataclass(frozen=True)
class LoadedModel:
    model: nn.Module
    class_to_idx: dict[str, int]
    idx_to_class: dict[int, str]


def load_run_model(run_dir: Path, device: torch.device) -> LoadedModel:
    ckpt_path = run_dir / "best.pt"
    classes_path = run_dir / "classes.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not classes_path.exists():
        raise FileNotFoundError(f"Missing classes: {classes_path}")

    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    class_to_idx = {k: int(v) for k, v in classes["class_to_idx"].items()}
    idx_to_class = {int(k): v for k, v in classes["idx_to_class"].items()}

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    model = build_model(
        CnnConfig(
            backbone=str(cfg["model"]["backbone"]),
            pretrained=False,
            dropout=float(cfg["model"]["dropout"]),
            num_classes=len(class_to_idx),
        )
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return LoadedModel(model=model, class_to_idx=class_to_idx, idx_to_class=idx_to_class)


def default_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


@torch.no_grad()
def predict_topk(
    *,
    model: nn.Module,
    image: Image.Image,
    transform: transforms.Compose,
    idx_to_class: dict[int, str],
    device: torch.device,
    k: int = 3,
) -> list[tuple[str, float]]:
    x = transform(image).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    topk = torch.topk(probs, k=min(k, probs.numel()))
    results: list[tuple[str, float]] = []
    for idx, val in zip(topk.indices.tolist(), topk.values.tolist(), strict=True):
        results.append((idx_to_class[int(idx)], float(val)))
    return results

