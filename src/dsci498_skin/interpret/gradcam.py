from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class GradCamResult:
    class_idx: int
    score: float
    heatmap: np.ndarray  # (H, W), float32 in [0, 1]


class _HookStore:
    def __init__(self) -> None:
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None


def _normalize_heatmap(cam: np.ndarray) -> np.ndarray:
    cam = cam.astype(np.float32)
    cam = cam - cam.min()
    denom = cam.max() + 1e-8
    cam = cam / denom
    return cam


def gradcam(
    *,
    model: nn.Module,
    target_layer: nn.Module,
    x: torch.Tensor,
    class_idx: int | None = None,
    score_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> GradCamResult:
    """
    Minimal Grad-CAM for image classifiers.
    - x: (1, 3, H, W)
    """
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError("x must have shape (1, C, H, W)")

    model.eval()
    store = _HookStore()

    def fwd_hook(_module, _inp, out):
        store.activations = out.detach()

    def bwd_hook(_module, _grad_in, grad_out):
        store.gradients = grad_out[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    try:
        logits = model(x)
        if score_fn is None:
            scores = torch.softmax(logits, dim=1)
        else:
            scores = score_fn(logits)

        if class_idx is None:
            class_idx = int(torch.argmax(scores, dim=1).item())
        score = float(scores[0, class_idx].item())

        model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()

        if store.activations is None or store.gradients is None:
            raise RuntimeError("Failed to capture activations/gradients for Grad-CAM")

        # activations, gradients: (1, C, h, w)
        activations = store.activations
        grads = store.gradients

        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=False)  # (1, h, w)
        cam = torch.relu(cam)
        cam = cam[0].cpu().numpy()
        cam = _normalize_heatmap(cam)

        return GradCamResult(class_idx=class_idx, score=score, heatmap=cam)
    finally:
        h1.remove()
        h2.remove()


def infer_target_layer(model: nn.Module) -> nn.Module:
    """
    Best-effort heuristics for torchvision backbones used in this repo.
    """
    if hasattr(model, "features"):
        # EfficientNet-like
        feats = getattr(model, "features")
        if isinstance(feats, nn.Sequential) and len(feats) > 0:
            return feats[-1]
        return feats

    if hasattr(model, "layer4"):
        return getattr(model, "layer4")

    raise ValueError("Could not infer a target layer; pass target_layer explicitly.")

