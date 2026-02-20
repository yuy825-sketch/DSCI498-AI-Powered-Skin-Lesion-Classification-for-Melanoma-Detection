from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dsci498_skin.infer import default_eval_transform, load_run_model
from dsci498_skin.interpret.gradcam import gradcam, infer_target_layer


def _overlay(image_rgb: np.ndarray, heatmap01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import cv2

    h, w = image_rgb.shape[:2]
    heatmap = (heatmap01 * 255).astype(np.uint8)
    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap_color + (1 - alpha) * image_rgb).astype(np.uint8)
    return overlay


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/gradcam_example.png"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--class-idx", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_run_model(args.run_dir, device=device)

    image = Image.open(args.image).convert("RGB")
    transform = default_eval_transform(args.image_size)
    x = transform(image).unsqueeze(0).to(device)

    target_layer = infer_target_layer(loaded.model)
    res = gradcam(model=loaded.model, target_layer=target_layer, x=x, class_idx=args.class_idx)

    image_rgb = np.array(image)
    overlay = _overlay(image_rgb, res.heatmap, alpha=0.45)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(args.out)
    print("Saved:", args.out)
    print("Predicted class idx:", res.class_idx, "score:", f"{res.score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

