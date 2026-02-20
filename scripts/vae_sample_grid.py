from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _load_first_images(class_dir: Path, n: int) -> list[Image.Image]:
    imgs: list[Image.Image] = []
    for p in sorted(class_dir.glob("*.png"))[:n]:
        imgs.append(Image.open(p).convert("RGB"))
    return imgs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth-root", type=Path, default=Path("artifacts/synth"))
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("results/vae_samples_grid.png"))
    args = parser.parse_args()

    classes = sorted([p.name for p in args.synth_root.iterdir() if p.is_dir()])
    if not classes:
        raise FileNotFoundError(f"No class folders found under {args.synth_root}")

    rows = []
    for dx in classes:
        imgs = _load_first_images(args.synth_root / dx, args.per_class)
        if not imgs:
            continue
        rows.append((dx, imgs))

    if not rows:
        raise FileNotFoundError("No images found to build a grid.")

    w, h = rows[0][1][0].size
    margin = 6
    label_w = 90
    grid_w = label_w + args.per_class * w + (args.per_class + 1) * margin
    grid_h = len(rows) * h + (len(rows) + 1) * margin

    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    y = margin
    for dx, imgs in rows:
        draw.text((margin, y + h // 3), dx, fill=(0, 0, 0), font=font)
        x = label_w
        for img in imgs:
            canvas.paste(img, (x + margin, y))
            x += w + margin
        y += h + margin

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print("Saved:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

