from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def main() -> int:
    out = Path("assets/demo_input.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    w = h = 320

    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[:] = (232, 201, 180)  # skin-like background
    noise = rng.normal(0, 6, size=base.shape).astype(np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(base, mode="RGB")

    draw = ImageDraw.Draw(img, "RGBA")
    # irregular dark blob (synthetic "lesion-like" shape)
    for k in range(6):
        dx = int(rng.integers(-20, 20))
        dy = int(rng.integers(-20, 20))
        x0, y0 = 110 + dx, 110 + dy
        x1, y1 = 230 + dx, 235 + dy
        color = (90, 55, 45, 70 + 15 * k)
        draw.ellipse([x0, y0, x1, y1], fill=color)

    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    img.save(out)
    print("Wrote:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

