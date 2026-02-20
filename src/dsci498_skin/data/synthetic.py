from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


class SyntheticImageFolder(Dataset[tuple[Any, int]]):
    """
    Expects:
      <root>/<dx>/*.png (or jpg)
    where <dx> is the HAM10000 dx code (e.g., mel, nv, bkl, ...).
    """

    def __init__(self, root: Path, class_to_idx: dict[str, int], transform=None) -> None:
        self.root = root
        self.class_to_idx = class_to_idx
        self.transform = transform

        items: list[tuple[Path, int]] = []
        for dx, idx in class_to_idx.items():
            d = root / dx
            if not d.exists():
                continue
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for p in sorted(d.glob(ext)):
                    items.append((p, idx))
        if not items:
            raise FileNotFoundError(f"No synthetic images found under {root}")
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, label = self.items[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

