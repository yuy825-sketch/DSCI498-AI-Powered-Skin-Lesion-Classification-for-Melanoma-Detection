from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


DX_TO_NAME: dict[str, str] = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}


@dataclass(frozen=True)
class HamSample:
    image_id: str
    lesion_id: str
    dx: str
    image_path: Path


def _find_image_path(root: Path, image_id: str) -> Path:
    candidates = [
        root / f"{image_id}.jpg",
        root / "images" / f"{image_id}.jpg",
        root / "HAM10000_images" / f"{image_id}.jpg",
        root / "ham10000_images" / f"{image_id}.jpg",
        root / "ham10000_images_part_1" / f"{image_id}.jpg",
        root / "ham10000_images_part_2" / f"{image_id}.jpg",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find image {image_id}.jpg under {root}. "
        "Expected either images/ or ham10000_images_part_1/ + ham10000_images_part_2/."
    )


def load_metadata(root: Path) -> pd.DataFrame:
    csv_path = root / "HAM10000_metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"image_id", "lesion_id", "dx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {sorted(missing)}")
    return df


def build_samples(root: Path) -> list[HamSample]:
    df = load_metadata(root)
    samples: list[HamSample] = []
    for row in df.itertuples(index=False):
        image_id = getattr(row, "image_id")
        lesion_id = getattr(row, "lesion_id")
        dx = getattr(row, "dx")
        image_path = _find_image_path(root, image_id)
        samples.append(HamSample(image_id=image_id, lesion_id=lesion_id, dx=dx, image_path=image_path))
    return samples


class Ham10000Dataset(Dataset[tuple[Any, int]]):
    def __init__(
        self,
        samples: list[HamSample],
        class_to_idx: dict[str, int],
        transform=None,
    ) -> None:
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.class_to_idx[sample.dx]
        return image, label
