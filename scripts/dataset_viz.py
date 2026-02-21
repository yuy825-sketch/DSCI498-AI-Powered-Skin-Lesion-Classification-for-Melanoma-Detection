from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def _find_image(root: Path, image_id: str) -> Path:
    candidates = [
        root / f"{image_id}.jpg",
        root / "images" / f"{image_id}.jpg",
        root / "HAM10000_images" / f"{image_id}.jpg",
        root / "ham10000_images" / f"{image_id}.jpg",
        root / "ham10000_images_part_1" / f"{image_id}.jpg",
        root / "ham10000_images_part_2" / f"{image_id}.jpg",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {image_id}.jpg under {root}")


def plot_class_distribution(df: pd.DataFrame, out_png: Path) -> None:
    counts = df["dx"].value_counts().sort_values(ascending=False)
    labels = counts.index.tolist()
    values = counts.values.astype(int)
    total = int(values.sum())
    perc = values / max(1, total) * 100.0

    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["tab:red" if c == "mel" else "tab:blue" for c in labels]
    bars = ax.bar(labels, values, color=colors, alpha=0.9)
    ax.set_title("HAM10000 class distribution (dx)")
    ax.set_xlabel("Class (dx)")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.25)

    for bar, p in zip(bars, perc, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{p:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_metadata_stats(df: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    # Age histogram
    ax = axes[0]
    age = pd.to_numeric(df["age"], errors="coerce").dropna()
    ax.hist(age.values, bins=30, color="tab:purple", alpha=0.85)
    ax.set_title("Age distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.25)

    # Sex distribution
    ax = axes[1]
    sex_counts = df["sex"].value_counts().reindex(["male", "female", "unknown"]).dropna()
    ax.bar(sex_counts.index.tolist(), sex_counts.values.astype(int), color="tab:green", alpha=0.85)
    ax.set_title("Sex distribution")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.25)

    # Localization (top 10)
    ax = axes[2]
    loc_counts = df["localization"].value_counts().head(10).sort_values(ascending=True)
    ax.barh(loc_counts.index.tolist(), loc_counts.values.astype(int), color="tab:orange", alpha=0.85)
    ax.set_title("Top localizations (top 10)")
    ax.set_xlabel("Count")
    ax.grid(True, axis="x", alpha=0.25)

    fig.suptitle("HAM10000 metadata overview", y=1.02, fontsize=13)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_samples_grid(
    df: pd.DataFrame,
    root: Path,
    out_png: Path,
    *,
    per_class: int,
    seed: int,
    thumb_size: int = 160,
) -> None:
    rng = np.random.default_rng(int(seed))
    classes = sorted(df["dx"].unique().tolist())

    fig_h = max(6.0, 1.25 * len(classes))
    fig, axes = plt.subplots(len(classes), per_class, figsize=(2.4 * per_class, fig_h))
    if len(classes) == 1:
        axes = np.asarray(axes).reshape(1, per_class)
    if per_class == 1:
        axes = np.asarray(axes).reshape(len(classes), 1)

    for i, c in enumerate(classes):
        subset = df[df["dx"] == c]["image_id"].tolist()
        if len(subset) < per_class:
            chosen = subset
        else:
            chosen = rng.choice(subset, size=per_class, replace=False).tolist()

        for j in range(per_class):
            ax = axes[i, j]
            ax.axis("off")
            if j >= len(chosen):
                continue
            image_id = str(chosen[j])
            img_path = _find_image(root, image_id)
            img = Image.open(img_path).convert("RGB")
            img = img.resize((thumb_size, thumb_size))
            ax.imshow(img)
            if j == 0:
                ax.set_title(f"{c}", loc="left", fontsize=11, pad=6)
            ax.set_title(image_id, fontsize=8, pad=2)

    fig.suptitle("HAM10000 sample thumbnails (random, per class)", y=1.01, fontsize=13)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_samples_strip(
    df: pd.DataFrame,
    root: Path,
    out_png: Path,
    *,
    seed: int,
    thumb_size: int = 140,
) -> None:
    rng = np.random.default_rng(int(seed))
    classes = sorted(df["dx"].unique().tolist())

    fig, axes = plt.subplots(1, len(classes), figsize=(2.1 * len(classes), 2.6))
    if len(classes) == 1:
        axes = np.asarray([axes])

    for ax, c in zip(axes, classes, strict=True):
        ax.axis("off")
        subset = df[df["dx"] == c]["image_id"].tolist()
        image_id = str(rng.choice(subset, size=1, replace=False).tolist()[0])
        img_path = _find_image(root, image_id)
        img = Image.open(img_path).convert("RGB").resize((thumb_size, thumb_size))
        ax.imshow(img)
        ax.set_title(c, fontsize=10, pad=4)

    fig.suptitle("HAM10000 sample strip (one random example per class)", y=1.02, fontsize=12)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/ham10000"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/dataset"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-class", type=int, default=2, help="Samples per class in the sample grid.")
    args = parser.parse_args()

    meta_path = args.root / "HAM10000_metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing HAM10000 metadata CSV: {meta_path}")

    df = pd.read_csv(meta_path)
    required = {"image_id", "dx", "age", "sex", "localization"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {sorted(missing)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_class_distribution(df, args.out_dir / "class_distribution.png")
    plot_metadata_stats(df, args.out_dir / "metadata_stats.png")
    plot_samples_grid(
        df,
        args.root,
        args.out_dir / "samples_grid.png",
        per_class=int(args.per_class),
        seed=int(args.seed),
    )
    plot_samples_strip(df, args.root, args.out_dir / "samples_strip.png", seed=int(args.seed))

    print("Wrote:", args.out_dir / "class_distribution.png")
    print("Wrote:", args.out_dir / "metadata_stats.png")
    print("Wrote:", args.out_dir / "samples_grid.png")
    print("Wrote:", args.out_dir / "samples_strip.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
