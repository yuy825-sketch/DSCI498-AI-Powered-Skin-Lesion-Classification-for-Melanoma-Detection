from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_confusion_matrix(cm: np.ndarray, labels: list[str], out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True",
        xlabel="Predicted",
        title="Confusion Matrix (Test)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() * 0.6 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _write_summary(metrics: dict, out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    test = metrics["test"]
    per_class_recall = test["per_class_recall"]
    mel_recall = per_class_recall.get("mel", None)

    lines: list[str] = []
    lines.append("# Results Summary\n")
    lines.append("## Key metrics (test)\n")
    lines.append(f"- Accuracy: **{test['accuracy']:.4f}**")
    lines.append(f"- Macro-F1: **{test['macro_f1']:.4f}**")
    if mel_recall is not None:
        lines.append(f"- Melanoma sensitivity (recall for `mel`): **{mel_recall:.4f}**")
    lines.append("\n## Per-class recall (test)\n")
    for k in sorted(per_class_recall.keys()):
        lines.append(f"- `{k}`: {per_class_recall[k]:.4f}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_run(*, run_dir: Path, out_dir: Path, tag: str | None = None) -> None:
    metrics_path = run_dir / "metrics.json"
    classes_path = run_dir / "classes.json"
    metrics = _load_json(metrics_path)
    classes = _load_json(classes_path)

    idx_to_class = classes["idx_to_class"]
    labels = [idx_to_class[str(i)] for i in range(len(idx_to_class))]
    cm = np.array(metrics["test"]["confusion_matrix"], dtype=np.int64)

    suffix = f"_{tag}" if tag else ""
    _write_summary(metrics, out_dir / f"summary{suffix}.md")
    _save_confusion_matrix(cm, labels, out_dir / f"confusion_matrix{suffix}.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to a run folder containing metrics.json")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--tag", type=str, default=None, help="Optional suffix for output filenames")
    args = parser.parse_args()

    export_run(run_dir=args.run_dir, out_dir=args.out_dir, tag=args.tag)

    suffix = f"_{args.tag}" if args.tag else ""
    print("Wrote:", args.out_dir / f"summary{suffix}.md")
    print("Wrote:", args.out_dir / f"confusion_matrix{suffix}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
