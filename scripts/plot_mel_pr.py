from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/mel_pr_curve.png"))
    args = parser.parse_args()

    npz = np.load(args.run_dir / "test_outputs.npz", allow_pickle=False)
    y_true = npz["y_true"]
    probs = npz["probs"]
    classes = [c for c in npz["classes"].tolist()]

    if "mel" not in classes:
        raise ValueError("Class 'mel' not found in classes.")
    mel_idx = classes.index("mel")

    y_bin = (y_true == mel_idx).astype(np.int32)
    scores = probs[:, mel_idx]
    ap = float(average_precision_score(y_bin, scores))
    precision, recall, _ = precision_recall_curve(y_bin, scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AP={ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Melanoma (mel) Precision-Recall (test)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print("Saved:", args.out)
    print("Average precision:", ap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

