from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _metrics(y_true_bin: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[float, float]:
    y_pred = (scores >= threshold).astype(np.int32)
    tp = int(((y_true_bin == 1) & (y_pred == 1)).sum())
    fp = int(((y_true_bin == 0) & (y_pred == 1)).sum())
    fn = int(((y_true_bin == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    return float(precision), float(recall)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/mel_threshold_curve.png"))
    parser.add_argument("--min-recall", type=float, default=0.85)
    args = parser.parse_args()

    npz = np.load(args.run_dir / "test_outputs.npz", allow_pickle=False)
    y_true = npz["y_true"].astype(np.int64)
    probs = npz["probs"].astype(np.float32)
    classes = [c for c in npz["classes"].tolist()]
    mel_idx = classes.index("mel")

    y_true_bin = (y_true == mel_idx).astype(np.int32)
    scores = probs[:, mel_idx]

    thresholds = np.linspace(0.0, 1.0, 101)
    precisions = []
    recalls = []
    best = None
    for t in thresholds:
        p, r = _metrics(y_true_bin, scores, float(t))
        precisions.append(p)
        recalls.append(r)
        if r >= args.min_recall:
            if best is None or p > best[1]:
                best = (float(t), p, r)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, recalls, label="Recall (sensitivity)")
    ax.plot(thresholds, precisions, label="Precision")
    ax.set_xlabel("Threshold on P(mel)")
    ax.set_ylabel("Value")
    ax.set_title("Melanoma one-vs-rest: precision/recall vs threshold (test)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    if best is not None:
        t, p, r = best
        ax.axvline(t, color="black", linestyle="--", alpha=0.6)
        ax.scatter([t], [r], color="tab:blue")
        ax.scatter([t], [p], color="tab:orange")
        ax.text(t + 0.01, r, f" recall={r:.2f}", fontsize=9)
        ax.text(t + 0.01, p, f" precision={p:.2f}", fontsize=9)
    ax.legend(loc="best")
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print("Saved:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

