from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _metrics(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> tuple[float, float]:
    tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
    fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    return float(precision), float(recall)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=Path("results/mel_threshold.md"))
    parser.add_argument("--min-recall", type=float, default=0.85)
    args = parser.parse_args()

    npz = np.load(args.run_dir / "test_outputs.npz", allow_pickle=False)
    y_true = npz["y_true"].astype(np.int64)
    probs = npz["probs"].astype(np.float32)
    classes = [c for c in npz["classes"].tolist()]

    if "mel" not in classes:
        raise ValueError("Class 'mel' not found in classes.")
    mel_idx = classes.index("mel")
    y_true_bin = (y_true == mel_idx).astype(np.int32)
    scores = probs[:, mel_idx]

    thresholds = np.linspace(0.0, 1.0, 101)
    rows: list[tuple[float, float, float]] = []
    best = None
    for t in thresholds:
        y_pred_bin = (scores >= t).astype(np.int32)
        precision, recall = _metrics(y_true_bin, y_pred_bin)
        rows.append((float(t), precision, recall))
        if recall >= args.min_recall:
            if best is None or precision > best[1]:
                best = (float(t), precision, recall)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Melanoma Threshold Analysis (one-vs-rest)\n")
    lines.append(f"- Run dir: `{args.run_dir}`")
    lines.append(f"- Target minimum recall: **{args.min_recall:.2f}**\n")

    if best is None:
        lines.append("No threshold achieved the target recall on this test set.\n")
    else:
        t, p, r = best
        lines.append("## Operating point (max precision under recall constraint)\n")
        lines.append(f"- Threshold on `P(mel)`: **{t:.2f}**")
        lines.append(f"- Precision: **{p:.3f}**")
        lines.append(f"- Recall (sensitivity): **{r:.3f}**\n")

    lines.append("## Sweep (threshold → precision/recall)\n")
    lines.append("| Threshold | Precision | Recall |")
    lines.append("|---:|---:|---:|")
    for t, p, r in rows:
        lines.append(f"| {t:.2f} | {p:.3f} | {r:.3f} |")

    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote:", args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

