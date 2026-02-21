from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/training_curves.png"))
    args = parser.parse_args()

    hist = _load_json(args.run_dir / "history.json")["history"]
    epochs = [h["epoch"] for h in hist]
    train_loss = [h["train_loss"] for h in hist]
    val_acc = [h["val_acc"] for h in hist]
    val_macro_f1 = [h["val_macro_f1"] for h in hist]
    val_mel = [h["val_mel_recall"] for h in hist]

    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax = ax.reshape(-1)

    ax[0].plot(epochs, train_loss, marker="o")
    ax[0].set_title("Train loss")
    ax[0].set_xlabel("Epoch")

    ax[1].plot(epochs, val_acc, marker="o")
    ax[1].set_title("Val accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylim(0, 1)

    ax[2].plot(epochs, val_macro_f1, marker="o")
    ax[2].set_title("Val macro-F1")
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylim(0, 1)

    ax[3].plot(epochs, val_mel, marker="o")
    ax[3].set_title("Val melanoma recall (`mel`)")
    ax[3].set_xlabel("Epoch")
    ax[3].set_ylim(0, 1)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print("Saved:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

