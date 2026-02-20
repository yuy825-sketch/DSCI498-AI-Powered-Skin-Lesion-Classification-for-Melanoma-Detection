from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import transforms

from dsci498_skin.data.ham10000 import DX_TO_NAME, Ham10000Dataset, build_samples
from dsci498_skin.data.split import group_shuffle_split
from dsci498_skin.data.synthetic import SyntheticImageFolder
from dsci498_skin.models.cnn import CnnConfig, build_model
from dsci498_skin.runpack import RunMeta, copy_config, create_run_dir, git_head_sha, utc_now, write_meta
from dsci498_skin.train_utils import evaluate, save_json, seed_everything


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _make_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=25),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_tf, eval_tf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default="baseline")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    dataset_root = Path(cfg["dataset"]["root"])
    image_size = int(cfg["dataset"]["image_size"])

    train_seed = int(cfg["train"]["seed"])
    seed_everything(train_seed)

    samples = build_samples(dataset_root)
    classes = sorted({s.dx for s in samples})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    split = group_shuffle_split(
        groups=[s.lesion_id for s in samples],
        train_frac=float(cfg["dataset"]["train_frac"]),
        val_frac=float(cfg["dataset"]["val_frac"]),
        test_frac=float(cfg["dataset"]["test_frac"]),
        seed=int(cfg["dataset"]["split_seed"]),
    )

    train_tf, eval_tf = _make_transforms(image_size)
    train_ds = Subset(Ham10000Dataset(samples=samples, class_to_idx=class_to_idx, transform=train_tf), split.train_idx)
    val_ds = Subset(Ham10000Dataset(samples=samples, class_to_idx=class_to_idx, transform=eval_tf), split.val_idx)
    test_ds = Subset(Ham10000Dataset(samples=samples, class_to_idx=class_to_idx, transform=eval_tf), split.test_idx)

    synthetic_root = cfg["dataset"].get("synthetic_root", None)
    if synthetic_root:
        synth_ds = SyntheticImageFolder(root=Path(synthetic_root), class_to_idx=class_to_idx, transform=train_tf)
        train_ds = ConcatDataset([train_ds, synth_ds])

    num_workers = int(cfg["train"]["num_workers"])
    batch_size = int(cfg["train"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        CnnConfig(
            backbone=str(cfg["model"]["backbone"]),
            pretrained=bool(cfg["model"]["pretrained"]),
            dropout=float(cfg["model"]["dropout"]),
            num_classes=len(classes),
        )
    ).to(device)

    use_class_weights = bool(cfg["train"].get("use_class_weights", False))
    if use_class_weights:
        y_train = [class_to_idx[samples[i].dx] for i in split.train_idx]
        counts = np.bincount(np.array(y_train), minlength=len(classes)).astype(np.float32)
        weights = (counts.sum() / (counts + 1e-6))
        weights = weights / weights.mean()
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    repo_root = Path(__file__).resolve().parent
    if args.outdir is None:
        outdir = create_run_dir(runs_root=repo_root / "runs", name=args.run_name)
    else:
        outdir = args.outdir
        outdir.mkdir(parents=True, exist_ok=True)

    config_copy = copy_config(args.config, outdir)
    write_meta(
        outdir / "meta.json",
        RunMeta(
            created_utc=utc_now(),
            name=args.run_name,
            cmd=f"python train.py --config {args.config}",
            config_path=str(config_copy),
            git_head_sha=git_head_sha(repo_root),
            extra={"train_seed": train_seed},
        ),
    )
    save_json(
        outdir / "classes.json",
        {
            "class_to_idx": class_to_idx,
            "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
            "dx_to_name": DX_TO_NAME,
        },
    )

    best_val_macro_f1 = -1.0
    best_ckpt = outdir / "best.pt"
    epochs = int(cfg["train"]["epochs"])
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(x.shape[0])
            n += int(x.shape[0])

        val_metrics, _ = evaluate(model=model, loader=val_loader, idx_to_class=idx_to_class, device=device)
        avg_loss = total_loss / max(1, n)
        print(
            f"epoch={epoch:03d} loss={avg_loss:.4f} "
            f"val_acc={val_metrics.accuracy:.4f} val_macro_f1={val_metrics.macro_f1:.4f}"
        )
        if val_metrics.macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_metrics.macro_f1
            torch.save({"model": model.state_dict(), "cfg": cfg}, best_ckpt)

    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics, test_report = evaluate(model=model, loader=test_loader, idx_to_class=idx_to_class, device=device)

    save_json(
        outdir / "metrics.json",
        {
            "test": {
                "accuracy": test_metrics.accuracy,
                "macro_f1": test_metrics.macro_f1,
                "per_class_recall": test_metrics.per_class_recall,
                "confusion_matrix": test_metrics.confusion_matrix,
            },
            "test_report": test_report,
        },
    )
    print("Saved:", outdir / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
