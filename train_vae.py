from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dsci498_skin.data.ham10000 import Ham10000Dataset, build_samples
from dsci498_skin.models.cvae import ConditionalVAE, vae_loss
from dsci498_skin.runpack import RunMeta, copy_config, create_run_dir, git_head_sha, utc_now, write_meta
from dsci498_skin.train_utils import save_json, seed_everything


def _load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default="cvae")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    dataset_root = Path(cfg["dataset"]["root"])
    image_size = int(cfg["dataset"]["image_size"])

    seed = int(cfg["train"]["seed"])
    seed_everything(seed)

    samples = build_samples(dataset_root)
    classes = sorted({s.dx for s in samples})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    ds = Ham10000Dataset(samples=samples, class_to_idx=class_to_idx, transform=tf)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalVAE(
        num_classes=len(classes),
        image_size=image_size,
        latent_dim=int(cfg["model"]["latent_dim"]),
        label_emb_dim=int(cfg["model"]["label_emb_dim"]),
    ).to(device)
    optim = torch.optim.AdamW(
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
            cmd=f"python train_vae.py --config {args.config}",
            config_path=str(config_copy),
            git_head_sha=git_head_sha(repo_root),
            extra={"seed": seed},
        ),
    )
    save_json(outdir / "classes.json", {"class_to_idx": class_to_idx, "idx_to_class": {str(k): v for k, v in idx_to_class.items()}})

    epochs = int(cfg["train"]["epochs"])
    model.train()
    for epoch in range(1, epochs + 1):
        pbar = tqdm(loader, desc=f"epoch {epoch}/{epochs}")
        avg_loss = 0.0
        avg_recon = 0.0
        avg_kl = 0.0
        n = 0
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            recon, mu, logvar = model(x, y)
            loss, parts = vae_loss(recon, x, mu, logvar)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            bsz = x.shape[0]
            avg_loss += float(loss.item()) * bsz
            avg_recon += float(parts["recon_mse"].item()) * bsz
            avg_kl += float(parts["kl"].item()) * bsz
            n += bsz
            pbar.set_postfix(loss=avg_loss / n, recon=avg_recon / n, kl=avg_kl / n)

    torch.save({"model": model.state_dict(), "cfg": cfg, "class_to_idx": class_to_idx}, outdir / "cvae.pt")
    print("Saved:", outdir / "cvae.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

