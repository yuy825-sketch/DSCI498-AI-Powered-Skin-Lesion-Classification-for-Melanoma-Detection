from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from dsci498_skin.models.cvae import ConditionalVAE


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True, help="Run dir containing cvae.pt + config.json + classes.json")
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/synth"))
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    cfg = _load_json(args.run_dir / "config.json")
    classes = _load_json(args.run_dir / "classes.json")
    class_to_idx: dict[str, int] = {k: int(v) for k, v in classes["class_to_idx"].items()}

    image_size = int(cfg["dataset"]["image_size"])
    latent_dim = int(cfg["model"]["latent_dim"])
    label_emb_dim = int(cfg["model"]["label_emb_dim"])

    model = ConditionalVAE(
        num_classes=len(class_to_idx),
        image_size=image_size,
        latent_dim=latent_dim,
        label_emb_dim=label_emb_dim,
    ).to(device)
    ckpt = torch.load(args.run_dir / "cvae.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    inv = {v: k for k, v in class_to_idx.items()}
    args.out_root.mkdir(parents=True, exist_ok=True)

    for class_idx in range(len(inv)):
        dx = inv[class_idx]
        out_dir = args.out_root / dx
        out_dir.mkdir(parents=True, exist_ok=True)
        y = torch.full((args.per_class,), class_idx, dtype=torch.long, device=device)
        z = torch.randn((args.per_class, latent_dim), device=device)
        x = model.decode(z, y)  # (N, 3, S, S) in [0,1]
        x = (x.clamp(0, 1) * 255).to(torch.uint8).cpu()
        for i in range(args.per_class):
            img = x[i].permute(1, 2, 0).numpy()
            Image.fromarray(img).save(out_dir / f"synth_{i:05d}.png")

    print("Wrote synthetic images to:", args.out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

