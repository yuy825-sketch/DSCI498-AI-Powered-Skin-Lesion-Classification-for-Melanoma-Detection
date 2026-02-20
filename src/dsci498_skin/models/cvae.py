from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalVAE(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        image_size: int = 64,
        latent_dim: int = 128,
        label_emb_dim: int = 16,
    ) -> None:
        super().__init__()
        if image_size not in (64, 128):
            raise ValueError("image_size must be 64 or 128 for this simple CVAE")

        self.num_classes = num_classes
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(num_classes, label_emb_dim)

        # Encoder: (3, S, S) -> (C, S/8, S/8)
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
        )

        feat_size = image_size // 8
        enc_dim = 128 * feat_size * feat_size
        self.fc_mu = nn.Linear(enc_dim + label_emb_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_dim + label_emb_dim, latent_dim)

        # Decoder: z + y -> (3, S, S)
        self.fc_dec = nn.Linear(latent_dim + label_emb_dim, enc_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        h = torch.flatten(h, start_dim=1)
        yemb = self.label_emb(y)
        h = torch.cat([h, yemb], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        yemb = self.label_emb(y)
        h = torch.cat([z, yemb], dim=1)
        h = self.fc_dec(h)
        feat_size = self.image_size // 8
        h = h.view(h.shape[0], 128, feat_size, feat_size)
        return self.dec(h)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar


def vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> tuple[torch.Tensor, dict]:
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + kl
    return loss, {"recon_mse": recon_loss.detach(), "kl": kl.detach()}

