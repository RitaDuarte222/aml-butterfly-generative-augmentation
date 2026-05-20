"""
gan1.py — Conditional GAN (CGAN) for butterfly image generation.

Architecture
------------
Generator     : Embedding(label) ++ z  →  FC  →  ConvTranspose layers  →  Tanh
Discriminator : Conv layers over [image | class_map]  →  FC  →  logit (BCEWithLogits)

Conditioning strategy
---------------------
  Generator     : class label is embedded and concatenated with the noise vector z
                  before the first FC layer.
  Discriminator : class label is embedded, projected to a (1 x 64 x 64) spatial map,
                  and concatenated with the input image as an extra channel.

Input  : 3 x 64 x 64 RGB images normalised to [-1, 1]  (Tanh output from G)
Latent : configurable (default LATENT_DIM = 128)
Classes: 75
"""

import torch
import torch.nn as nn
from typing import Union

# ── Default hyper-parameters ────────────────────────────────────────────────
LATENT_DIM    = 128
CLASS_EMB_DIM = 64
NUM_CLASSES   = 75
IMAGE_SIZE    = 64

# ── Weight initialisation (DCGAN paper recommendation) ──────────────────────
def _init_weights(module: nn.Module) -> None:
    """Apply normal(0, 0.02) to Conv/Linear weights; normal(1, 0.02) to BN."""
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, mean=1.0, std=0.02)
            nn.init.zeros_(m.bias)

# ── Generator ────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    """
    Conditional Generator G(z, y).
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        class_emb_dim: int = CLASS_EMB_DIM,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.latent_dim    = latent_dim
        self.class_emb_dim = class_emb_dim

        self.class_emb = nn.Embedding(num_classes, class_emb_dim)
        self.fc = nn.Linear(latent_dim + class_emb_dim, 512 * 4 * 4)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),
        )

        _init_weights(self)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = self.class_emb(labels)
        x   = torch.cat([z, emb], dim=1)
        x   = self.fc(x).view(-1, 512, 4, 4)
        return self.deconv(x)

    @torch.no_grad()
    def generate(
        self,
        labels: torch.Tensor,
        device: Union[str, torch.device] = "cpu",  # Python 3.9 compatible
    ) -> torch.Tensor:
        labels = labels.to(device)
        z      = torch.randn(len(labels), self.latent_dim, device=device)
        return self(z, labels)

# ── Discriminator ────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        class_emb_dim: int = CLASS_EMB_DIM,
    ):
        super().__init__()
        self.class_emb  = nn.Embedding(num_classes, class_emb_dim)
        self.class_proj = nn.Linear(class_emb_dim, IMAGE_SIZE * IMAGE_SIZE)

        self.conv = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),
        )

        _init_weights(self)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb       = self.class_emb(labels)
        class_map = self.class_proj(emb).view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        x_cond    = torch.cat([x, class_map], dim=1)
        return self.fc(self.conv(x_cond))

# ── Label-smoothed targets ────────────────────────────────────────────────────
def real_label(batch_size: int, device: torch.device, smoothing: float = 0.1) -> torch.Tensor:
    return torch.empty(batch_size, 1, device=device).uniform_(1.0 - smoothing, 1.0)

def fake_label(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(batch_size, 1, device=device)

# ── Convenience function ─────────────────────────────────────────────────────
def build_cgan(
    latent_dim: int = LATENT_DIM,
    class_emb_dim: int = CLASS_EMB_DIM,
    num_classes: int = NUM_CLASSES,
    device: Union[str, torch.device] = "cpu",  # Python 3.9 compatible
):
    G = Generator(latent_dim, class_emb_dim, num_classes).to(device)
    D = Discriminator(num_classes, class_emb_dim).to(device)
    return G, D