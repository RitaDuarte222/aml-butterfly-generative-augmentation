"""
gan.py — Conditional GAN (CGAN) for butterfly image generation.

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

    Parameters
    ----------
    latent_dim    : int  — dimension of the noise vector z (default 128)
    class_emb_dim : int  — dimension of the class embedding (default 64)
    num_classes   : int  — number of butterfly species (default 75)
    """

    def __init__(
        self,
        latent_dim:    int = LATENT_DIM,
        class_emb_dim: int = CLASS_EMB_DIM,
        num_classes:   int = NUM_CLASSES,
    ):
        super().__init__()
        self.latent_dim    = latent_dim
        self.class_emb_dim = class_emb_dim

        # Class conditioning
        self.class_emb = nn.Embedding(num_classes, class_emb_dim)

        # Project concatenated (z || emb) → spatial feature map
        self.fc = nn.Linear(latent_dim + class_emb_dim, 512 * 4 * 4)

        # Upsample: 512x4x4 → 3x64x64
        self.deconv = nn.Sequential(
            # 4x4 → 8x8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 8x8 → 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 16x16 → 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 32x32 → 64x64
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # output ∈ [-1, 1]
        )

        _init_weights(self)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z      : (B, latent_dim)   — noise sampled from N(0, I)
        labels : (B,)              — integer class indices

        Returns
        -------
        images : (B, 3, 64, 64)   — generated images in [-1, 1]
        """
        emb = self.class_emb(labels)               # (B, class_emb_dim)
        x   = torch.cat([z, emb], dim=1)           # (B, latent_dim + class_emb_dim)
        x   = self.fc(x).view(-1, 512, 4, 4)       # (B, 512, 4, 4)
        return self.deconv(x)                       # (B, 3, 64, 64)

    @torch.no_grad()
    def generate(
        self,
        labels:    torch.Tensor,
        device:    Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        Generate one image per label. Labels can be a 1-D tensor of class indices.
        """
        labels = labels.to(device)
        z      = torch.randn(len(labels), self.latent_dim, device=device)
        return self(z, labels)


# ── Discriminator ────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    """
    Conditional Discriminator D(x, y).

    The class label is embedded and projected to a (1 x 64 x 64) spatial map
    which is concatenated with the image as an additional channel, so the
    discriminator sees 4-channel input.

    Parameters
    ----------
    num_classes   : int  — number of butterfly species (default 75)
    class_emb_dim : int  — dimension of the class embedding (default 64)
    """

    def __init__(
        self,
        num_classes:   int = NUM_CLASSES,
        class_emb_dim: int = CLASS_EMB_DIM,
    ):
        super().__init__()

        # Class conditioning: label → (1 x 64 x 64) map concatenated to the image
        self.class_emb  = nn.Embedding(num_classes, class_emb_dim)
        self.class_proj = nn.Linear(class_emb_dim, IMAGE_SIZE * IMAGE_SIZE)

        # Downsample: [3+1] x 64x64 → 512x4x4
        self.conv = nn.Sequential(
            # 4x64x64 → 64x32x32  (no BN in first layer, as per DCGAN paper)
            nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x32x32 → 128x16x16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128x16x16 → 256x8x8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256x8x8 → 512x4x4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),   # raw logit — use BCEWithLogitsLoss
        )

        _init_weights(self)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x      : (B, 3, 64, 64)  — real or generated images in [-1, 1]
        labels : (B,)             — integer class indices

        Returns
        -------
        logits : (B, 1)           — raw (un-sigmoided) real/fake scores
        """
        emb       = self.class_emb(labels)                   # (B, class_emb_dim)
        class_map = self.class_proj(emb)                     # (B, 64*64)
        class_map = class_map.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)  # (B, 1, 64, 64)
        x_cond    = torch.cat([x, class_map], dim=1)         # (B, 4, 64, 64)
        return self.fc(self.conv(x_cond))                    # (B, 1)


# ── Label-smoothed targets ────────────────────────────────────────────────────
def real_label(batch_size: int, device: torch.device, smoothing: float = 0.1) -> torch.Tensor:
    """Soft real target: U(1-smoothing, 1)  — reduces D overconfidence."""
    return torch.empty(batch_size, 1, device=device).uniform_(1.0 - smoothing, 1.0)


def fake_label(batch_size: int, device: torch.device) -> torch.Tensor:
    """Hard fake target: 0."""
    return torch.zeros(batch_size, 1, device=device)


# ── Convenience: build both models at once ────────────────────────────────────
def build_cgan(
    latent_dim:    int = LATENT_DIM,
    class_emb_dim: int = CLASS_EMB_DIM,
    num_classes:   int = NUM_CLASSES,
    device:        Union[str, torch.device] = "cpu",
):
    """Instantiate and move G and D to *device*. Returns (G, D)."""
    G = Generator(latent_dim, class_emb_dim, num_classes).to(device)
    D = Discriminator(num_classes, class_emb_dim).to(device)
    return G, D
