"""
autoencoder.py — Variational AutoEncoder (VAE) for butterfly image generation.

Architecture
------------
Encoder : Conv layers → Flatten → FC → (mu, log_var)
Sampling: z = mu + eps * std   (reparameterisation trick)
Decoder : FC → Reshape → ConvTranspose layers → tanh

Input  : 3 × 64 × 64 RGB images
Latent : configurable (default 128)
Output : 3 × 64 × 64 RGB images
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        # 3×64×64 → 256×4×4
        self.conv = nn.Sequential(
            nn.Conv2d(3,   32,  4, 2, 1),  # 32×32×32
            nn.LeakyReLU(0.2),

            nn.Conv2d(32,  64,  4, 2, 1),  # 64×16×16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64,  128, 4, 2, 1),  # 128×8×8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),  # 256×4×4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        # Flat size: 256 * 4 * 4 = 4096
        self.fc_mu      = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_log_var(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)

        # 256×4×4 → 3×64×64
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 128×8×8
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64,  4, 2, 1),  # 64×16×16
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64,  32,  4, 2, 1),  # 32×32×32
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32,  3,   4, 2, 1),  # 3×64×64
            nn.Tanh(),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, 4, 4)
        return self.deconv(h)


class VAE(nn.Module):
    """
    Variational AutoEncoder.

    Parameters
    ----------
    latent_dim : int
        Size of the latent space vector z (default 128).
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    # ── Reparameterisation trick ───────────────────────────────────────────────
    def reparameterise(self, mu, log_var):
        """z = mu + eps * std  where  eps ~ N(0, I)."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterise(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

    @torch.no_grad()
    def generate(self, n: int = 16, device: str = "cpu"):
        """Sample n images from the prior N(0, I)."""
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)

    @torch.no_grad()
    def reconstruct(self, x):
        """Reconstruct a batch of images (no sampling noise)."""
        mu, _ = self.encoder(x)
        return self.decoder(mu)


def vae_loss(recon_x, x, mu, log_var, beta: float = 1.0):
    """
    VAE loss = Reconstruction loss + β * KL divergence.

    beta=1   → standard VAE
    beta>1   → β-VAE (stronger disentanglement, may sacrifice quality)

    Parameters
    ----------
    recon_x  : reconstructed images (batch)
    x        : original images (batch), assumed in [-1, 1] (Tanh output)
    mu       : latent mean
    log_var  : latent log variance
    beta     : weight on the KL term (default 1.0)
    """
    # MSE works well for images; scale by batch size
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum") / x.size(0)

    # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)

    return recon_loss + beta * kl_loss, recon_loss, kl_loss
