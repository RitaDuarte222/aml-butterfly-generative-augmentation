"""
autoencoder.py — Variational AutoEncoder (VAE) for butterfly image generation.

Architecture
------------
Encoder : Conv layers -> Flatten -> FC -> (mu, log_var)
Sampling: z = mu + eps * std   (reparameterisation trick)
Decoder : FC -> Reshape -> ConvTranspose layers -> Tanh

Input  : 3 x 64 x 64 RGB images (normalised to [-1, 1])
Latent : configurable (default 128)
Output : 3 x 64 x 64 RGB images in [-1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        # 3x64x64 -> 256x4x4  (each Conv2d stride=2 halves spatial dims)
        self.conv = nn.Sequential(
            nn.Conv2d(3,   32,  4, 2, 1), nn.LeakyReLU(0.2),          # 32x32
            nn.Conv2d(32,  64,  4, 2, 1), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2),  # 16x16
            nn.Conv2d(64,  128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),  # 8x8
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),  # 4x4
        )
        self.fc_mu      = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_log_var(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        # 256x4x4 -> 3x64x64
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),  # 8x8
            nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU(),  # 16x16
            nn.ConvTranspose2d(64,  32,  4, 2, 1), nn.BatchNorm2d(32),  nn.ReLU(),  # 32x32
            nn.ConvTranspose2d(32,  3,   4, 2, 1), nn.Tanh(),                        # 64x64
        )

    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, 256, 4, 4))


class VAE(nn.Module):
    """
    Variational AutoEncoder (beta-VAE).

    Parameters
    ----------
    latent_dim : int   — size of the latent space z (default 128)
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterise(self, mu, log_var):
        """z = mu + eps * std,  eps ~ N(0, I)  — reparameterisation trick."""
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        log_var = torch.clamp(log_var, min=-20.0, max=20.0)  # prevent overflow/NaN on MPS
        z = self.reparameterise(mu, log_var)
        return self.decoder(z), mu, log_var

    @torch.no_grad()
    def generate(self, n: int = 16, device='cpu'):
        """Sample n images from the prior N(0, I)."""
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)

    @torch.no_grad()
    def reconstruct(self, x):
        """Encode then decode without sampling noise (uses mu directly)."""
        mu, _ = self.encoder(x)
        return self.decoder(mu)


def vae_loss(recon_x, x, mu, log_var, beta: float = 1.0):
    """
    ELBO loss = Reconstruction (MSE) + beta * KL divergence.

    beta = 1  -> standard VAE
    beta > 1  -> beta-VAE (stronger disentanglement)

    All terms are averaged over the batch.
    """
    recon = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    kl    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
    return recon + beta * kl, recon, kl
