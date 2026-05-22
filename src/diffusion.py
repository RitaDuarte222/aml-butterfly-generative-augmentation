"""
diffusion.py — DDPM (Denoising Diffusion Probabilistic Model) for butterfly image generation.

Architecture
------------
Forward process : q(x_t | x_{t-1}) = N(sqrt(1-beta_t)*x_{t-1}, beta_t*I)
Reverse process : p_theta(x_{t-1} | x_t) predicted by a class-conditional UNet
UNet            : DownBlocks -> Bottleneck -> UpBlocks + time/class embeddings
                  Attention at 32x32, 16x16, and 8x8 (bottleneck) resolutions
CFG             : Classifier-Free Guidance support (guidance_scale > 1.0)

Input  : 3 x 64 x 64 RGB images in [-1, 1]
Output : 3 x 64 x 64 denoised images
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. Noise Schedules
# ─────────────────────────────────────────────

def linear_beta_schedule(timesteps: int, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0001, 0.9999)


# ─────────────────────────────────────────────
# 2. Gaussian Diffusion Utilities
# ─────────────────────────────────────────────

class GaussianDiffusion:
    """
    Precomputes all alpha/beta statistics for a given schedule.
    Provides q_sample (forward) and predict_x0 helpers.
    Supports Classifier-Free Guidance (CFG) during sampling.
    """

    def __init__(self, betas: torch.Tensor):
        self.T = len(betas)
        betas = betas.float()

        alphas          = 1.0 - betas
        alphas_cumprod  = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register("betas",               betas)
        self.register("alphas_cumprod",      alphas_cumprod)
        self.register("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register("sqrt_alphas_cumprod",          alphas_cumprod.sqrt())
        self.register("sqrt_one_minus_alphas_cumprod",(1.0 - alphas_cumprod).sqrt())
        self.register("log_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).log())
        self.register("sqrt_recip_alphas_cumprod",    (1.0 / alphas_cumprod).sqrt())
        self.register("sqrt_recipm1_alphas_cumprod",  (1.0 / alphas_cumprod - 1).sqrt())

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register("posterior_variance",           posterior_variance)
        self.register("posterior_log_variance_clipped",
                       posterior_variance.clamp(min=1e-20).log())
        self.register("posterior_mean_coef1",
                       betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod))
        self.register("posterior_mean_coef2",
                       (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod))

    def register(self, name, val):
        setattr(self, name, val)

    def _extract(self, a, t, shape):
        b = t.shape[0]
        out = a[t]
        return out.reshape(b, *((1,) * (len(shape) - 1)))

    def q_sample(self, x0, t, noise=None):
        """Forward: x_t = sqrt(alpha_bar)*x0 + sqrt(1-alpha_bar)*eps."""
        if noise is None:
            noise = torch.randn_like(x0)
        s1 = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        s2 = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return s1 * x0 + s2 * noise

    def p_losses(self, model, x0, t, class_labels=None, loss_type='l2', cfg_dropout=0.1):
        """
        Compute training loss (predict noise).
        With CFG dropout: randomly drop class labels to train unconditional path.
        """
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)

        # Classifier-Free Guidance: randomly null out class labels during training
        if class_labels is not None and cfg_dropout > 0:
            drop_mask = torch.rand(class_labels.shape[0], device=class_labels.device) < cfg_dropout
            # Use num_classes as null token (model must accept num_classes index)
            null_label = torch.full_like(class_labels, model.num_classes)
            class_labels = torch.where(drop_mask, null_label, class_labels)

        pred = model(x_noisy, t, class_labels)
        if loss_type == 'l1':
            return F.l1_loss(pred, noise)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, class_labels=None, guidance_scale=1.0):
        """Single reverse step with optional CFG."""
        betas_t = self._extract(self.betas, t, x.shape)
        s1      = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        recip   = self._extract(self.sqrt_recip_alphas_cumprod, t, x.shape)

        if guidance_scale > 1.0 and class_labels is not None:
            # CFG: run model twice — conditioned and unconditioned
            null_labels = torch.full_like(class_labels, model.num_classes)
            noise_cond   = model(x, t, class_labels)
            noise_uncond = model(x, t, null_labels)
            noise_pred   = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = model(x, t, class_labels)

        model_mean = recip * (x - betas_t / s1 * noise_pred)
        if t_index == 0:
            return model_mean
        post_log_var = self._extract(self.posterior_log_variance_clipped, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + (0.5 * post_log_var).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, class_labels=None, device='cpu', guidance_scale=1.0):
        """Full reverse diffusion loop: x_T -> x_0 with optional CFG."""
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, i, class_labels, guidance_scale=guidance_scale)
        return x

    def to(self, device):
        for attr in ['betas','alphas_cumprod','alphas_cumprod_prev',
                     'sqrt_alphas_cumprod','sqrt_one_minus_alphas_cumprod',
                     'log_one_minus_alphas_cumprod','sqrt_recip_alphas_cumprod',
                     'sqrt_recipm1_alphas_cumprod','posterior_variance',
                     'posterior_log_variance_clipped','posterior_mean_coef1',
                     'posterior_mean_coef2']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self


# ─────────────────────────────────────────────
# 3. UNet Building Blocks
# ─────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding for timestep t."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    """
    Residual block with time and class conditioning.
    num_classes+1 embeddings: indices 0..num_classes-1 = real classes,
    index num_classes = unconditional (null) token for CFG.
    """
    def __init__(self, in_ch, out_ch, time_dim, num_classes=0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.drop  = nn.Dropout(dropout)

        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch * 2))
        if num_classes > 0:
            # +1 for unconditional (null) token
            self.class_emb = nn.Embedding(num_classes + 1, out_ch)
        else:
            self.class_emb = None
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, class_emb=None):
        h = self.conv1(F.silu(self.norm1(x)))
        # time conditioning
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
        h = h * (scale[:, :, None, None] + 1) + shift[:, :, None, None]
        # class conditioning (includes null token for CFG)
        if self.class_emb is not None and class_emb is not None:
            h = h + self.class_emb(class_emb)[:, :, None, None]
        h = self.drop(self.conv2(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Attention(nn.Module):
    """Multi-head self-attention for spatial feature maps."""
    def __init__(self, ch, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).reshape(B, C, H, W)


class Down(nn.Module):
    """Downsampling block with optional attention."""
    def __init__(self, in_ch, out_ch, time_dim, num_classes=0, use_attn=False):
        super().__init__()
        self.res  = ResBlock(in_ch, out_ch, time_dim, num_classes)
        self.attn = Attention(out_ch) if use_attn else nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t, c=None):
        x = self.res(x, t, c)
        x = self.attn(x)
        return self.down(x), x   # return downsampled + skip


class Up(nn.Module):
    """Upsampling block with optional attention."""
    def __init__(self, in_ch, skip_ch, out_ch, time_dim, num_classes=0, use_attn=False):
        super().__init__()
        self.up  = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.res = ResBlock(in_ch + skip_ch, out_ch, time_dim, num_classes)
        self.attn = Attention(out_ch) if use_attn else nn.Identity()

    def forward(self, x, skip, t, c=None):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t, c)
        return self.attn(x)


# ─────────────────────────────────────────────
# 4. Class-conditional UNet with CFG support
# ─────────────────────────────────────────────

class UNet(nn.Module):
    """
    Compact UNet noise predictor for 64x64 images.
    Supports Classifier-Free Guidance (CFG) via null class token.
    Attention at 32x32, 16x16, and bottleneck (8x8).

    Parameters
    ----------
    base_ch     : base channel count (multiplied per level)
    num_classes : number of conditioning classes (0 = unconditional)
    time_dim    : sinusoidal time embedding dimension
    """
    def __init__(self, base_ch=64, num_classes=75, time_dim=256):
        super().__init__()
        self.num_classes = num_classes  # stored for CFG null token

        self.time_mlp = nn.Sequential(
            SinusoidalPE(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        nc = num_classes
        # Encoder
        # 64x64 -> 32x32 (with attention)
        self.in_conv = nn.Conv2d(3, base_ch, 3, padding=1)
        self.d1 = Down(base_ch,      base_ch * 2, time_dim, nc, use_attn=True)   # ->32, attn
        self.d2 = Down(base_ch * 2,  base_ch * 4, time_dim, nc, use_attn=True)   # ->16, attn
        self.d3 = Down(base_ch * 4,  base_ch * 8, time_dim, nc, use_attn=False)  # ->8

        # Bottleneck with attention
        self.mid1  = ResBlock(base_ch * 8, base_ch * 8, time_dim, nc)
        self.attn  = Attention(base_ch * 8)
        self.mid2  = ResBlock(base_ch * 8, base_ch * 8, time_dim, nc)

        # Decoder
        self.u1 = Up(base_ch * 8, base_ch * 8, base_ch * 4, time_dim, nc, use_attn=False) # ->16
        self.u2 = Up(base_ch * 4, base_ch * 4, base_ch * 2, time_dim, nc, use_attn=True)  # ->32, attn
        self.u3 = Up(base_ch * 2, base_ch * 2, base_ch,     time_dim, nc, use_attn=True)  # ->64, attn

        self.out_conv = nn.Sequential(
            nn.GroupNorm(min(8, base_ch), base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, 3, 1),
        )

    def forward(self, x, t, class_labels=None):
        t_emb = self.time_mlp(t)

        x = self.in_conv(x)
        x, s1 = self.d1(x, t_emb, class_labels)
        x, s2 = self.d2(x, t_emb, class_labels)
        x, s3 = self.d3(x, t_emb, class_labels)

        x = self.mid1(x, t_emb, class_labels)
        x = self.attn(x)
        x = self.mid2(x, t_emb, class_labels)

        x = self.u1(x, s3, t_emb, class_labels)
        x = self.u2(x, s2, t_emb, class_labels)
        x = self.u3(x, s1, t_emb, class_labels)

        return self.out_conv(x)
