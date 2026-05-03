"""
models/diffusion.py
Multi-Timeframe Diffusion Model for intraday return forecasting.

Architecture overview
─────────────────────
  Fine encoder   : Transformer over 5-min windows              (L × F)
  Coarse encoder : Transformer over aggregated context          (L/k × F)
  Fusion          : Cross-attention — fine queries, coarse keys/values
  Diffusion head  : MLP that denoises a Gaussian sample conditioned
                    on the fused context → predicts next log return.

Root causes fixed in this version
──────────────────────────────────
  1. Target scale mismatch (PRIMARY cause of flat 0.69 loss):
     SPY 30-min log returns are ~0.001-0.003 in magnitude.
     The diffusion forward process adds noise ε ~ N(0,1).
     SNR = sqrt(ā)*0.002 / sqrt(1-ā)*1.0 ≈ 0.002 at mid-diffusion —
     the signal is 500x smaller than the noise. The model learns to
     predict ε and completely ignores x0. No amount of epochs fixes this.
     Fix: store target_scale = std(train targets) at fit time, normalise
     x0 → x0/scale before diffusion, denormalise at inference.

  2. Broken DDIM sampling formula (cause of epoch-50/59 explosions):
     Old code: x = (x - sqrt(1-a)*eps) / sqrt(a)
     This is the formula for x0_pred (estimating clean signal), NOT
     the denoising step to x_{t-1}. At t→0 where a_bar→0, sqrt(a)→0
     and the division explodes x to 1e4+, corrupting model weights.
     Fix: proper two-step DDIM update — compute x0_pred first (clamped),
     then compute x_{t-1} = sqrt(a_prev)*x0_pred + sqrt(1-a_prev)*eps.

  3. No numerical guards (cause of NaN cascade after explosion):
     Fix: clamp x_t to [-5, 5] in forward() and x in sample() loop.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Noise schedule
# ─────────────────────────────────────────────────────────────────────────────

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule (Nichol & Dhariwal 2021)."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-4, 0.9999).float()


# ─────────────────────────────────────────────────────────────────────────────
# Positional encoding
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ─────────────────────────────────────────────────────────────────────────────
# Transformer encoder branch
# ─────────────────────────────────────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        in_dim:   int,
        d_model:  int = 128,
        nhead:    int = 4,
        n_layers: int = 3,
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pe   = SinusoidalPE(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.pe(x)
        return self.norm(self.encoder(x))


# ─────────────────────────────────────────────────────────────────────────────
# Cross-attention fusion
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True, dropout=0.1
        )
        self.norm = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, fine: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(fine, coarse, coarse)
        fused = self.norm(fine + attn_out)
        fused = fused + self.ff(fused)
        return fused.mean(dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Timestep embedding
# ─────────────────────────────────────────────────────────────────────────────

class TimestepEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, t: torch.Tensor, d_model: int) -> torch.Tensor:
        half  = d_model // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.mlp(emb)


# ─────────────────────────────────────────────────────────────────────────────
# Denoising head
# ─────────────────────────────────────────────────────────────────────────────

class DenoisingHead(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.t_emb = TimestepEmbedding(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model + d_model + 1, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_emb(t, context.size(-1))
        inp   = torch.cat([x_t, context, t_emb], dim=-1)
        return self.net(inp)


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class MultiTimeframeDiffusion(nn.Module):
    def __init__(
        self,
        feature_dim:     int = config.FEATURE_DIM,
        d_model:         int = 128,
        nhead:           int = 4,
        n_enc_layers:    int = 3,
        diffusion_steps: int = config.DIFFUSION_STEPS,
        target_scale:    float = 1.0,   # set by trainer after seeing train targets
    ):
        super().__init__()
        self.T            = diffusion_steps
        self.target_scale = target_scale   # stored for inference denormalisation

        self.fine_enc   = TransformerEncoder(feature_dim, d_model, nhead, n_enc_layers)
        self.coarse_enc = TransformerEncoder(feature_dim, d_model, nhead, n_enc_layers)
        self.fusion     = CrossAttentionFusion(d_model, nhead)
        self.denoise    = DenoisingHead(d_model)

        betas  = cosine_beta_schedule(diffusion_steps)
        alphas = 1.0 - betas
        a_bar  = torch.cumprod(alphas, dim=0)

        # Prepend a_bar[0]=1.0 so a_bar_prev[t] = a_bar[t-1] for t>=1
        a_bar_prev = torch.cat([torch.ones(1), a_bar[:-1]])

        self.register_buffer("betas",          betas)
        self.register_buffer("alphas",         alphas)
        self.register_buffer("a_bar",          a_bar)
        self.register_buffer("a_bar_prev",     a_bar_prev)
        self.register_buffer("sqrt_abar",      a_bar.sqrt())
        self.register_buffer("sqrt_one_minus", (1 - a_bar).sqrt())

    # ── Forward (training) ────────────────────────────────────────────────────
    def forward(
        self,
        fine:   torch.Tensor,   # (B, L,   F)
        coarse: torch.Tensor,   # (B, L/k, F)
        target: torch.Tensor,   # (B,) — next log return (raw scale)
    ) -> torch.Tensor:
        B      = fine.size(0)
        device = fine.device

        f_enc = self.fine_enc(fine)
        c_enc = self.coarse_enc(coarse)
        ctx   = self.fusion(f_enc, c_enc)    # (B, D)

        t   = torch.randint(0, self.T, (B,), device=device)

        # ── Fix 1: normalise target to N(0,1) scale ──────────────────────────
        # Divide by target_scale (std of training targets) so the diffusion
        # process operates on unit-variance data. Without this, x0 ≈ 0.002
        # and the SNR is ~0.002, making it impossible for the model to learn
        # the signal — it just learns to predict noise.
        x0  = (target / self.target_scale).unsqueeze(1).clamp(-10, 10)  # (B,1)
        eps = torch.randn_like(x0)

        x_t = (self.sqrt_abar[t].unsqueeze(1) * x0 +
               self.sqrt_one_minus[t].unsqueeze(1) * eps)

        # ── Fix 3: clamp x_t to prevent extreme inputs to DenoisingHead ─────
        x_t = x_t.clamp(-10, 10)

        eps_pred = self.denoise(x_t, t, ctx)
        return F.mse_loss(eps_pred, eps)

    # ── Inference: DDIM fast sampling ─────────────────────────────────────────
    @torch.no_grad()
    def sample(
        self,
        fine:   torch.Tensor,
        coarse: torch.Tensor,
        steps:  int = 50,
    ) -> torch.Tensor:    # (B,) predicted next log returns in ORIGINAL scale
        device = fine.device
        B      = fine.size(0)

        f_enc = self.fine_enc(fine)
        c_enc = self.coarse_enc(coarse)
        ctx   = self.fusion(f_enc, c_enc)

        ts = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=device)
        x  = torch.randn(B, 1, device=device)

        for i, t_val in enumerate(ts):
            t_batch  = t_val.expand(B)
            eps_pred = self.denoise(x, t_batch, ctx)

            a      = self.a_bar[t_val]
            a_prev = self.a_bar_prev[t_val]

            # ── Fix 2: correct DDIM two-step update ───────────────────────────
            # Step 1: estimate clean signal x0 from noisy x_t
            #   x0_pred = (x_t - sqrt(1-a)*eps_pred) / sqrt(a)
            # Step 2: project to x_{t-1} using a_prev
            #   x_prev = sqrt(a_prev)*x0_pred + sqrt(1-a_prev)*eps_pred
            # Clamping sqrt(a) prevents division explosion near t=0 where a→0.
            # Old code used only step 1 and set x = x0_pred directly, which
            # amplified x by 1/sqrt(a) → 100x+ at the last denoising steps.
            sqrt_a      = a.sqrt().clamp(min=1e-6)
            x0_pred     = ((x - (1 - a).sqrt() * eps_pred) / sqrt_a).clamp(-10, 10)
            x           = a_prev.sqrt() * x0_pred + (1 - a_prev).sqrt() * eps_pred
            x           = x.clamp(-10, 10)

        # ── Denormalise: scale back to original log return magnitude ─────────
        return x.squeeze(1) * self.target_scale


# ─────────────────────────────────────────────────────────────────────────────
# Save / load helpers
# ─────────────────────────────────────────────────────────────────────────────

CKPT_PATH = config.MODEL_DIR / "diffusion_latest.pt"


def save_checkpoint(model: MultiTimeframeDiffusion, epoch: int, loss: float):
    torch.save(
        {"epoch": epoch, "loss": loss,
         "target_scale": model.target_scale,
         "state_dict": model.state_dict()},
        CKPT_PATH,
    )


def load_checkpoint(model: MultiTimeframeDiffusion) -> dict:
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    if "target_scale" in ckpt:
        model.target_scale = ckpt["target_scale"]
    return ckpt


Architecture overview
─────────────────────
  Fine encoder   : Transformer over 5-min windows              (L × F)
  Coarse encoder : Transformer over aggregated context          (L/k × F)
  Fusion          : Cross-attention — fine queries, coarse keys/values
  Diffusion head  : Lightweight U-Net-style MLP that iteratively
                    denoises a Gaussian noise sample conditioned
                    on the fused context → predicts next log return.

The diffusion process uses a cosine noise schedule and is trained
with a simple ε-prediction MSE loss (DDPM-style).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Noise schedule
# ─────────────────────────────────────────────────────────────────────────────

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule (Nichol & Dhariwal 2021)."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-4, 0.9999).float()


# ─────────────────────────────────────────────────────────────────────────────
# Positional encoding
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, L, D)
        return x + self.pe[:, : x.size(1)]


# ─────────────────────────────────────────────────────────────────────────────
# Transformer encoder branch
# ─────────────────────────────────────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        in_dim:   int,
        d_model:  int = 128,
        nhead:    int = 4,
        n_layers: int = 3,
        dropout:  float = 0.2,   # increased from 0.1 to slow overfitting
    ):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pe   = SinusoidalPE(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        # enable_nested_tensor=False: norm_first=True disables nested tensor
        # optimisation anyway; setting it explicitly suppresses the UserWarning
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, L, F) → (B, L, D)
        x = self.proj(x)
        x = self.pe(x)
        return self.norm(self.encoder(x))


# ─────────────────────────────────────────────────────────────────────────────
# Cross-attention fusion
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True, dropout=0.1
        )
        self.norm = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(
        self,
        fine:   torch.Tensor,    # (B, L_fine,   D) — queries
        coarse: torch.Tensor,    # (B, L_coarse, D) — keys / values
    ) -> torch.Tensor:           # (B, D)  — pooled context
        attn_out, _ = self.attn(fine, coarse, coarse)
        fused = self.norm(fine + attn_out)
        fused = fused + self.ff(fused)
        return fused.mean(dim=1)   # mean-pool along time


# ─────────────────────────────────────────────────────────────────────────────
# Time-step embedding for diffusion
# ─────────────────────────────────────────────────────────────────────────────

class TimestepEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, t: torch.Tensor, d_model: int) -> torch.Tensor:
        # Sinusoidal embedding
        half = d_model // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.mlp(emb)


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion denoising head
# ─────────────────────────────────────────────────────────────────────────────

class DenoisingHead(nn.Module):
    """Conditioned on context + timestep, predicts added noise ε."""
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.t_emb = TimestepEmbedding(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model + d_model + 1, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        x_t:     torch.Tensor,   # (B, 1)  — noisy sample
        t:       torch.Tensor,   # (B,)    — diffusion timestep
        context: torch.Tensor,   # (B, D)  — fused market context
    ) -> torch.Tensor:           # (B, 1)
        t_emb = self.t_emb(t, context.size(-1))
        inp   = torch.cat([x_t, context, t_emb], dim=-1)
        return self.net(inp)


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class MultiTimeframeDiffusion(nn.Module):
    def __init__(
        self,
        feature_dim:     int = config.FEATURE_DIM,
        d_model:         int = 128,
        nhead:           int = 4,
        n_enc_layers:    int = 3,
        diffusion_steps: int = config.DIFFUSION_STEPS,
    ):
        super().__init__()
        self.T = diffusion_steps

        # Encoders
        self.fine_enc   = TransformerEncoder(feature_dim, d_model, nhead, n_enc_layers)
        self.coarse_enc = TransformerEncoder(feature_dim, d_model, nhead, n_enc_layers)
        self.fusion     = CrossAttentionFusion(d_model, nhead)
        self.denoise    = DenoisingHead(d_model)

        # Noise schedule buffers
        betas   = cosine_beta_schedule(diffusion_steps)
        alphas  = 1.0 - betas
        a_bar   = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas",   betas)
        self.register_buffer("alphas",  alphas)
        self.register_buffer("a_bar",   a_bar)
        self.register_buffer("sqrt_abar",       a_bar.sqrt())
        self.register_buffer("sqrt_one_minus",  (1 - a_bar).sqrt())

    # ── Forward (training) ────────────────────────────────────────────────────
    def forward(
        self,
        fine:   torch.Tensor,    # (B, L,   F)
        coarse: torch.Tensor,    # (B, L/k, F)
        target: torch.Tensor,    # (B,)  — next log return (clean x_0)
    ) -> torch.Tensor:           # scalar loss
        B = fine.size(0)
        device = fine.device

        # Encode market context
        f_enc = self.fine_enc(fine)
        c_enc = self.coarse_enc(coarse)
        ctx   = self.fusion(f_enc, c_enc)    # (B, D)

        # Sample random diffusion timestep
        t = torch.randint(0, self.T, (B,), device=device)

        # Forward diffusion: q(x_t | x_0) = N(√ā·x_0, (1-ā)·I)
        x0  = target.unsqueeze(1)            # (B, 1)
        eps = torch.randn_like(x0)
        x_t = self.sqrt_abar[t].unsqueeze(1) * x0 + \
              self.sqrt_one_minus[t].unsqueeze(1) * eps

        # Predict noise
        eps_pred = self.denoise(x_t, t, ctx)
        return F.mse_loss(eps_pred, eps)

    # ── Inference: DDIM-style fast sampling ───────────────────────────────────
    @torch.no_grad()
    def sample(
        self,
        fine:   torch.Tensor,
        coarse: torch.Tensor,
        steps:  int = 50,
    ) -> torch.Tensor:           # (B,) predicted next log returns
        device = fine.device
        f_enc = self.fine_enc(fine)
        c_enc = self.coarse_enc(coarse)
        ctx   = self.fusion(f_enc, c_enc)

        # DDIM subset of timesteps (evenly spaced, reversed)
        ts = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=device)
        x  = torch.randn(fine.size(0), 1, device=device)

        for t_val in ts:
            t_batch = t_val.expand(fine.size(0))
            eps_pred = self.denoise(x, t_batch, ctx)
            a    = self.a_bar[t_val]
            x    = (x - (1 - a).sqrt() * eps_pred) / a.sqrt()

        return x.squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# Save / load helpers
# ─────────────────────────────────────────────────────────────────────────────

CKPT_PATH = config.MODEL_DIR / "diffusion_latest.pt"


def save_checkpoint(model: MultiTimeframeDiffusion, epoch: int, loss: float):
    torch.save(
        {"epoch": epoch, "loss": loss, "state_dict": model.state_dict()},
        CKPT_PATH,
    )


def load_checkpoint(model: MultiTimeframeDiffusion) -> dict:
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    return ckpt
