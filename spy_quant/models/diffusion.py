"""
models/diffusion.py
Multi-Timeframe Diffusion Model for intraday return forecasting.

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
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
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
