"""
models/trainer.py
Training loop for MultiTimeframeDiffusion.

LR schedule: linear warmup (10% of epochs) → cosine decay to eta_min=1e-6.
This replaced the original CosineAnnealingLR-from-epoch-1 approach which
decayed the LR too aggressively before the model had converged — causing the
flat loss plateau seen from epoch 20 onward in early runs.

On the fixed-vs-scheduled LR question:
  A fixed LR is NOT proven more profitable. What matters is the *shape*:
  - Fixed LR: simple, avoids late-stage decay killing gradients, but
    risks overshooting sharp minima and never annealing to convergence.
  - Cosine with warmup (this file): the warmup prevents large early updates
    from a randomly-initialised model destroying the initial loss landscape.
    The cosine tail lets the model settle into a sharper minimum without
    oscillating. This is the consensus best practice for transformer training
    (Vaswani 2017, Chinchilla 2022).
  - OneCycleLR: aggressive — works well for CNNs and supervised classification
    but its fast rise-and-fall can destabilise diffusion model training where
    the loss landscape is much flatter.

  The key insight: the original schedule was decaying from 3e-4 to 3e-6 over
  50 epochs with no warmup. By epoch 20 the LR was already at 1.97e-4 and the
  model hadn't had a chance to properly explore. Warmup + longer cosine tail
  gives it that chance.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import copy

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from models.diffusion import MultiTimeframeDiffusion, save_checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule: linear warmup + cosine decay
# ─────────────────────────────────────────────────────────────────────────────

def _warmup_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs:  int,
    eta_min_ratio: float = 0.01,   # eta_min = lr * eta_min_ratio
) -> LambdaLR:
    """
    Linear warmup for `warmup_epochs`, then cosine decay to lr * eta_min_ratio.

    Warmup prevents the large random-init gradients from destroying the early
    loss landscape. Cosine decay lets the model settle without oscillating.

    warmup_epochs = max(1, int(total_epochs * 0.10)) gives a 10% warmup —
    e.g. 5 epochs warm-up for a 50-epoch run, 10 for 100 epochs.
    """
    def _lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            # Linear ramp 0 → 1
            return float(epoch + 1) / float(max(1, warmup_epochs))
        # Cosine decay from 1 → eta_min_ratio
        progress = float(epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs)
        )
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clone_state(model: nn.Module) -> dict:
    return copy.deepcopy(model.state_dict())


class EarlyStopping:
    def __init__(self, patience: int = 80, mode: str = "min"):
        self.patience  = patience
        self.mode      = mode
        self.best      = float("inf") if mode == "min" else float("-inf")
        self.counter   = 0

    def __call__(self, value: float) -> bool:
        improved = value < self.best if self.mode == "min" else value > self.best
        if improved:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def _quick_trading_score(
    model:     MultiTimeframeDiffusion,
    val_arr,
    device:    torch.device,
    n_samples: int = 200,
) -> float:
    """
    Fast in-training trading score: sample predictions on a random subset
    of val windows and compute sign-accuracy against actual next returns.
    Returns value in [0, 1] — higher is better.
    Used as checkpoint selection metric every eval_interval epochs.
    """
    import numpy as np
    from data.dataset import SPYWindowDataset

    ds  = SPYWindowDataset(val_arr)
    idx = np.random.choice(len(ds), size=min(n_samples, len(ds)), replace=False)

    preds, actuals = [], []
    model.eval()
    with torch.no_grad():
        for i in idx:
            fine, coarse, target = ds[i]
            fine   = fine.unsqueeze(0).to(device)
            coarse = coarse.unsqueeze(0).to(device)
            pred   = model.sample(fine, coarse, steps=10).item()
            preds.append(pred)
            actuals.append(target.item())

    preds   = np.array(preds)
    actuals = np.array(actuals)
    sign_acc = float((np.sign(preds) == np.sign(actuals)).mean())
    return sign_acc


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model:          MultiTimeframeDiffusion,
    train_loader,
    val_loader,
    val_arr         = None,
    epochs:         int   = config.EPOCHS,
    lr:             float = config.LEARNING_RATE,
    patience:       int   = 80,
    device:         str   = config.DEVICE,
    eval_interval:  int   = 5,
    use_trading_selection: bool = True,
    warmup_ratio:   float = 0.10,  # fraction of epochs used for LR warmup
) -> list[dict]:
    """
    Returns list of epoch metric dicts.

    warmup_ratio: fraction of total epochs devoted to linear LR warmup.
      0.10 (default) = 5 epochs warmup on a 50-epoch run.
      Set to 0.0 to disable warmup entirely (reverts to pure cosine from ep 1).
    """
    # ── Device setup ──────────────────────────────────────────────────────────
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU.")
        device = "cpu"

    dev     = torch.device(device)
    use_amp = dev.type == "cuda"
    logger.info("Training on: " + str(dev) + "  AMP=" + str(use_amp))

    # ── CUDA optimisations ────────────────────────────────────────────────────
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark        = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        logger.info("CUDA: benchmark=True  TF32=True")

    model = model.to(dev)

    # AdamW: lr=1e-3 is a better starting point for transformer training than
    # 3e-4 — the warmup prevents it from being too aggressive at init.
    # weight_decay=1e-5 kept low (diffusion models are sensitive to over-regularisation).
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-5, betas=(0.9, 0.999))

    # Warmup + cosine schedule
    warmup_epochs = max(1, int(epochs * warmup_ratio))
    sched = _warmup_cosine_schedule(opt, warmup_epochs, epochs, eta_min_ratio=0.01)

    logger.info(
        f"LR schedule: warmup {warmup_epochs} ep → cosine decay  "
        f"peak={lr:.1e}  eta_min={lr*0.01:.1e}"
    )

    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast_ = lambda: torch.amp.autocast(device_type=dev.type, enabled=use_amp)

    # ── Checkpoint selection ──────────────────────────────────────────────────
    use_trade_sel = use_trading_selection and val_arr is not None
    if use_trade_sel:
        logger.info(
            "Checkpoint selection: trading sign-accuracy (every "
            + str(eval_interval) + " epochs)"
        )
    else:
        logger.info("Checkpoint selection: validation loss")

    best_loss        = float("inf")
    best_trade_score = 0.0
    best_state       = _clone_state(model)
    early_stop       = EarlyStopping(patience=patience, mode="min")
    history          = []

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        t_loss = 0.0
        for fine, coarse, target in train_loader:
            fine   = fine.to(dev,   non_blocking=True)
            coarse = coarse.to(dev, non_blocking=True)
            target = target.to(dev, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast_():
                loss = model(fine, coarse, target)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            t_loss += loss.item()

        t_loss /= max(1, len(train_loader))
        sched.step()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for fine, coarse, target in val_loader:
                fine   = fine.to(dev,   non_blocking=True)
                coarse = coarse.to(dev, non_blocking=True)
                target = target.to(dev, non_blocking=True)
                with autocast_():
                    v_loss += model(fine, coarse, target).item()
        v_loss /= max(1, len(val_loader))

        # ── Trading score ─────────────────────────────────────────────────────
        trade_score = None
        if use_trade_sel and (epoch % eval_interval == 0 or epoch == epochs):
            trade_score = _quick_trading_score(model, val_arr, dev)
            model.train()

        # ── Checkpoint selection ──────────────────────────────────────────────
        if use_trade_sel and trade_score is not None:
            if trade_score > best_trade_score:
                best_trade_score = trade_score
                best_state       = _clone_state(model)
                save_checkpoint(model, epoch, v_loss)
                best_ckpt_path = config.MODEL_DIR / "diffusion_best_trading.pt"
                torch.save(
                    {"epoch": epoch, "loss": v_loss,
                     "trade_score": trade_score,
                     "state_dict": model.state_dict()},
                    best_ckpt_path,
                )
                logger.success(
                    "  ✓ checkpoint (trade_score=" + str(round(trade_score, 4)) + ")"
                    + "  → diffusion_best_trading.pt"
                )
        else:
            if v_loss < best_loss:
                best_loss  = v_loss
                best_state = _clone_state(model)
                save_checkpoint(model, epoch, v_loss)
                logger.success("  ✓ checkpoint (val=" + str(round(v_loss, 5)) + ")")

        # ── Logging ───────────────────────────────────────────────────────────
        lr_now = sched.get_last_lr()[0]
        row    = {"epoch": epoch, "train_loss": t_loss,
                  "val_loss": v_loss, "lr": lr_now}
        if trade_score is not None:
            row["trade_score"] = trade_score
        history.append(row)

        # Flag warmup phase in log so it's obvious
        phase = " [warmup]" if epoch <= warmup_epochs else ""
        msg   = (
            "Epoch " + str(epoch).zfill(3) + "/" + str(epochs)
            + "  train=" + str(round(t_loss, 5))
            + "  val="   + str(round(v_loss, 5))
            + "  lr="    + f"{lr_now:.2e}"
            + phase
        )
        if trade_score is not None:
            msg += "  sign_acc=" + str(round(trade_score * 100, 1)) + "%"
        logger.info(msg)

        # ── Early stopping ────────────────────────────────────────────────────
        # Don't trigger early stopping during warmup — loss is still rising
        if epoch > warmup_epochs and early_stop(v_loss):
            logger.warning(
                "Early stopping at epoch " + str(epoch)
                + " (patience=" + str(patience) + ")"
            )
            break

    model.load_state_dict(best_state)
    logger.success(
        "Training complete. Best val=" + str(round(best_loss, 5))
        + ("  best_trade=" + str(round(best_trade_score, 4)) if use_trade_sel else "")
    )
    return history
