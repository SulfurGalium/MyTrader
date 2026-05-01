"""
models/trainer.py
Improved training loop incorporating:
  - CosineAnnealingLR (better for diffusion than OneCycleLR)
  - patience=80 early stopping (diffusion loss plateaus then drops)
  - CUDA benchmark + TF32 flags for faster GPU training
  - non_blocking data transfer
  - trading-score checkpoint selection (saves best Sharpe, not best loss)
  - lower weight_decay (1e-5 vs 1e-4)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import copy

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from models.diffusion import MultiTimeframeDiffusion, save_checkpoint


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
    model:  MultiTimeframeDiffusion,
    val_arr,
    device: torch.device,
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

    # Sign accuracy: fraction of times predicted direction matches actual
    sign_acc = float((np.sign(preds) == np.sign(actuals)).mean())
    return sign_acc


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model:          MultiTimeframeDiffusion,
    train_loader,
    val_loader,
    val_arr         = None,       # raw val numpy array for trading score
    epochs:         int   = config.EPOCHS,
    lr:             float = config.LEARNING_RATE,
    patience:       int   = 80,   # large patience — diffusion plateaus then drops
    device:         str   = config.DEVICE,
    eval_interval:  int   = 5,    # compute trading score every N epochs
    use_trading_selection: bool = True,  # select checkpoint by trading score not loss
) -> list[dict]:
    """
    Returns list of epoch metric dicts.
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
        torch.backends.cudnn.benchmark    = True   # auto-tune kernels
        torch.backends.cuda.matmul.allow_tf32 = True  # faster matmul on Ampere
        torch.backends.cudnn.allow_tf32   = True
        logger.info("CUDA: benchmark=True  TF32=True")

    model = model.to(dev)
    opt   = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)  # lower WD than before
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast_ = lambda: torch.amp.autocast(device_type=dev.type, enabled=use_amp)

    # ── Checkpoint selection ──────────────────────────────────────────────────
    # If val_arr provided + trading selection enabled: save best trading score
    # Otherwise: save best val loss
    use_trade_sel = use_trading_selection and val_arr is not None
    if use_trade_sel:
        logger.info("Checkpoint selection: trading sign-accuracy (every " +
                    str(eval_interval) + " epochs)")
    else:
        logger.info("Checkpoint selection: validation loss")

    best_loss         = float("inf")
    best_trade_score  = 0.0
    best_state        = _clone_state(model)
    early_stop        = EarlyStopping(patience=patience, mode="min")
    history           = []

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

        # ── Validate loss ─────────────────────────────────────────────────────
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

        # ── Trading score (every eval_interval epochs) ────────────────────────
        trade_score = None
        if use_trade_sel and (epoch % eval_interval == 0 or epoch == epochs):
            trade_score = _quick_trading_score(model, val_arr, dev)
            model.train()  # restore train mode after eval

        # ── Checkpoint selection ──────────────────────────────────────────────
        if use_trade_sel and trade_score is not None:
            if trade_score > best_trade_score:
                best_trade_score = trade_score
                best_state       = _clone_state(model)
                save_checkpoint(model, epoch, v_loss)
                # Also save a dedicated best-trading checkpoint that is
                # NEVER overwritten by later epochs — so you can always
                # recover the peak even if training continues past it
                best_ckpt_path = config.MODEL_DIR / "diffusion_best_trading.pt"
                torch.save(
                    {"epoch": epoch, "loss": v_loss,
                     "trade_score": trade_score,
                     "state_dict": model.state_dict()},
                    best_ckpt_path,
                )
                logger.success("  ✓ checkpoint (trade_score=" +
                               str(round(trade_score, 4)) + ")" +
                               "  → also saved diffusion_best_trading.pt")
        else:
            if v_loss < best_loss:
                best_loss  = v_loss
                best_state = _clone_state(model)
                save_checkpoint(model, epoch, v_loss)
                logger.success("  ✓ checkpoint (val=" + str(round(v_loss, 5)) + ")")

        # ── Log ───────────────────────────────────────────────────────────────
        lr_now = sched.get_last_lr()[0]
        row    = {"epoch": epoch, "train_loss": t_loss,
                  "val_loss": v_loss, "lr": lr_now}
        if trade_score is not None:
            row["trade_score"] = trade_score
        history.append(row)

        msg = ("Epoch " + str(epoch).zfill(3) + "/" + str(epochs) +
               "  train=" + str(round(t_loss, 5)) +
               "  val="   + str(round(v_loss, 5)) +
               "  lr="    + f"{lr_now:.2e}")
        if trade_score is not None:
            msg += "  sign_acc=" + str(round(trade_score * 100, 1)) + "%"
        logger.info(msg)

        # ── Early stopping (on val loss — stable signal) ──────────────────────
        if early_stop(v_loss):
            logger.warning("Early stopping at epoch " + str(epoch) +
                           " (patience=" + str(patience) + ")")
            break

    # Restore best state
    model.load_state_dict(best_state)
    logger.success("Training complete. Best val=" + str(round(best_loss, 5)) +
                   ("  best_trade=" + str(round(best_trade_score, 4))
                    if use_trade_sel else ""))
    return history
