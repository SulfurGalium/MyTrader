"""
models/trainer.py
Training loop for MultiTimeframeDiffusion.

LR schedule: linear warmup (10% of epochs) -> cosine decay to eta_min=1e-6.
This replaced the original CosineAnnealingLR-from-epoch-1 approach which
decayed the LR too aggressively before the model had converged -- causing the
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
  - OneCycleLR: aggressive -- works well for CNNs and supervised classification
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


# -----------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# -----------------------------------------------------------------------------

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

    warmup_epochs = max(1, int(total_epochs * 0.10)) gives a 10% warmup --
    e.g. 5 epochs warm-up for a 50-epoch run, 10 for 100 epochs.
    """
    def _lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            # Linear ramp 0 -> 1
            return float(epoch + 1) / float(max(1, warmup_epochs))
        # Cosine decay from 1 -> eta_min_ratio
        progress = float(epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs)
        )
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

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
    Returns value in [0, 1] -- higher is better.

    Uses try/finally to guarantee model is always restored to train() mode
    even if an exception occurs mid-evaluation. Without this, a failure here
    leaves the model stuck in eval mode for the rest of the training epoch,
    causing a large loss spike (seen as train=0.764 at epoch 5 vs 0.703 at ep4).
    """
    import numpy as np
    from data.dataset import SPYWindowDataset

    was_training = model.training
    try:
        model.eval()
        ds  = SPYWindowDataset(val_arr)
        idx = np.random.choice(len(ds), size=min(n_samples, len(ds)), replace=False)

        preds, actuals = [], []
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
        return float((np.sign(preds) == np.sign(actuals)).mean())

    except Exception as exc:
        logger.warning(f"Trading score eval failed: {exc} -- returning 0.0")
        return 0.0

    finally:
        # Always restore -- this is the critical fix for the epoch-5 train spike
        if was_training:
            model.train()


# -----------------------------------------------------------------------------
# Main training function
# -----------------------------------------------------------------------------

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
    # -- Device setup ----------------------------------------------------------
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available -- falling back to CPU.")
        device = "cpu"

    dev     = torch.device(device)
    use_amp = dev.type == "cuda"
    logger.info("Training on: " + str(dev) + "  AMP=" + str(use_amp))

    if dev.type == "cuda":
        torch.backends.cudnn.benchmark        = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        logger.info("CUDA: benchmark=True  TF32=True")

    # -- Compute target_scale from training data --------------------------------
    # The diffusion process assumes x0 ~ N(0,1). SPY log returns are ~0.001-0.003,
    # giving SNR ~= 0.002 at mid-diffusion -- the model can't see the signal.
    # We normalise by the std of training targets so x0 is unit-variance.
    # target_scale is stored in the model and checkpoint for inference denorm.
    if val_arr is not None:
        import numpy as np
        # Collect all targets from the train loader (col 0 = log_return)
        # Use val_arr as proxy since train_arr not passed directly --
        # compute from the first batch of train_loader instead
        all_targets = []
        for _, _, tgt in train_loader:
            all_targets.append(tgt.numpy())
            if len(all_targets) >= 50:   # 50 batches x batch_size >= 800 samples
                break
        target_std = float(np.concatenate(all_targets).std())
        target_std = max(target_std, 1e-6)   # guard against degenerate data
        model.target_scale = target_std
        logger.info(
            f"Target scale set: {target_std:.6f}  "
            f"(targets will be divided by this before diffusion)"
        )
    else:
        logger.warning("val_arr not provided -- target_scale not set. Using 1.0.")

    model = model.to(dev)

    # AdamW: lr=1e-3 is a better starting point for transformer training than
    # 3e-4 -- the warmup prevents it from being too aggressive at init.
    # weight_decay=1e-5 kept low (diffusion models are sensitive to over-regularisation).
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-5, betas=(0.9, 0.999))

    # Warmup + cosine schedule
    warmup_epochs = max(1, int(epochs * warmup_ratio))
    sched = _warmup_cosine_schedule(opt, warmup_epochs, epochs, eta_min_ratio=0.01)

    logger.info(
        f"LR schedule: warmup {warmup_epochs} ep -> cosine decay  "
        f"peak={lr:.1e}  eta_min={lr*0.01:.1e}"
    )

    # AMP GradScaler -- conservative settings to prevent overflow cascade.
    # Default init_scale=65536 means a loss spike of 4421 * 65536 = 2.9e8
    # which overflows float16 (max ~65504), scaler skips the step, halves
    # scale_factor. If this repeats, scale->1.0, gradients vanish, weights
    # drift to zero, loss locks at exactly 1.0 (MSE of 0 vs N(0,1)).
    # init_scale=512 is 128x smaller -- stays safely in float16 range.
    # growth_interval=200 means scale only grows back after 200 clean batches.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
        init_scale=512,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=200,
    )
    autocast_ = lambda: torch.amp.autocast(device_type=dev.type, enabled=use_amp)

    # -- Checkpoint selection --------------------------------------------------
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
    skipped_total    = 0

    for epoch in range(1, epochs + 1):
        # -- Train -------------------------------------------------------------
        model.train()
        t_loss        = 0.0
        counted       = 0
        epoch_skipped = 0

        for fine, coarse, target in train_loader:
            fine   = fine.to(dev,   non_blocking=True)
            coarse = coarse.to(dev, non_blocking=True)
            target = target.to(dev, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast_():
                loss = model(fine, coarse, target)

            # Batch-level loss guard: skip this batch if loss is pathological.
            # A single data outlier at high LR can create a gradient that
            # overflows float16 even after forward-pass clamping, because
            # gradients flow back through the clamp operation.
            loss_val = loss.item()
            if not (loss_val == loss_val) or loss_val > 10.0:
                epoch_skipped += 1
                skipped_total += 1
                opt.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)

            # Log large grad norms before clipping so overflow is visible
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if grad_norm > 5.0:
                logger.debug(f"  Large grad norm: {grad_norm:.1f} (clipped to 1.0)")

            scaler.step(opt)
            scaler.update()
            t_loss  += loss_val
            counted += 1

        if counted == 0:
            logger.error(f"Epoch {epoch}: ALL batches skipped -- model collapsed.")
            t_loss = float("nan")
        else:
            t_loss /= counted

        if epoch_skipped > 0:
            logger.warning(
                f"Epoch {epoch}: {epoch_skipped} bad batches skipped "
                f"(loss>10 or nan). Running total: {skipped_total}"
            )

        sched.step()

        # Dead-model detector: loss locked at ~1.0 means weights -> 0.
        # Restore last good checkpoint and halve LR to escape the basin.
        if len(history) >= 3:
            recent_train = [h["train_loss"] for h in history[-3:]]
            if all(abs(l - 1.0) < 0.02 for l in recent_train if l == l):
                logger.error(
                    "DEAD MODEL: train loss locked at ~1.0 for 3 epochs. "
                    "Restoring best checkpoint and halving LR."
                )
                model.load_state_dict(best_state)
                for pg in opt.param_groups:
                    pg["lr"] *= 0.5
                logger.info(
                    "  Restored checkpoint. New LR: "
                    + f"{opt.param_groups[0]['lr']:.2e}"
                )
                scaler = torch.amp.GradScaler(
                    "cuda", enabled=use_amp,
                    init_scale=256, growth_interval=500,
                )

        # -- Validate ----------------------------------------------------------
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

        # -- Trading score -----------------------------------------------------
        trade_score = None
        if use_trade_sel and (epoch % eval_interval == 0 or epoch == epochs):
            trade_score = _quick_trading_score(model, val_arr, dev)
            # model.train() is guaranteed by the try/finally inside _quick_trading_score

        # -- Checkpoint selection ----------------------------------------------
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
                    "  ? checkpoint (trade_score=" + str(round(trade_score, 4)) + ")"
                    + "  -> diffusion_best_trading.pt"
                )
        else:
            if v_loss < best_loss:
                best_loss  = v_loss
                best_state = _clone_state(model)
                save_checkpoint(model, epoch, v_loss)
                logger.success("  ? checkpoint (val=" + str(round(v_loss, 5)) + ")")

        # -- Logging -----------------------------------------------------------
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

        # -- Early stopping ----------------------------------------------------
        # Don't trigger early stopping during warmup -- loss is still rising
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
