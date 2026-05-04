"""
data/dataset.py
Memory-efficient lazy dataset - windows are sliced on-demand, never stored
all at once.  Yields (fine_window, coarse_window, target) tuples.

  fine_window  : (SEQ_LEN, F)         - 5-min bars
  coarse_window: (SEQ_LEN//FACTOR, F) - aggregated coarser context
  target       : scalar float         - forward log return (horizon bars ahead)

The `horizon` parameter controls the prediction target:
  horizon=1  (original) - predict next single 5-min bar return
  horizon=6  (default)  - predict 30-min forward return (sum of 6 bars)
                          substantially better SNR at 5-min granularity;
                          matches a realistic hold-and-exit cadence
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Resolve base class - lazy so torch isn't required just for imports
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    _TORCH_AVAILABLE = True
except ImportError:
    Dataset = object  # type: ignore
    _TORCH_AVAILABLE = False

# Default horizon: 6 bars = 30 minutes at 5-min frequency.
# Override via env var or pass explicitly.
_DEFAULT_HORIZON = int(getattr(config, "TARGET_HORIZON", 6))


class SPYWindowDataset(Dataset):
    """
    Lazily slices rolling windows from a pre-scaled numpy array.

    Parameters
    ----------
    data          : np.ndarray  shape (T, F)
    seq_len       : look-back window in bars
    coarse_factor : downsample factor for coarse branch
    target_col    : column index of log_return (default 0)
    horizon       : how many bars ahead to predict.
                    target = sum(log_returns[i : i+horizon])
                    horizon=1 - original behaviour (next bar only)
                    horizon=6 - 30-min forward return (recommended)
    """

    def __init__(
        self,
        data:          np.ndarray,
        seq_len:       int = config.SEQ_LEN,
        coarse_factor: int = config.COARSE_FACTOR,
        target_col:    int = 0,
        horizon:       int = _DEFAULT_HORIZON,
    ) -> None:
        self.data          = data.astype(np.float32) if data.dtype != np.float32 else data
        self.seq_len       = seq_len
        self.coarse_factor = coarse_factor
        self.target_col    = target_col
        self.horizon       = max(1, horizon)
        # Need seq_len bars of context + horizon bars of future
        self._start        = seq_len
        self._end          = len(data) - self.horizon   # leave room for horizon

    def __len__(self) -> int:
        return max(0, self._end - self._start)

    def __getitem__(self, idx: int):
        import torch
        i      = self._start + idx
        window = self.data[i - self.seq_len : i]       # (L, F)  already float32

        # Sum log returns over the horizon window for a less-noisy target.
        # For horizon=1 this is identical to the original single-bar target.
        future = self.data[i : i + self.horizon, self.target_col]
        target = float(future.sum())

        fine = torch.from_numpy(window)                # zero-copy view

        # Coarse: non-overlapping mean-pool along time axis
        L, F   = window.shape
        trim   = (L // self.coarse_factor) * self.coarse_factor
        coarse = torch.from_numpy(
            window[:trim]
            .reshape(-1, self.coarse_factor, F)
            .mean(axis=1)
        )                                              # (L/k, F)

        return fine, coarse, torch.tensor(target, dtype=torch.float32)


def make_loader(
    data:        np.ndarray,
    batch_size:  int  = config.BATCH_SIZE,
    shuffle:     bool = True,
    num_workers: int  = 2,       # 2 workers: parallel prefetch, safe on Windows
    pin_memory:  bool = True,    # pin host memory -> faster GPU transfer
    horizon:     int  = _DEFAULT_HORIZON,
    cuda:        bool = True,    # set False on CPU-only machines
    **kwargs,
):
    """
    Build a DataLoader with performance settings from reference train.py.

    Key improvements over the old defaults:
      num_workers=2   : background workers preload batches in parallel.
                        With 5320 batches/epoch at batch_size=16, data loading
                        was the bottleneck -- workers hide that latency.
      pin_memory=True : copies tensors to page-locked memory before transfer,
                        allowing the GPU DMA engine to overlap data transfer
                        with compute. ~10-15% throughput improvement.
      persistent_workers=True : workers stay alive between epochs, avoiding
                        the spawn/teardown overhead (significant on Windows).
      prefetch_factor=4 : each worker preloads 4 batches ahead of consumption.

    num_workers must be 0 on some Windows configurations. If you see
    "BrokenPipeError" or hangs at startup, set num_workers=0.
    pin_memory and persistent_workers are automatically disabled when
    num_workers=0 to avoid conflicts.
    """
    from torch.utils.data import DataLoader
    import torch

    # Detect CUDA availability to set pin_memory correctly
    use_cuda = cuda and torch.cuda.is_available()

    # persistent_workers and prefetch_factor require num_workers > 0
    persistent = num_workers > 0
    prefetch   = 4 if num_workers > 0 else None
    pin        = pin_memory and use_cuda and num_workers > 0

    ds = SPYWindowDataset(data, horizon=horizon)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
        **kwargs,
    )
