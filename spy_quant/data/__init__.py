# data/__init__.py
from .loader import load_ohlcv_parquet, load_ohlcv_alpaca, build_raw_dataset
from .features import compute_features
from .preprocessing import preprocess

def __getattr__(name):
    if name in ("SPYWindowDataset", "make_loader"):
        from .dataset import SPYWindowDataset, make_loader
        return {"SPYWindowDataset": SPYWindowDataset, "make_loader": make_loader}[name]
    raise AttributeError(f"module 'data' has no attribute {name!r}")

__all__ = [
    "load_ohlcv_parquet", "load_ohlcv_alpaca", "build_raw_dataset",
    "compute_features", "preprocess",
    "SPYWindowDataset", "make_loader",
]
