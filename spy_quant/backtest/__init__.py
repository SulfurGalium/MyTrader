# backtest/__init__.py
from .simulation import evaluate, walk_forward, monte_carlo, detect_regime

__all__ = ["evaluate", "walk_forward", "monte_carlo", "detect_regime"]
