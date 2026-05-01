# trading/__init__.py
# torch-dependent submodules are imported on demand to allow
# non-inference code paths (report, live order logic) to work
# in environments without torch installed.

def __getattr__(name):
    if name in ("generate_signal",):
        from .inference import generate_signal
        return generate_signal
    if name in ("TradingSession", "AlpacaClient"):
        from .live import TradingSession, AlpacaClient
        return {"TradingSession": TradingSession, "AlpacaClient": AlpacaClient}[name]
    raise AttributeError(f"module 'trading' has no attribute {name!r}")

__all__ = ["TradingSession", "AlpacaClient", "generate_signal"]
