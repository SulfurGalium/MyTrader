# models/__init__.py
def __getattr__(name):
    if name in ("MultiTimeframeDiffusion", "save_checkpoint", "load_checkpoint"):
        from .diffusion import MultiTimeframeDiffusion, save_checkpoint, load_checkpoint
        return {"MultiTimeframeDiffusion": MultiTimeframeDiffusion,
                "save_checkpoint": save_checkpoint,
                "load_checkpoint": load_checkpoint}[name]
    if name == "train":
        from .trainer import train
        return train
    raise AttributeError(f"module 'models' has no attribute {name!r}")

__all__ = ["MultiTimeframeDiffusion", "save_checkpoint", "load_checkpoint", "train"]
