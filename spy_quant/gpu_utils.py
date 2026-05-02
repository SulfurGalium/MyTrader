"""
gpu_utils.py — CUDA / device setup utilities

Called by trainer.py and inference.py to initialize the compute device.
Provides clear diagnostics when DEVICE=cuda is set but CUDA isn't available,
which is the most common silent failure (CPU PyTorch wheel installed but GPU
wheel needed, or CUDA drivers not installed).
"""
import torch
import torch.nn as nn
from loguru import logger


def setup_cuda_for_training(device_str: str = "cuda") -> torch.device:
    """
    Initialize device with optimal settings. Returns a torch.device.

    If device_str == "cuda" but CUDA is not available, prints a detailed
    diagnostic explaining exactly why and how to fix it, then falls back
    to CPU so training still runs rather than crashing.
    """
    if device_str != "cuda":
        logger.info(f"Device set to '{device_str}' — using as-is.")
        return torch.device(device_str)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        cuda_ver = torch.version.cuda
        cudnn_ver = torch.backends.cudnn.version()
        vram_gb   = torch.cuda.get_device_properties(0).total_memory / 1e9

        logger.info(f"GPU      : {gpu_name}")
        logger.info(f"VRAM     : {vram_gb:.1f} GB")
        logger.info(f"CUDA     : {cuda_ver}")
        logger.info(f"cuDNN    : {cudnn_ver}")

        # Enable standard CUDA optimisations
        torch.backends.cudnn.benchmark        = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.set_float32_matmul_precision("high")

        logger.info("CUDA optimisations enabled: benchmark=True, TF32=True, matmul=high")
        return device

    # ── CUDA requested but not available — give a clear, actionable diagnosis ──
    torch_ver    = torch.__version__
    has_cpu_only = "+cpu" in torch_ver or "cpu" in torch_ver.lower()

    logger.error("=" * 60)
    logger.error("DEVICE=cuda is set but torch.cuda.is_available() is False.")
    logger.error(f"Installed PyTorch : {torch_ver}")

    if has_cpu_only:
        logger.error("")
        logger.error("CAUSE: You have the CPU-only PyTorch wheel installed.")
        logger.error("The CPU wheel never supports CUDA regardless of your GPU.")
        logger.error("")
        logger.error("FIX — uninstall and reinstall with a CUDA wheel:")
        logger.error("  pip uninstall torch torchvision torchaudio -y")
        logger.error("  # CUDA 12.1 (most common for RTX 30/40 series):")
        logger.error("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
        logger.error("  # CUDA 11.8 (older cards / drivers):")
        logger.error("  pip install torch --index-url https://download.pytorch.org/whl/cu118")
        logger.error("")
        logger.error("Check your driver CUDA version with: nvidia-smi")
    else:
        logger.error("")
        logger.error("POSSIBLE CAUSES:")
        logger.error("  1. No NVIDIA GPU in this machine.")
        logger.error("  2. NVIDIA drivers not installed or outdated.")
        logger.error("     Install from: https://www.nvidia.com/drivers")
        logger.error("  3. CUDA toolkit version mismatch with PyTorch.")
        logger.error("     Run 'nvidia-smi' and match the CUDA version shown.")

    logger.error("")
    logger.error("Falling back to CPU. Set DEVICE=cpu in .env to silence this.")
    logger.error("=" * 60)

    return torch.device("cpu")


def compile_model(model: nn.Module, mode: str = "reduce-overhead") -> nn.Module:
    """
    Compile with torch.compile for faster execution (PyTorch 2.0+).
    'reduce-overhead' = best for training, 'max-autotune' = best for inference.
    Returns the original uncompiled model if compilation fails.
    """
    try:
        compiled = torch.compile(model, mode=mode)
        logger.info(f"Model compiled (mode='{mode}')")
        return compiled
    except Exception as exc:
        logger.warning(f"torch.compile failed: {exc} — using uncompiled model.")
        return model


def get_gpu_memory_stats() -> dict:
    """Return current GPU memory stats, or empty dict if no CUDA."""
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_gb":     torch.cuda.memory_allocated()     / 1e9,
        "reserved_gb":      torch.cuda.memory_reserved()      / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def clear_gpu_cache() -> None:
    """Clear GPU memory cache if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU cache cleared.")

