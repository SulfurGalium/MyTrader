"""
gpu_utils.py — CUDA 13.2 optimization utilities
"""
import torch
import torch.nn as nn
from loguru import logger


def setup_cuda_for_training(device_str: str = "cuda") -> torch.device:
    """
    Initialize CUDA with optimal settings for CUDA 13.2.
    Returns the device object.
    """
    if device_str == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Using CPU.")
            return torch.device("cpu")
        
        # Get GPU info
        device = torch.device("cuda")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"cuDNN Version: {torch.backends.cudnn.version()}")
        
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Optional: compile models for faster execution (PyTorch 2.0+)
        # This can give 10-50% speedup but requires PyTorch 2.0+
        torch.set_float32_matmul_precision('high')
        
        logger.info("CUDA optimizations enabled: benchmark=True, TF32=True")
        
        return device
    else:
        logger.info("Using CPU (DEVICE not set to 'cuda')")
        return torch.device("cpu")


def compile_model(model: nn.Module, mode: str = "reduce-overhead") -> nn.Module:
    """
    Compile the model with torch.compile for faster execution (PyTorch 2.0+).
    Modes: 'default', 'reduce-overhead', 'max-autotune'
    reduce-overhead is best for training, max-autotune best for inference.
    """
    try:
        compiled = torch.compile(model, mode=mode)
        logger.info(f"Model compiled with mode='{mode}'")
        return compiled
    except Exception as e:
        logger.warning(f"Model compilation failed: {e}. Using uncompiled model.")
        return model


def get_gpu_memory_stats() -> dict:
    """Return GPU memory stats."""
    if not torch.cuda.is_available():
        return {}
    
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def clear_gpu_cache():
    """Clear GPU cache if CUDA available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")
