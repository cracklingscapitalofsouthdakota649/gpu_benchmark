# scripts/gpu_utils.py
import torch
import platform
import subprocess

def detect_gpu_backend():
    """Detects available GPU acceleration backend."""
    backends = {}

    # NVIDIA CUDA
    if torch.cuda.is_available():
        backends["CUDA"] = torch.cuda.get_device_name(0)

    # AMD ROCm
    try:
        if torch.version.hip:
            backends["ROCm"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    # Intel OneAPI (through SYCL / Level Zero)
    try:
        import dpctl
        devices = [str(d) for d in dpctl.get_devices()]
        if devices:
            backends["IntelGPU"] = devices
    except Exception:
        pass

    # Apple Metal (MPS)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        backends["MPS"] = platform.system()

    return backends


def get_gpu_info():
    """Returns summary string of GPU accelerators."""
    backends = detect_gpu_backend()
    if not backends:
        return "No GPU accelerator detected"
    return ", ".join([f"{k}: {v}" for k, v in backends.items()])


def benchmark_matrix_multiplication(device="cuda", size=4096):
    """Run a quick matrix multiplication benchmark on selected device."""
    import time
    import torch

    if not torch.cuda.is_available() and device == "cuda":
        return None

    A = torch.rand((size, size), device=device)
    B = torch.rand((size, size), device=device)
    torch.cuda.synchronize()
    start = time.time()
    C = A @ B
    torch.cuda.synchronize()
    end = time.time()
    return round(end - start, 4)
