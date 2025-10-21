# device_utils.py
"""
Helper functions to pick GPU/CPU device for testing.
"""
import torch
import os
import sys

# Dynamically find project root (directory containing 'supports')
current_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = current_dir

# Walk up until we find 'supports/gpu_check.py'
for _ in range(3):
    candidate = os.path.join(root_dir, "supports", "gpu_check.py")
    if os.path.exists(candidate):
        break
    root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from supports.gpu_check import get_gpu_info

def get_device(dev_type=None):
    """
    Return a PyTorch device based on available hardware or requested type.
    """
    info = get_gpu_info()
    summary = info.get("summary", {})

    if dev_type is None:
        if summary.get("cuda"):
            dev_type = "cuda"
        elif summary.get("rocm"):
            dev_type = "rocm"
        elif summary.get("directml"):
            dev_type = "directml"
        elif summary.get("metal"):
            dev_type = "mps"
        elif summary.get("opencl_gpu_devices", 0) > 0:
            dev_type = "opencl"
        else:
            dev_type = "cpu"

    # Map dev_type to torch device
    try:
        if dev_type == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif dev_type == "rocm" and getattr(torch.version, "hip", None):
            return torch.device("cuda")
        elif dev_type == "directml":
            import torch_directml
            return torch.device("dml")
        elif dev_type == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass

    # Fallback to CPU
    return torch.device("cpu")


def pick_device():
    """
    Returns string representing the preferred available device.
    """
    info = get_gpu_info()
    summary = info.get("summary", {})

    if summary.get("cuda"):
        return "cuda"
    if summary.get("rocm"):
        return "rocm"
    if summary.get("directml"):
        return "directml"
    if summary.get("metal"):
        return "mps"
    if summary.get("opencl_gpu_devices", 0) > 0:
        return "opencl"
    return "cpu"


def synchronize(dev_type):
    """
    Ensures GPU operations completion.
    """
    import torch

    if dev_type in ("cuda", "rocm"):
        torch.cuda.synchronize()
    elif dev_type == "directml":
        pass
    elif dev_type == "mps":
        torch.backends.mps.synchronize()
    # CPU/OpenCL: no-op
