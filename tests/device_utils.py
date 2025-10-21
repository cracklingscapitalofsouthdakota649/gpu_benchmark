# device_utils.py
"""
Helper functions to pick GPU/CPU device for testing.
"""
import os
import sys
import torch

# Dynamically ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Attempt to import get_gpu_info safely, with cross-platform fallback
try:
    from supports.gpu_check import get_gpu_info
except ModuleNotFoundError as e:
    print(f"⚠️ Import warning: {e}. Attempting fallback import...")
    # Add 'supports' explicitly if it isn't being resolved as a package
    SUPPORTS_DIR = os.path.join(PROJECT_ROOT, "supports")
    if SUPPORTS_DIR not in sys.path:
        sys.path.insert(0, SUPPORTS_DIR)
    try:
        from gpu_check import get_gpu_info
    except ModuleNotFoundError as e2:
        raise ImportError(
            f"❌ Failed to import gpu_check.py even after adjusting sys.path.\n"
            f"PROJECT_ROOT={PROJECT_ROOT}\nSUPPORTS_DIR={SUPPORTS_DIR}\n"
            f"sys.path={sys.path}\nError: {e2}"
        )

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
