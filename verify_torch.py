# verify_torch.py
import sys
import importlib.util
import os

# CRITICAL: prevent AMD SMI from crashing
os.environ["TORCH_HIP_PROHIBIT_AMDSMI_INIT"] = "1"

try:
    import torch
    torch_version = torch.__version__
    cuda_available = getattr(torch.cuda, "is_available", lambda: False)()
    has_xpu = hasattr(torch, "xpu") and getattr(torch.xpu, "is_available", lambda: False)()
except Exception as e:
    print("[WARN] Torch import failed:", e)
    torch_version = "N/A"
    cuda_available = False
    has_xpu = False

try:
    directml_available = importlib.util.find_spec("torch_directml") is not None
except Exception:
    directml_available = False

print("torch_version:", torch_version)
print("cuda_available:", cuda_available)
print("xpu_available:", has_xpu)
print("directml_available:", directml_available)
