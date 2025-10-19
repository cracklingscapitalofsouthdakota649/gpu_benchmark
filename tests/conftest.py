# tests/conftest.py
import warnings
import pytest

class GpuInfo:
    def __init__(self):
        self.vendor = "cpu"
        self.name = "CPU"
        self.available = False

        # 1️⃣ CUDA detection
        try:
            import torch
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                self.available = True
                self.vendor = "nvidia"
                self.name = torch.cuda.get_device_name(0)
                return
        except Exception:
            pass

        # 2️⃣ Intel GPU via OpenCL
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            for plat in platforms:
                for dev in plat.get_devices():
                    if cl.device_type.to_string(dev.type) == "GPU":
                        self.available = True
                        self.vendor = "intel"
                        self.name = dev.name
                        return
        except Exception:
            pass

        # 3️⃣ DirectML (Windows)
        try:
            import torch_directml as dml
            _ = dml.device()
            self.available = True
            self.vendor = "dml"
            self.name = "DirectML Device"
            return
        except Exception:
            pass
