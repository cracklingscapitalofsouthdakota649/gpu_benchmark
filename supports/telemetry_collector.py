# supports/telemetry_collector.py
import time
import threading
import json
import os
from datetime import datetime

import psutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional dependencies
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    import pyopencl as cl
    HAS_PYOPENCL = True
except Exception:
    HAS_PYOPENCL = False


class TelemetryCollector:
    """
    Per-module telemetry collector.
    Collects CPU, RAM, and GPU metrics (supports NVIDIA, AMD, Intel via OpenCL).
    """

    def __init__(self, module_name: str, sample_interval: float = 1.0):
        self.module_name = module_name.replace("/", "_").replace("\\", "_")
        self.sample_interval = sample_interval
        self._data = []
        self._running = threading.Event()
        self._thread = None

        self._use_nvml = HAS_NVML
        self._use_torch = HAS_TORCH and torch.cuda.is_available()
        self._use_opencl = HAS_PYOPENCL

        self._nvml_handle = None
        if self._use_nvml:
            try:
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._use_nvml = False

        self._opencl_dev = None
        if self._use_opencl:
            try:
                plats = cl.get_platforms()
                for p in plats:
                    for d in p.get_devices():
                        if d.type & cl.device_type.GPU:
                            self._opencl_dev = d
                            break
                    if self._opencl_dev:
                        break
            except Exception:
                self._use_opencl = False

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"telemetry-{self.module_name}")
        self._thread.start()

    def stop(self, timeout: float = 2.0):
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=timeout)
        return list(self._data)

    def _sample_once(self):
        ts = time.time()
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent

        gpu_vendor = "None"
        gpu_util = None
        gpu_mem_used_gb = None
        gpu_mem_total_gb = None

        # NVIDIA via NVML
        if self._use_nvml and self._nvml_handle:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                gpu_vendor = "NVIDIA"
                gpu_util = util.gpu
                gpu_mem_used_gb = round(mem.used / (1024**3), 3)
                gpu_mem_total_gb = round(mem.total / (1024**3), 3)
            except Exception:
                self._use_nvml = False

        # PyTorch fallback (NVIDIA only)
        elif self._use_torch:
            try:
                dev = torch.device("cuda")
                mem_used = torch.cuda.memory_allocated(dev)
                mem_total = torch.cuda.get_device_properties(0).total_memory
                gpu_vendor = "NVIDIA (Torch)"
                gpu_mem_used_gb = round(mem_used / (1024**3), 3)
                gpu_mem_total_gb = round(mem_total / (1024**3), 3)
            except Exception:
                self._use_torch = False

        # OpenCL fallback — covers Intel and AMD GPUs
        elif self._use_opencl and self._opencl_dev:
            try:
                gpu_vendor = self._opencl_dev.vendor.strip()
                total_bytes = int(self._opencl_dev.get_info(cl.device_info.GLOBAL_MEM_SIZE))
                gpu_mem_total_gb = round(total_bytes / (1024**3), 3)
                gpu_mem_used_gb = None  # OpenCL can’t report dynamic usage
            except Exception:
                self._use_opencl = False

        return {
            "time": ts,
            "time_iso": datetime.utcfromtimestamp(ts).isoformat() + "Z",
            "cpu_percent": float(cpu),
            "ram_percent": float(ram),
            "gpu_vendor": gpu_vendor,
            "gpu_util_percent": gpu_util,
            "gpu_mem_used_gb": gpu_mem_used_gb,
            "gpu_mem_total_gb": gpu_mem_total_gb,
        }

    def _run(self):
        while self._running.is_set():
            try:
                self._data.append(self._sample_once())
            except Exception:
                pass
            time.sleep(self.sample_interval)

    def save_json(self, dest_dir: str):
        os.makedirs(dest_dir, exist_ok=True)
        path = os.path.join(dest_dir, f"telemetry_{self.module_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
        return path

    def plot_png(self, dest_dir: str):
        """Save telemetry chart as PNG."""
        if not self._data:
            return None

        os.makedirs(dest_dir, exist_ok=True)
        times = np.array([d["time"] for d in self._data])
        t0 = times[0]
        rel = times - t0

        cpu = [d["cpu_percent"] for d in self._data]
        ram = [d["ram_percent"] for d in self._data]
        gpu_util = [
            d["gpu_util_percent"] if d["gpu_util_percent"] is not None else 0
            for d in self._data
        ]

        plt.figure(figsize=(10, 5))
        plt.plot(rel, cpu, label="CPU (%)", linewidth=1.5)
        plt.plot(rel, ram, label="RAM (%)", linewidth=1.5)
        plt.plot(rel, gpu_util, label="GPU Util (%)", linewidth=1.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Utilization (%)")
        plt.title(f"System Telemetry - {self.module_name}")
        plt.legend()
        plt.tight_layout()

        path = os.path.join(dest_dir, f"telemetry_{self.module_name}.png")
        plt.savefig(path)
        plt.close()
        return path
