# scripts/system_metrics.py
import psutil
import torch
import warnings

# --- GPU Library Imports ---
# Try to import, but DO NOT initialize here
try:
    import pynvml
    NVIDIA_SUPPORT = True
except ImportError:
    NVIDIA_SUPPORT = False

try:
    import amdsmi_py
    AMD_SUPPORT = True
except ImportError:
    AMD_SUPPORT = False
# ---------------------------

class SystemMetrics:
    """A centralized class for collecting CPU and GPU system metrics."""

    def __init__(self, device: torch.device):
        self.device = device
        self.gpu_name = "Unknown GPU"
        self.gpu_handle = None
        self.amd_device_handle = None
        self.cpu_count = psutil.cpu_count()
        self.is_nvidia = False
        self.is_amd = False

        if self.device.type == "cuda":
            if NVIDIA_SUPPORT:
                try:
                    # --- INITIALIZE HERE ---
                    pynvml.nvmlInit() 
                    self.gpu_index = self.device.index if self.device.index is not None else 0
                    self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
                    self.gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle).decode('utf-8')
                    self.is_nvidia = True
                except pynvml.NVMLError as e:
                    warnings.warn(f"NVIDIA NVML found but failed to initialize: {e}. GPU metrics will be 0.")
                    NVIDIA_SUPPORT = False  # Disable on failure
            
            if AMD_SUPPORT and not self.is_nvidia: # Don't try AMD if NVIDIA worked
                try:
                    # --- INITIALIZE HERE ---
                    amdsmi_py.AmdSmi.init()
                    self.gpu_index = self.device.index if self.device.index is not None else 0
                    all_devices = amdsmi_py.AmdSmi.get_devices()
                    if all_devices:
                        self.amd_device_handle = all_devices[self.gpu_index]
                        self.gpu_name = amdsmi_py.AmdSmi.get_device_name(self.amd_device_handle)
                        self.is_amd = True
                except Exception as e:
                    warnings.warn(f"AMD SMI found but failed to initialize: {e}. GPU metrics will be 0.")
                    AMD_SUPPORT = False  # Disable on failure
            
            if not self.is_nvidia and not self.is_amd:
                self.gpu_name = "CUDA device (Metrics Lib not found/failed)"

        elif self.device.type == "cpu":
            self.gpu_name = "N/A (CPU only)"
        
        # Get one reading from psutil to set interval=None baseline
        psutil.cpu_percent(interval=None)


    def get_cpu_util(self) -> float:
        """Returns overall CPU utilization percentage."""
        return psutil.cpu_percent(interval=None)

    def get_gpu_util(self) -> float:
        """Returns GPU utilization percentage for the selected device."""
        if self.is_nvidia and self.gpu_handle:
            try:
                return pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu
            except pynvml.NVMLError:
                return 0.0 # Handle potential error during poll
        if self.is_amd and self.amd_device_handle:
            try:
                return amdsmi_py.AmdSmi.get_gpu_activity(self.amd_device_handle)['busy_percent']
            except Exception:
                return 0.0 # Handle potential error during poll
        return 0.0

    def get_gpu_memory_util(self) -> float:
        """Returns GPU memory utilization percentage for the selected device."""
        if self.is_nvidia and self.gpu_handle:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                return (mem_info.used / mem_info.total) * 100
            except pynvml.NVMLError:
                return 0.0
        if self.is_amd and self.amd_device_handle:
            try:
                mem_total = amdsmi_py.AmdSmi.get_memory_total(self.amd_device_handle)
                mem_used = amdsmi_py.AmdSmi.get_memory_used(self.amd_device_handle)
                if mem_total > 0:
                    return (mem_used / mem_total) * 100
            except Exception:
                return 0.0
        return 0.0

    def get_all_metrics(self) -> dict:
        """Helper to get all metrics in a single dictionary."""
        return {
            "cpu_util": self.get_cpu_util(),
            "gpu_util": self.get_gpu_util(),
            "gpu_mem_util": self.get_gpu_memory_util()
        }

    def get_gpu_name(self) -> str:
        return self.gpu_name

    def __del__(self):
        """Cleanup GPU libraries on object deletion."""
        if self.is_nvidia:
            try:
                pynvml.nvmlShutdown()
            except:
                pass # Ignore shutdown errors
        if self.is_amd:
            try:
                amdsmi_py.AmdSmi.shut_down()
            except:
                pass # Ignore shutdown errors