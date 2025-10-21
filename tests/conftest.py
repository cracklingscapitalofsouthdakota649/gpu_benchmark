# tests/conftest.py
import warnings
import pytest
import allure
import os
import platform
import threading
import time
import io
import psutil
import matplotlib.pyplot as plt

# Attempt to import pynvml for real-time GPU stats
try:
    import pynvml
except ImportError:
    pynvml = None

# Get the path to the root of the project (one directory up from 'tests')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Prepend the project root to sys.path
# This ensures Python can find the 'supports' package from the 'tests' directory.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f"✅ Conftest: Added project root to sys.path: {PROJECT_ROOT}")

# ───────────────────────────────────────────────────────────────
# 1. GPU Hardware Detection
# ───────────────────────────────────────────────────────────────

class GpuInfo:
    """Detects available GPU hardware and sets vendor/device info."""
    def __init__(self):
        self.vendor = "cpu"
        self.name = "CPU"
        self.device_str = "cpu"
        self.available = False

        try:
            import torch
            
            # 1️⃣ CUDA (NVIDIA)
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                self.available = True
                self.vendor = "nvidia"
                self.name = torch.cuda.get_device_name(0)
                self.device_str = "cuda:0"
                return

            # 2️⃣ ROCm (AMD)
            if hasattr(torch, "version") and hasattr(torch.version, "hip") and torch.version.hip:
                self.available = True
                self.vendor = "amd"
                self.name = torch.cuda.get_device_name(0) # Uses cuda interface
                self.device_str = "cuda:0" # Uses cuda interface
                return

            # 3️⃣ DirectML (Windows - Intel/AMD/NVIDIA)
            if hasattr(torch, "is_directml_available") and torch.is_directml_available():
                self.available = True
                self.vendor = "dml"
                self.name = "DirectML Device"
                self.device_str = "dml"
                return
        except Exception:
            pass # Fallback to CPU

        # 4️⃣ Intel GPU via OpenCL (Less common for PyTorch)
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            for plat in platforms:
                for dev in plat.get_devices():
                    if "Intel" in dev.vendor and cl.device_type.to_string(dev.type) == "GPU":
                        self.available = True
                        self.vendor = "intel_opencl"
                        self.name = dev.name
                        self.device_str = "cpu" # PyTorch doesn't natively support OpenCL device
                        return
        except Exception:
            pass # Fallback to CPU

# Create a single, session-wide instance for detection and markers
_global_gpu_info = GpuInfo()


# ───────────────────────────────────────────────────────────────
# 2. Pytest Markers for Conditional Skipping
# ───────────────────────────────────────────────────────────────

# Define skip markers based on the hardware detected *before* tests run
requires_gpu = pytest.mark.skipif(
    not _global_gpu_info.available, 
    reason="Test requires a GPU"
)
requires_cuda = pytest.mark.skipif(
    _global_gpu_info.vendor != "nvidia", 
    reason="Test requires an NVIDIA GPU (CUDA)"
)
requires_rocm = pytest.mark.skipif(
    _global_gpu_info.vendor != "amd", 
    reason="Test requires an AMD GPU (ROCm)"
)
requires_dml = pytest.mark.skipif(
    _global_gpu_info.vendor != "dml", 
    reason="Test requires a DirectML-capable device"
)

def pytest_configure(config):
    """Register custom markers to avoid Pytest warnings."""
    config.addinivalue_line("markers", "requires_gpu: Skip test if no GPU is available")
    config.addinivalue_line("markers", "requires_cuda: Skip test if no NVIDIA GPU is available")
    config.addinivalue_line("markers", "requires_rocm: Skip test if no AMD GPU (ROCm) is available")
    config.addinivalue_line("markers", "requires_dml: Skip test if no DirectML device is available")


# ───────────────────────────────────────────────────────────────
# 3. Core Pytest Fixtures (GPU Info & Torch Device)
# ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def gpu_info():
    """Provides the session-wide GpuInfo object to tests."""
    yield _global_gpu_info

@pytest.fixture(scope="session")
def torch_device(gpu_info):
    """Provides the correct torch.device object (e.g., 'cuda:0', 'dml', 'cpu') for tests."""
    import torch
    return torch.device(gpu_info.device_str)


# ───────────────────────────────────────────────────────────────
# 4. Allure Report Integration Fixtures
# ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def allure_environment(request, gpu_info):
    """
    This autouse, session-scoped fixture dynamically creates the 
    environment.properties file for the Allure report.
    """
    alluredir = request.config.getoption("--alluredir")
    
    if not alluredir:
        return # Don't do anything if Allure is not active

    # Ensure the allure results directory exists
    allure_results_dir = os.path.abspath(alluredir)
    os.makedirs(allure_results_dir, exist_ok=True)
    
    env_props_path = os.path.join(allure_results_dir, "environment.properties")
    
    with open(env_props_path, "w") as f:
        f.write(f"OS={platform.system()} ({platform.release()})\n")
        f.write(f"Python.Version={platform.python_version()}\n")
        f.write(f"CPU={platform.processor()}\n")
        f.write("\n# --- GPU Info ---\n")
        f.write(f"GPU.Available={gpu_info.available}\n")
        f.write(f"GPU.Vendor={gpu_info.vendor.upper()}\n")
        f.write(f"GPU.Name={gpu_info.name}\n")
        f.write(f"Torch.Device={gpu_info.device_str}\n")
    
    yield # Let the test session run


# ───────────────────────────────────────────────────────────────
# 5. Performance Monitoring Fixture (Updated for GPU metrics)
# ───────────────────────────────────────────────────────────────

class SystemMonitor:
    """Monitors CPU, System Memory, GPU Utilization, and GPU Memory in a background thread."""
    def __init__(self, interval=0.2):
        self.data = {
            "time": [], 
            "cpu": [], 
            "memory": [], 
            "gpu_util": [], 
            "gpu_mem_used": []
        }
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self.run, daemon=True)
        self.interval = interval
        self.start_time = time.time()
        self.gpu_monitoring_enabled = False
        self._gpu_handle = None
        self._gpu_device_name = _global_gpu_info.name
        
        # Initialize NVIDIA GPU Monitoring (pynvml)
        if _global_gpu_info.vendor == "nvidia" and pynvml:
            try:
                pynvml.nvmlInit()
                # Assuming single-GPU tests, using index 0
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_monitoring_enabled = True
            except pynvml.NVMLError_DriverNotLoaded:
                warnings.warn("NVIDIA driver not loaded. GPU monitoring disabled.")
            except Exception as e:
                warnings.warn(f"Failed to initialize pynvml for GPU monitoring: {e}")
        elif _global_gpu_info.available and _global_gpu_info.vendor != "nvidia":
            warnings.warn(f"GPU monitoring for {_global_gpu_info.vendor.upper()} not implemented yet (requires specific tooling). Tracking disabled.")

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        if self.gpu_monitoring_enabled and pynvml:
            pynvml.nvmlShutdown()

    def _get_gpu_stats(self):
        if self.gpu_monitoring_enabled and self._gpu_handle:
            try:
                # Utilization %
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                gpu_util = util.gpu
                
                # Memory Used (GB)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                gpu_mem_used_gb = mem.used / (1024**3)
                
                return gpu_util, gpu_mem_used_gb
            except Exception:
                return 0, 0
        
        return None, None

    def run(self):
        """The main loop for the monitoring thread."""
        while not self._stop_event.is_set():
            try:
                timestamp = time.time() - self.start_time

                # 1. System Stats
                self.data["time"].append(timestamp)
                self.data["cpu"].append(psutil.cpu_percent())
                self.data["memory"].append(psutil.virtual_memory().percent)
                
                # 2. GPU Stats
                gpu_util, gpu_mem_used = self._get_gpu_stats()
                if gpu_util is not None:
                    self.data["gpu_util"].append(gpu_util)
                    self.data["gpu_mem_used"].append(gpu_mem_used)
                
                time.sleep(self.interval)
            except psutil.NoSuchProcess:
                break
            except Exception as e:
                warnings.warn(f"Monitor thread encountered an error: {e}")
                break

    def get_plot_as_png(self):
        """Generates a Matplotlib plot and returns it as a PNG in memory."""
        if not self.data["time"]:
            return None
            
        try:
            # Use 2x1 subplots for clear separation of system and GPU metrics
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            ax1 = axes[0] # Top subplot: CPU and System Memory
            ax3 = axes[1] # Bottom subplot: GPU Metrics

            # ----------------- Subplot 1: CPU and System Memory -----------------
            color_cpu = 'tab:red'
            ax1.set_ylabel('CPU %', color=color_cpu)
            ax1.plot(self.data['time'], self.data['cpu'], color=color_cpu, label='CPU')
            ax1.tick_params(axis='y', labelcolor=color_cpu)
            ax1.set_ylim([0, 105])

            ax2 = ax1.twinx()
            color_mem = 'tab:blue'
            ax2.set_ylabel('System Memory %', color=color_mem)
            ax2.plot(self.data['time'], self.data['memory'], color=color_mem, label='System Memory')
            ax2.tick_params(axis='y', labelcolor=color_mem)
            ax2.set_ylim([0, 105])
            
            ax1.set_title('CPU and System Memory Usage')
            ax1.grid(True)
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            
            # ----------------- Subplot 2: GPU Utilization and Memory -----------------
            ax3.set_xlabel('Time (seconds)')
            if self.data["gpu_util"]:
                color_gpu_util = 'tab:green'
                ax3.set_ylabel('GPU Util %', color=color_gpu_util)
                ax3.plot(self.data['time'], self.data['gpu_util'], color=color_gpu_util, label='GPU Utilization')
                ax3.tick_params(axis='y', labelcolor=color_gpu_util)
                ax3.set_ylim([-5, 105]) # Allow for zero line
                
                ax4 = ax3.twinx()
                color_gpu_mem = 'tab:orange'
                ax4.set_ylabel('GPU Memory Used (GB)', color=color_gpu_mem)
                ax4.plot(self.data['time'], self.data['gpu_mem_used'], color=color_gpu_mem, label='GPU Memory Used')
                ax4.tick_params(axis='y', labelcolor=color_gpu_mem)
                
                # Dynamic Y-limit for GPU memory based on max collected value
                max_mem = max(self.data['gpu_mem_used']) if self.data['gpu_mem_used'] else 1
                ax4.set_ylim([0, max_mem * 1.1]) 
                
                ax3.set_title(f'{self._gpu_device_name} Performance')
                ax3.grid(True)
                ax3.legend(loc='upper left')
                ax4.legend(loc='upper right')
            else:
                ax3.text(0.5, 0.5, f'GPU Monitoring Disabled. Check pynvml installation for {_global_gpu_info.name}.', 
                         horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
                ax3.set_title('GPU Metrics Unavailable')
                ax3.grid(True)
                
            fig.suptitle('System Performance During Test', fontsize=16)
            fig.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust for suptitle

            # Save the plot to an in-memory buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            return buf.getvalue()
        except Exception as e:
            warnings.warn(f"Failed to generate performance plot: {e}")
            return None


@pytest.fixture
def system_monitor(request):
    """
    A pytest fixture that monitors system and GPU performance during a test
    and attaches a plot to the Allure report.
    """
    monitor = SystemMonitor()
    monitor.start()

    # Run the actual test
    yield

    # Test is finished, stop monitoring
    monitor.stop()

    # Generate the plot
    plot_png = monitor.get_plot_as_png()

    # Attach the plot to the Allure report
    if plot_png:
        try:
            allure.attach(
                plot_png, 
                name=f"Performance Graph ({request.node.name})", 
                attachment_type=allure.attachment_type.PNG
            )
        except Exception as e:
            warnings.warn(f"Failed to attach Allure plot: {e}")