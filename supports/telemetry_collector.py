# supports/telemetry_collector.py
"""
Telemetry collector for per-test performance sampling.
Records CPU, GPU, and Memory utilization periodically into JSON.
"""

import time
import json
import psutil
import threading
import signal
import atexit
import os

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

# --- graceful stop signal shared across tests ---\
STOP_EVENT = threading.Event()

def _graceful_stop(*_):
    STOP_EVENT.set()

signal.signal(signal.SIGINT, _graceful_stop)
atexit.register(_graceful_stop)
# -----------------------------------------------


class TelemetryCollector:
    """Thread-based telemetry sampler."""

    # Note: duration=0 means run until explicitly stopped (used by the Pytest hook)
    def __init__(self, test_name: str, dest_dir: str, duration: int = 0, interval: float = 0.5):
        self.test_name = test_name
        self.dest_dir = dest_dir
        self.duration = duration 
        self.interval = interval
        self.samples = []
        self.thread = None
        self._stop_local = threading.Event()

    def _collect_metrics(self):
        """Collect one sample."""
        gpu_util = vram_util = 0
        if GPU_AVAILABLE:
            try:
                # Assuming index 0 for the primary GPU
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # GPU Utilization (%)
                gpu_util = util.gpu
                
                # VRAM Utilization (%)
                vram_util = (mem.used / mem.total) * 100
            except Exception:
                # Handle cases where NVML might fail or the GPU index is invalid
                pass

        # CPU & RAM utilization for the whole system
        cpu_util = psutil.cpu_percent(interval=None)
        ram_util = psutil.virtual_memory().percent

        # Using absolute time for consistent storage
        return {
            "timestamp": round(time.time(), 2), 
            "cpu_percent": cpu_util,
            "ram_percent": ram_util,
            "gpu_util_percent": gpu_util,
            "vram_percent": vram_util,
        }

    def _run(self):
        start_time = time.time()
        while not STOP_EVENT.is_set() and not self._stop_local.is_set():
            sample = self._collect_metrics()
            self.samples.append(sample)
            time.sleep(self.interval)
            # Check if duration limit is hit only if duration > 0
            if self.duration > 0 and (time.time() - start_time > self.duration):
                break
        print(f"[INFO] Telemetry thread finished for {self.test_name} ({len(self.samples)} samples collected).")

    def start(self):
        os.makedirs(self.dest_dir, exist_ok=True)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        # Local stop first to ensure this test's thread exits its loop
        self._stop_local.set()
        if self.thread and self.thread.is_alive():
            # Wait for the thread to finish its run loop
            self.thread.join(timeout=5)
        # We must save here in case the thread joined and didn't call _save() itself
        self._save()
    
    # New helper to get the full path for the JSON file
    def _get_json_path(self):
        filename = f"telemetry_{self.test_name}.json"
        return os.path.join(self.dest_dir, filename)

    def _save(self):
        """Write collected samples to JSON file and generate visualization."""
        if not self.samples:
            return
        
        # 1. Write JSON data
        dest = self._get_json_path()
        try:
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(self.samples, f, indent=2)
            print(f"[INFO] Saved telemetry data to {dest}")
        except Exception as e:
            print(f"[ERROR] Failed to save telemetry JSON for {self.test_name}: {e}")
            return
        
        # 2. Generate visualization and attach to Allure report
        try:
            # Local import to prevent issues with dependencies (like matplotlib)
            # and potential circular imports if other modules rely on telemetry_collector
            from .telemetry_visualizer import plot_telemetry_json
            
            image_path = plot_telemetry_json(dest, self.test_name, self.dest_dir)

            if image_path:
                print(f"[INFO] Saved telemetry plot to {image_path}. This PNG should be attached to the Allure report.")
            
        except ImportError:
            print("[WARN] Telemetry visualizer utility or dependencies (matplotlib) are missing. Skipping plot generation.")
        except Exception as e:
            print(f"[ERROR] Plot generation failed for {self.test_name}: {e}")
