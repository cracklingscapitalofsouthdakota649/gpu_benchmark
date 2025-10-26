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

# --- graceful stop signal shared across tests ---
STOP_EVENT = threading.Event()

def _graceful_stop(*_):
    STOP_EVENT.set()

signal.signal(signal.SIGINT, _graceful_stop)
atexit.register(_graceful_stop)
# -----------------------------------------------


class TelemetryCollector:
    """Thread-based telemetry sampler."""

    def __init__(self, test_name: str, dest_dir: str, duration: int = 10, interval: float = 0.5):
        self.test_name = test_name
        self.dest_dir = dest_dir
        self.duration = duration
        self.interval = interval
        self.samples = []
        self.thread = None
        self._stop_local = threading.Event()

    def _collect_metrics(self):
        """Collect one sample."""
        gpu_util = mem_util = 0
        if GPU_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_util = util.gpu
                mem_util = (mem.used / mem.total) * 100
            except Exception:
                pass

        cpu_util = psutil.cpu_percent(interval=None)
        ram_util = psutil.virtual_memory().percent
        return {
            "timestamp": round(time.time(), 2),
            "cpu_percent": cpu_util,
            "ram_percent": ram_util,
            "gpu_util_percent": gpu_util,
            "vram_percent": mem_util,
        }

    def _run(self):
        start_time = time.time()
        while not STOP_EVENT.is_set() and not self._stop_local.is_set():
            sample = self._collect_metrics()
            self.samples.append(sample)
            time.sleep(self.interval)
            if self.duration and (time.time() - start_time > self.duration):
                break
        print(f"[INFO] Telemetry thread finished ({len(self.samples)} samples).")

    def start(self):
        os.makedirs(self.dest_dir, exist_ok=True)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop_local.set()
        STOP_EVENT.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        self._save()

    def _save(self):
        """Write collected samples to JSON file."""
        if not self.samples:
            return
        filename = f"telemetry_{self.test_name}.json"
        dest = os.path.join(self.dest_dir, filename)
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(self.samples, f, indent=2)
        print(f"[INFO] Saved telemetry data to {dest}")


# Manual test run
if __name__ == "__main__":
    c = TelemetryCollector("manual_test", "allure-results/manual", duration=5, interval=1)
    c.start()
    c.thread.join()
    c._save()
