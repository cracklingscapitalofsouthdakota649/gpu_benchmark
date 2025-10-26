# supports/gpu_monitor.py
import time
import json
import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


def collect_gpu_metrics(duration=10, interval=0.5):
    """Collect GPU, VRAM, and CPU utilization samples."""
    samples = []
    start_time = time.time()

    while time.time() - start_time < duration:
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
        samples.append({
            "timestamp": round(time.time() - start_time, 2),
            "gpu_util": gpu_util,
            "mem_util": mem_util,
            "cpu_util": cpu_util
        })
        time.sleep(interval)
    return samples


if __name__ == "__main__":
    metrics = collect_gpu_metrics(duration=5, interval=1)
    print(json.dumps(metrics, indent=2))
