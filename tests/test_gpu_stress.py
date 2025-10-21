# tests/test_gpu_stress.py
import os
import json
import time
import allure
import pytest
import torch

from scripts.plot_gpu_metrics import attach_chart_to_allure
from scripts.system_metrics import SystemMetrics
from tests.device_utils import pick_device, get_device, synchronize

RESULTS_DIR = "allure-results"

@allure.feature("GPU Stability and Stress Testing") # ⬅️ ADDED
@allure.story("Sustained Compute Stress Test")      # ⬅️ ADDED
@pytest.mark.gpu
@pytest.mark.cpu
@pytest.mark.stress
def test_gpu_stress():
    dev_type = pick_device()
    device = get_device(dev_type)
    metrics_monitor = SystemMetrics(device)

    results = []
    start_time = time.time()
    num_steps = 30

    # Adjust workload sizes for Intel/OpenCL fallback
    size = 1024 if dev_type == "opencl" else 4096

    x = torch.randn(size, size, device=device, dtype=torch.float32)
    for i in range(num_steps):
        _ = torch.matmul(x, x)
        synchronize(dev_type)
        metrics = metrics_monitor.get_all_metrics()
        metrics["step"] = i
        results.append(metrics)

    total_time = time.time() - start_time
    fps = round(num_steps / total_time, 2) if total_time > 0 else 0.0

    # Attach metrics
    allure.attach(json.dumps(results, indent=2), "Stress Metrics", allure.attachment_type.JSON)
    attach_chart_to_allure(results)

    # Summary
    avg_gpu = sum(m.get("gpu_util", 0) for m in results) / len(results)
    avg_cpu = sum(m.get("cpu_util", 0) for m in results) / len(results)
    avg_mem = sum(m.get("gpu_mem_util", 0) for m in results) / len(results)

    summary = {
        "test_name": "gpu_stress",
        "gpu_name": metrics_monitor.get_gpu_name(),
        "avg_util": round(avg_gpu, 2),
        "avg_mem": round(avg_mem, 2),
        "avg_cpu_util": round(avg_cpu, 2),
        "fps": fps,
        "backend": dev_type,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_file = os.path.join(RESULTS_DIR, "summary_gpu_stress.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    allure.attach(json.dumps(summary, indent=2), "Stress Test Summary", allure.attachment_type.JSON)

    # Assertions only for sanity
    assert avg_gpu < 101
    assert avg_cpu < 101