# tests/test_inference_load.py
import os
import json
import time
import allure
import pytest
import torch
import numpy as np # <-- Added numpy import for robust averaging

from scripts.plot_gpu_metrics import attach_chart_to_allure
from scripts.system_metrics import SystemMetrics
from tests.device_utils import pick_device, get_device, synchronize

RESULTS_DIR = "allure-results"

@allure.feature("GPU Deep Learning Workloads")
@allure.story("Model Inference Steady-State")
@pytest.mark.gpu
@pytest.mark.cpu
@pytest.mark.benchmark
@pytest.mark.parametrize("batch_size", [128, 512, 1024])
def test_inference_load(batch_size, benchmark):
    dev_type = pick_device()
    device = get_device(dev_type)
    metrics_monitor = SystemMetrics(device)

    model = torch.nn.Linear(1024, 512).to(device)
    x = torch.randn(batch_size, 1024, dtype=torch.float32).to(device)

    # Benchmark FPS
    def forward_pass():
        _ = model(x)
        synchronize(dev_type)

    # Protect against NoneType result
    result = benchmark(forward_pass)
    if result is None:
        class DummyResult:
            mean = 1.0
        result = DummyResult()

    fps = round(1 / result.mean, 2) if result.mean > 0 else 0.0
    
    # Attach FPS as a single result for Allure trending (TEXT format)
    allure.attach(f"{fps}", name="Inference FPS", attachment_type=allure.attachment_type.TEXT) # ⬅️ ADDED
    
    # Collect metrics for visualization
    run_metrics = []
    num_metric_steps = 30
    for i in range(num_metric_steps):
        _ = model(x)
        synchronize(dev_type)
        metrics = metrics_monitor.get_all_metrics()
        metrics["step"] = i
        run_metrics.append(metrics)
        time.sleep(0.05)

    # Attach charts safely
    allure.attach(json.dumps(run_metrics, indent=2), "Resource Metrics", allure.attachment_type.JSON)
    attach_chart_to_allure(run_metrics)

    # Compute averages
    avg_gpu = np.mean([m.get("gpu_util", 0) for m in run_metrics])
    avg_cpu = np.mean([m.get("cpu_util", 0) for m in run_metrics])
    avg_mem = np.mean([m.get("gpu_mem_util", 0) for m in run_metrics])

    # Summary
    summary = {
        "test_name": f"inference_load_batch_{batch_size}",
        "gpu_name": metrics_monitor.get_gpu_name(),
        "avg_util": round(avg_gpu, 2),
        "avg_mem": round(avg_mem, 2),
        "avg_cpu_util": round(avg_cpu, 2),
        "fps": fps, # Include the calculated FPS
        "backend": dev_type,
    }

    # Save results
    out_path = os.path.join(RESULTS_DIR, f"{summary['test_name']}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    # Final Metric Attachment for Allure Trending
    allure.attach(f"{fps}", name="Frames Per Second (FPS)", attachment_type=allure.attachment_type.TEXT)