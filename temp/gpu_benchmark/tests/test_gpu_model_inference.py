#test/test_gpu_model_inference.py

# tests/test_inference_load.py
import os
import json
import time
import allure
import pytest
import torch
import numpy as np

# Assuming these scripts/modules exist and handle data collection/plotting
from scripts.plot_gpu_metrics import attach_chart_to_allure
from scripts.system_metrics import SystemMetrics
from tests.device_utils import pick_device, get_device, synchronize

RESULTS_DIR = "allure-results"

@allure.feature("GPU Performance Benchmarks")
@allure.story("Inference FPS Benchmark")
@pytest.mark.gpu
@pytest.mark.cpu
@pytest.mark.benchmark
@pytest.mark.parametrize("batch_size", [128, 512, 1024])
def test_inference_load(batch_size):  # Removed 'benchmark' fixture
    dev_type = pick_device()
    device = get_device(dev_type)
    metrics_monitor = SystemMetrics(device)

    model = torch.nn.Linear(1024, 512).to(device)
    x = torch.randn(batch_size, 1024, dtype=torch.float32).to(device)
    
    # --- FPS Graph Implementation ---
    run_metrics = []
    num_warmup_steps = 10
    num_metric_steps = 30
    
    # Warm-up (crucial for accurate GPU timing)
    for _ in range(num_warmup_steps):
        _ = model(x)
        synchronize(dev_type)

    # Collect metrics for visualization and compute per-step FPS
    total_fps = []
    for i in range(num_metric_steps):
        # Start timing for FPS
        start_time = time.time() 
        
        _ = model(x)
        synchronize(dev_type)
        
        # End timing and calculate FPS
        end_time = time.time()
        duration = end_time - start_time
        
        # Handle division by zero safety
        current_fps = round(batch_size / duration, 2) if duration > 0 else 0.0

        # Collect all metrics
        metrics = metrics_monitor.get_all_metrics()
        metrics["step"] = i
        metrics["fps"] = current_fps  # ⬅️ ADDED FPS METRIC HERE
        run_metrics.append(metrics)
        total_fps.append(current_fps)
        
        time.sleep(0.05) # Introduce a small delay for stable monitoring

    # --- End FPS Graph Implementation ---

    # Compute averages
    avg_gpu = np.mean([m.get("gpu_util", 0) for m in run_metrics])
    avg_cpu = np.mean([m.get("cpu_util", 0) for m in run_metrics])
    avg_mem = np.mean([m.get("gpu_mem_util", 0) for m in run_metrics])
    avg_fps = np.mean(total_fps) if total_fps else 0.0

    # Attach charts safely (The 'attach_chart_to_allure' script must be updated
    # to recognize and plot the "fps" key from run_metrics)
    attach_chart_to_allure(run_metrics)

    # Summary
    summary = {
        "test_name": f"inference_load_batch_{batch_size}",
        "gpu_name": metrics_monitor.get_gpu_name(),
        "avg_util": round(avg_gpu, 2),
        "avg_mem": round(avg_mem, 2),
        "avg_cpu_util": round(avg_cpu, 2),
        "fps": round(avg_fps, 2), # Use the average FPS in the summary
        "backend": dev_type,
    }
    
    # Attach FPS as a single result to Allure
    allure.attach(f"{round(avg_fps, 2)}", name="Average Frames Per Second", attachment_type=allure.attachment_type.TEXT)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_file = os.path.join(RESULTS_DIR, f"summary_inference_load_{batch_size}.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    allure.attach(json.dumps(summary, indent=2), "Performance Summary", allure.attachment_type.JSON)
    
    # Set Allure test name dynamically
    allure.dynamic.title(f"Inference Load (Batch {batch_size}) - FPS: {round(avg_fps, 2)}")