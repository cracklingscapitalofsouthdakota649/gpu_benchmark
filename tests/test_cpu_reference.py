# tests/test_cpu_reference.py
import torch
import pytest
import allure
import json # <-- ADDED
from supports.gpu_monitor import collect_gpu_metrics # <-- ADDED

@allure.feature("System Baseline Benchmarks")
@allure.story("CPU Matrix Multiplication Reference")
@pytest.mark.cpu
def test_cpu_reference_operations(benchmark):
    """Benchmark CPU matrix multiplication and track GFLOPS."""
    device = "cpu"
    N = 4096
    x = torch.rand(N, N, device=device)
    y = torch.rand(N, N, device=device)

    def cpu_op():
        return torch.mm(x, y)

    # --- 1. Run the benchmark ---
    result = benchmark(cpu_op)
    
    # --- 2. Calculate GFLOPS/sec ---
    # FLOPS for N x N * N x N is 2 * N^3
    flops = 2 * (N ** 3)
    duration_mean = result.stats.mean
    gflops_sec = flops / (duration_mean * 1e9) # FLOPS / (Time * 10^9)

    # --- 3. Collect Telemetry (3-second sample) ---
    # collect_gpu_metrics still captures CPU utilization via psutil.
    telemetry = collect_gpu_metrics(duration=3, interval=0.1) 
    
    # --- 4. Attach Metrics to Allure ---
    with allure.step("Performance Metrics"):
        allure.attach(f"{gflops_sec:.2f}", 
                      name="CPU GFLOPS/sec", 
                      attachment_type=allure.attachment_type.TEXT)
        allure.attach(
            json.dumps(telemetry, indent=2),
            name="System Utilization (JSON)",
            attachment_type=allure.attachment_type.JSON,
        )