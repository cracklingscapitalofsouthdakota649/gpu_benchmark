# tests/test_gpu_mixed_precision.py

import torch
import pytest
import allure
import json # <-- ADDED
from supports.gpu_monitor import collect_gpu_metrics # <-- ADDED

@allure.feature("GPU Specialized Performance")
@allure.story("Mixed Precision Matrix Multiplication")
@pytest.mark.gpu
def test_mixed_precision_matmul(benchmark):
    """Benchmark matrix multiplication using mixed precision and track TFLOPS."""
    if not torch.cuda.is_available():
        pytest.skip("Mixed precision requires CUDA.")
        
    device = "cuda"
    N = 2048
    x = torch.randn(N, N, device=device, dtype=torch.float16)
    y = torch.randn(N, N, device=device, dtype=torch.float16)

    def matmul():
        # autocast allows the use of Tensor Cores for high-speed FP16 math
        with torch.cuda.amp.autocast():
            z = x @ y
            torch.cuda.synchronize()

    # --- 1. Run the benchmark ---
    result = benchmark(matmul)
    
    # --- 2. Calculate TFLOPS/sec (FLOPS are the same as FP32, but result is faster) ---
    # FLOPS for N x N * N x N is 2 * N^3
    flops = 2 * (N ** 3) 
    duration_mean = result.stats.mean
    tflops_sec = flops / (duration_mean * 1e12) # FLOPS / (Time * 10^12)

    # --- 3. Collect GPU Telemetry (3-second sample) ---
    telemetry = collect_gpu_metrics(duration=3, interval=0.1) 
    
    # --- 4. Attach Metrics to Allure ---
    with allure.step("Performance Metrics"):
        allure.attach(f"{tflops_sec:.2f}", 
                      name="Mixed Precision TFLOPS/sec", 
                      attachment_type=allure.attachment_type.TEXT)
        allure.attach(
            json.dumps(telemetry, indent=2),
            name="GPU/CPU Utilization (JSON)",
            attachment_type=allure.attachment_type.JSON,
        )    