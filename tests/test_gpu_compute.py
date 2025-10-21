# tests/test_gpu_compute.py
import torch
import pytest
import time
import allure # Import allure is generally good practice when using decorators

@allure.feature("GPU Core Compute Benchmarks")  # ⬅️ ADDED
@allure.story("Tensor Arithmetic Operations")   # ⬅️ ADDED
@pytest.mark.gpu
def test_tensor_operations_benchmark(benchmark):
    """Benchmark GPU tensor arithmetic (element-wise ops)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand(10_000_000, device=device)
    y = torch.rand(10_000_000, device=device)

    def compute():
        z = x * y + x / (y + 1e-5)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        return z

    result = benchmark(compute)
    assert result is not None