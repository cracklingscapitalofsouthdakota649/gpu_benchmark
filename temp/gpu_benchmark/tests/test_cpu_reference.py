# tests/test_cpu_reference.py
import torch
import pytest
import allure
import gc

torch.set_num_threads(8)


def _get_benchmark_mean(benchmark, fn):
    """Run the function with pytest-benchmark and return the mean duration (s)."""
    mean_duration = benchmark.pedantic(
        fn,
        rounds=5,
        iterations=1,
        warmup_rounds=1,
    )
    # In pytest-benchmark 5.x, pedantic() already returns mean time in seconds.
    return float(mean_duration)


# ====================================================================
# 1️. Matrix Multiplication
# ====================================================================
@allure.feature("System Baseline Benchmarks")
@allure.story("CPU Matrix Multiplication Reference")
@pytest.mark.cpu
@pytest.mark.stress
@pytest.mark.timeout(180)
def test_cpu_reference_matmul(benchmark):
    device = "cpu"
    N = 2048
    x = torch.rand(N, N, device=device)
    y = torch.rand(N, N, device=device)

    def matmul_op():
        result = torch.mm(x, y)
        return result.sum().item()

    duration_mean = _get_benchmark_mean(benchmark, matmul_op)
    gc.collect()

    flops = 2 * (N ** 3)
    gflops_sec = flops / (duration_mean * 1e9) if duration_mean > 0 else 0

    print(f"[Matmul] Mean: {duration_mean:.6f}s, GFLOPS: {gflops_sec:.2f}")

    allure.attach(
        f"GFLOPS/sec: {gflops_sec:.2f}",
        name="Performance Metric (Matmul)",
        attachment_type=allure.attachment_type.TEXT,
    )
    allure.dynamic.title(f"CPU Baseline: {N}x{N} Matmul ({round(gflops_sec, 2)} GFLOPS/s)")
    allure.dynamic.severity(allure.severity_level.NORMAL)

    assert duration_mean > 0, "Mean duration must be greater than zero."
    assert gflops_sec > 0, "Matmul GFLOPS/s must be greater than zero."


# ====================================================================
# 2️. 2D Convolution
# ====================================================================
@allure.feature("System Baseline Benchmarks")
@allure.story("CPU 2D Convolution Reference")
@pytest.mark.cpu
@pytest.mark.accelerator
@pytest.mark.timeout(180)
def test_cpu_reference_convolution(benchmark):
    device = "cpu"
    B, C_in, H, W = 1, 64, 512, 512
    C_out, K = 128, 3

    x = torch.rand(B, C_in, H, W, device=device)
    w = torch.rand(C_out, C_in, K, K, device=device)

    def conv_op():
        result = torch.nn.functional.conv2d(x, w, padding=1)
        return result.sum().item()

    duration_mean = _get_benchmark_mean(benchmark, conv_op)
    gc.collect()

    print(f"[Conv2D] Mean: {duration_mean:.6f}s")

    allure.attach(
        f"Duration (Mean): {duration_mean:.4f}s",
        name="Performance Metric (Conv2D)",
        attachment_type=allure.attachment_type.TEXT,
    )
    allure.dynamic.title(f"CPU Baseline: Conv2D ({H}x{W} Input, {C_in}->{C_out} Ch)")
    allure.dynamic.severity(allure.severity_level.NORMAL)

    assert duration_mean > 0, "Conv2D mean duration must be greater than zero."


# ====================================================================
# 3️. Sorting
# ====================================================================
@allure.feature("System Baseline Benchmarks")
@allure.story("CPU Sorting Reference")
@pytest.mark.cpu
@pytest.mark.timeout(180)
def test_cpu_reference_sort(benchmark):
    device = "cpu"
    N_elements = 4 * 1024 * 1024
    x = torch.rand(N_elements, device=device)

    def sort_op():
        result, _ = torch.sort(x)
        return result.sum().item()

    duration_mean = _get_benchmark_mean(benchmark, sort_op)
    gc.collect()

    print(f"[Sort] Mean: {duration_mean:.6f}s")

    allure.attach(
        f"Duration (Mean): {duration_mean:.4f}s for {N_elements} elements",
        name="Performance Metric (Sort)",
        attachment_type=allure.attachment_type.TEXT,
    )
    allure.dynamic.title(f"CPU Baseline: Sort ({N_elements} elements)")
    allure.dynamic.severity(allure.severity_level.NORMAL)

    assert duration_mean > 0, "Sort mean duration must be greater than zero."
