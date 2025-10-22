# tests/test_nvidia_gpu_accelerator.py
# NVIDIA GPU validation, kernel performance, and tensor core benchmarks

import os
import time
import torch
import pytest
import allure
import numpy as np
from torch import nn

# ───────────────────────────────────────────────────────────────
# Detection utilities
# ───────────────────────────────────────────────────────────────

def detect_nvidia_gpu():
    """Return True if an NVIDIA GPU with CUDA support is available."""
    try:
        return torch.cuda.is_available() and torch.cuda.get_device_name(0).lower().startswith("nvidia")
    except Exception:
        return False


@pytest.mark.gpu
@pytest.mark.nvidia
class TestNvidiaGPUAccelerator:
    @pytest.fixture(scope="class", autouse=True)
    def setup_device(self):
        """Prepare the CUDA device."""
        if not detect_nvidia_gpu():
            pytest.skip("No NVIDIA GPU detected.")
        torch.cuda.empty_cache()
        device = torch.device("cuda")
        allure.attach(torch.cuda.get_device_name(device), name="GPU Device", attachment_type=allure.attachment_type.TEXT)
        return device

    # ───────────────────────────────────────────────────────────────
    # 1️. CUDA Kernel Latency + Synchronization Test
    # ───────────────────────────────────────────────────────────────
    @allure.feature("NVIDIA GPU Accelerator")
    @allure.story("CUDA Kernel Launch Latency")
    @pytest.mark.accelerator
    def test_cuda_kernel_latency(self, setup_device, benchmark):
        device = setup_device
        x = torch.randn((4096, 4096), device=device)
        y = torch.randn((4096, 4096), device=device)

        def matmul_op():
            torch.cuda.synchronize()
            start = time.perf_counter()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            return time.perf_counter() - start

        latency = benchmark(matmul_op)
        allure.attach(f"{latency*1000:.3f} ms", name="CUDA MatMul Latency")
        assert latency < 100, f"Kernel latency too high: {latency:.3f}s"

    # ───────────────────────────────────────────────────────────────
    # 2️. Tensor Core FP16 vs FP32 Speed Comparison
    # ───────────────────────────────────────────────────────────────
    @allure.feature("NVIDIA GPU Accelerator")
    @allure.story("Tensor Core FP16 vs FP32 Speedup")
    @pytest.mark.accelerator
    def test_tensor_core_speedup(self, setup_device, benchmark):
        device = setup_device
        if not torch.cuda.is_bf16_supported():
            pytest.skip("Tensor cores or mixed-precision not supported on this GPU.")

        x_fp32 = torch.randn((8192, 8192), device=device, dtype=torch.float32)
        y_fp32 = torch.randn((8192, 8192), device=device, dtype=torch.float32)
        x_fp16 = x_fp32.half()
        y_fp16 = y_fp32.half()

        def fp32_op():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = torch.matmul(x_fp32, y_fp32)
            torch.cuda.synchronize()
            return time.perf_counter() - start

        def fp16_op():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = torch.matmul(x_fp16, y_fp16)
            torch.cuda.synchronize()
            return time.perf_counter() - start

        t_fp32 = benchmark(fp32_op)
        t_fp16 = benchmark(fp16_op)
        speedup = t_fp32 / t_fp16 if t_fp16 > 0 else 0

        allure.attach(f"FP16: {t_fp16:.4f}s | FP32: {t_fp32:.4f}s | Speedup: {speedup:.2f}x",
                      name="Tensor Core FP16 Speedup", attachment_type=allure.attachment_type.TEXT)
        assert speedup >= 1.2, f"Expected at least 1.2x Tensor Core speedup, got {speedup:.2f}x"

    # ───────────────────────────────────────────────────────────────
    # 3️. GPU Memory Bandwidth Test
    # ───────────────────────────────────────────────────────────────
    @allure.feature("NVIDIA GPU Accelerator")
    @allure.story("GPU Memory Bandwidth")
    @pytest.mark.benchmark
    def test_gpu_memory_bandwidth(self, setup_device, benchmark):
        device = setup_device
        size_mb = 1024
        a = torch.randn(size_mb * 256, device=device)
        b = torch.randn(size_mb * 256, device=device)

        def bandwidth_op():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = a + b
            torch.cuda.synchronize()
            return time.perf_counter() - start

        duration = benchmark(bandwidth_op)
        bandwidth_gbps = (size_mb * 2) / duration / 1024
        allure.attach(f"{bandwidth_gbps:.2f} GB/s", name="Memory Bandwidth")
        assert bandwidth_gbps > 200, f"Low memory bandwidth: {bandwidth_gbps:.2f} GB/s"

    # ───────────────────────────────────────────────────────────────
    # 4️. Multi-GPU Synchronization + Scaling Test
    # ───────────────────────────────────────────────────────────────
    @allure.feature("NVIDIA GPU Accelerator")
    @allure.story("Multi-GPU Synchronization")
    @pytest.mark.stress
    def test_multi_gpu_sync(self):
        if torch.cuda.device_count() < 2:
            pytest.skip("Test requires at least 2 NVIDIA GPUs.")

        tensors = [torch.randn((2048, 2048), device=f"cuda:{i}") for i in range(torch.cuda.device_count())]
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = sum(t.sum().item() for t in tensors)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        allure.attach(f"{elapsed:.3f}s across {torch.cuda.device_count()} GPUs", name="Multi-GPU Sync Time")
        assert elapsed < 3.0, f"Slow multi-GPU synchronization: {elapsed:.3f}s"
