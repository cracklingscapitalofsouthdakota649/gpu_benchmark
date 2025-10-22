# tests/test_directml_gpu_accelerator.py
# Real DirectML GPU accelerator benchmarks (Windows)
# Compatible with NVIDIA, AMD, and Intel GPUs via torch-directml backend

import os
import time
import pytest
import allure
import numpy as np
import torch

try:
    import torch_directml
except ImportError:
    torch_directml = None


def detect_directml():
    """Check if DirectML backend is available."""
    if not torch_directml:
        return False
    try:
        device = torch_directml.device()
        return "DirectML" in str(device)
    except Exception:
        return False


@pytest.mark.gpu
@pytest.mark.accelerator
class TestDirectMLAccelerator:
    @pytest.fixture(scope="class", autouse=True)
    def setup_device(self):
        """Initialize DirectML device if available."""
        if not detect_directml():
            pytest.skip("DirectML not available or not initialized.")
        device = torch_directml.device()
        allure.attach("DirectML backend detected", name="DirectML Device", attachment_type=allure.attachment_type.TEXT)
        return device

    # 1. FP32 matrix multiplication throughput
    @allure.feature("DirectML GPU Accelerator")
    @allure.story("FP32 Matrix Multiplication")
    @pytest.mark.benchmark
    def test_fp32_matmul_throughput(self, setup_device, benchmark):
        device = setup_device
        a = torch.randn(2048, 2048, device=device)
        b = torch.randn(2048, 2048, device=device)

        def matmul_op():
            start = time.perf_counter()
            _ = torch.matmul(a, b)
            return time.perf_counter() - start

        duration = benchmark(matmul_op)
        gflops = (2 * (2048 ** 3)) / (duration * 1e9)
        allure.attach(f"{gflops:.2f} GFLOPs", name="FP32 MatMul Performance")
        assert gflops > 200, f"Low FP32 MatMul throughput: {gflops:.2f} GFLOPs"

    # 2. FP16 GEMM test
    @allure.feature("DirectML GPU Accelerator")
    @allure.story("FP16 Matrix Multiplication")
    def test_fp16_gemm(self, setup_device, benchmark):
        device = setup_device
        a = torch.randn(2048, 2048, device=device, dtype=torch.float16)
        b = torch.randn(2048, 2048, device=device, dtype=torch.float16)

        def matmul_op():
            start = time.perf_counter()
            _ = torch.matmul(a, b)
            return time.perf_counter() - start

        duration = benchmark(matmul_op)
        gflops = (2 * (2048 ** 3)) / (duration * 1e9)
        allure.attach(f"{gflops:.2f} GFLOPs", name="FP16 GEMM Performance")
        assert gflops > 300, f"Low FP16 GEMM performance: {gflops:.2f} GFLOPs"

    # 3. CNN forward throughput
    @allure.feature("DirectML GPU Accelerator")
    @allure.story("CNN Forward Throughput")
    def test_cnn_forward(self, setup_device, benchmark):
        device = setup_device
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 10)
        ).to(device)
        x = torch.randn(32, 3, 224, 224, device=device)

        def forward_op():
            start = time.perf_counter()
            _ = model(x)
            return time.perf_counter() - start

        duration = benchmark(forward_op)
        img_s = 32 / duration
        allure.attach(f"{img_s:.2f} img/s", name="CNN Throughput")
        assert img_s > 200, f"Low CNN throughput: {img_s:.2f} img/s"

    # 4. Transformer latency
    @allure.feature("DirectML GPU Accelerator")
    @allure.story("Transformer Encoder Latency")
    def test_transformer_latency(self, setup_device, benchmark):
        device = setup_device
        model = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=256, nhead=8).to(device),
            num_layers=2
        )
        data = torch.randn(8, 16, 256, device=device)

        def transformer_op():
            start = time.perf_counter()
            _ = model(data)
            return time.perf_counter() - start

        duration = benchmark(transformer_op)
        latency = duration * 1000
        allure.attach(f"{latency:.2f} ms", name="Transformer Latency")
        assert latency < 300, f"Transformer latency too high: {latency:.2f} ms"

    # 5. Tensor add bandwidth
    @allure.feature("DirectML GPU Accelerator")
    @allure.story("Tensor Add Bandwidth")
    def test_tensor_add_bandwidth(self, setup_device, benchmark):
        device = setup_device
        size_mb = 256
        x = torch.randn(size_mb * 256, device=device)
        y = torch.randn(size_mb * 256, device=device)

        def add_op():
            start = time.perf_counter()
            _ = x + y
            return time.perf_counter() - start

        duration = benchmark(add_op)
        bandwidth = (size_mb * 2) / (duration * 1024)
        allure.attach(f"{bandwidth:.2f} GB/s", name="Tensor Add Bandwidth")
        assert bandwidth > 40, f"Low tensor add bandwidth: {bandwidth:.2f} GB/s"

    # 6. Memory allocation + release
    @allure.feature("DirectML GPU Accelerator")
    @allure.story("Memory Alloc/Free Latency")
    def test_memory_allocation(self, setup_device, benchmark):
        device = setup_device

        def alloc_free():
            start = time.perf_counter()
            x = torch.randn(1024, 1024, device=device)
            del x
            return time.perf_counter() - start

        duration = benchmark(alloc_free)
        allure.attach(f"{duration*1000:.3f} ms", name="Alloc/Free Latency")
        assert duration < 0.1, f"High allocation latency: {duration:.3f}s"

    # 7. BatchNorm throughput
    @allure.feature("DirectML GPU Accelerator")
    @allure.story("BatchNorm Throughput")
    def test_batchnorm_throughput(self, setup_device, benchmark):
        device = setup_device
        bn = torch.nn.BatchNorm2d(128).to(device)
        x = torch.randn(64, 128, 56, 56, device=device)

        def batchnorm_op():
            start = time.perf_counter()
            _ = bn(x)
            return time.perf_counter() - start

        duration = benchmark(batchnorm_op)
        fps = 64 / duration
        allure.attach(f"{fps:.2f} batch/s", name="BatchNorm FPS")
        assert fps > 250, f"Low BatchNorm throughput: {fps:.2f} batch/s"

    # 8. Mixed precision (BF16) gain
    @allure.feature("DirectML GPU Accelerator")
    @allure.story("Mixed Precision Gain")
    def test_mixed_precision_gain(self, setup_device, benchmark):
        device = setup_device
        model_fp32 = torch.nn.Linear(2048, 2048).to(device)
        model_fp16 = model_fp32.half()
        x_fp32 = torch.randn(1024, 2048, device=device)
        x_fp16 = x_fp32.half()

        def run_fp32():
            start = time.perf_counter()
            _ = model_fp32(x_fp32)
            return time.perf_counter() - start

        def run_fp16():
            start = time.perf_counter()
            _ = model_fp16(x_fp16)
            return time.perf_counter() - start

        t32 = benchmark(run_fp32)
        t16 = benchmark(run_fp16)
        speedup = t32 / t16 if t16 > 0 else 0
        allure.attach(f"{speedup:.2f}×", name="Mixed Precision Speedup")
        assert speedup >= 1.2, f"Expected ≥1.2× speedup, got {speedup:.2f}×"

    # 9. End-to-end inference throughput
    @allure.feature("DirectML GPU Accelerator")
    @allure.story("End-to-End Inference")
    def test_end_to_end_inference(self, setup_device, benchmark):
        device = setup_device
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 10)
        ).to(device)
        data = torch.randn(64, 3, 224, 224, device=device)

        def inference_op():
            start = time.perf_counter()
            _ = model(data)
            return time.perf_counter() - start

        duration = benchmark(inference_op)
        throughput = 64 / duration
        allure.attach(f"{throughput:.2f} samples/s", name="E2E Throughput")
        assert throughput > 150, f"Low inference throughput: {throughput:.2f} samples/s"
