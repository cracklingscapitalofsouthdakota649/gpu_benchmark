# tests/test_nvidia_real_gpu.py
# Real-world NVIDIA GPU compute and memory benchmarks

import os
import time
import pytest
import torch
import allure
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def detect_nvidia_gpu():
    """Check for NVIDIA GPU availability."""
    try:
        return torch.cuda.is_available() and "nvidia" in torch.cuda.get_device_name(0).lower()
    except Exception:
        return False


@pytest.mark.nvidia
@pytest.mark.gpu
class TestRealNvidiaGPU:
    @pytest.fixture(scope="class", autouse=True)
    def setup_device(self):
        if not detect_nvidia_gpu():
            pytest.skip("No NVIDIA GPU detected.")
        torch.cuda.empty_cache()
        device = torch.device("cuda")
        allure.attach(torch.cuda.get_device_name(device), name="GPU Device", attachment_type=allure.attachment_type.TEXT)
        return device

    # ───────────────────────────────────────────────────────────────
    # 1️. Tensor Core GEMM Benchmark (Matrix Multiplication)
    # ───────────────────────────────────────────────────────────────
    @allure.feature("NVIDIA GPU")
    @allure.story("Matrix Multiplication Throughput")
    @pytest.mark.benchmark
    def test_tensor_core_gemm(self, setup_device, benchmark):
        """Benchmark FP16 matrix multiplication using Tensor Cores."""
        device = setup_device
        a = torch.randn(8192, 8192, device=device, dtype=torch.float16)
        b = torch.randn(8192, 8192, device=device, dtype=torch.float16)

        def matmul_op():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            return time.perf_counter() - start

        duration = benchmark(matmul_op)
        tflops = (2 * 8192**3) / (duration * 1e12)
        allure.attach(f"{tflops:.2f} TFLOPs", name="Tensor Core GEMM Performance")
        assert tflops > 30, f"Low Tensor Core throughput: {tflops:.2f} TFLOPs"

    # ───────────────────────────────────────────────────────────────
    # 2️. CNN Inference Benchmark (ResNet-like)
    # ───────────────────────────────────────────────────────────────
    @allure.feature("NVIDIA GPU")
    @allure.story("CNN Forward Pass Throughput")
    @pytest.mark.accelerator
    def test_cnn_forward_throughput(self, setup_device, benchmark):
        """Benchmark ResNet-like CNN forward throughput."""
        device = setup_device

        model = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 1000)
        ).to(device)

        model.eval()
        data = torch.randn((128, 3, 224, 224), device=device)

        def forward_pass():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(data)
            torch.cuda.synchronize()
            return time.perf_counter() - start

        duration = benchmark(forward_pass)
        samples_per_sec = 128 / duration
        allure.attach(f"{samples_per_sec:.2f} img/sec", name="CNN Inference Throughput")
        assert samples_per_sec > 500, f"CNN throughput below expectation: {samples_per_sec:.2f} img/sec"

    # ───────────────────────────────────────────────────────────────
    # 3️. Transformer Encoder Latency (BERT-like)
    # ───────────────────────────────────────────────────────────────
    @allure.feature("NVIDIA GPU")
    @allure.story("Transformer Encoder Inference")
    @pytest.mark.benchmark
    def test_transformer_inference_latency(self, setup_device, benchmark):
        """Simulate Transformer encoder forward latency."""
        device = setup_device
        model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=12).to(device), num_layers=6
        )
        data = torch.randn(32, 64, 768, device=device)

        def transformer_forward():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(data)
            torch.cuda.synchronize()
            return time.perf_counter() - start

        duration = benchmark(transformer_forward)
        latency_ms = duration * 1000
        allure.attach(f"{latency_ms:.2f} ms", name="Transformer Inference Latency")
        assert latency_ms < 200, f"High Transformer latency: {latency_ms:.2f} ms"

    # ───────────────────────────────────────────────────────────────
    # 4️. VRAM Fragmentation and Memory Stability
    # ───────────────────────────────────────────────────────────────
    @allure.feature("NVIDIA GPU")
    @allure.story("VRAM Stability Test")
    @pytest.mark.stress
    def test_vram_allocation_stability(self, setup_device):
        """Allocate/release VRAM repeatedly to detect fragmentation."""
        device = setup_device
        free_before = torch.cuda.mem_get_info(device)[0]

        for _ in range(100):
            tensors = [torch.randn((512, 512, 512), device=device) for _ in range(8)]
            del tensors
            torch.cuda.empty_cache()

        free_after = torch.cuda.mem_get_info(device)[0]
        diff_mb = abs(free_before - free_after) / (1024**2)
        allure.attach(f"Free memory delta: {diff_mb:.2f} MB", name="VRAM Stability")
        assert diff_mb < 200, f"Potential memory leak detected ({diff_mb:.2f} MB lost)"

    # ───────────────────────────────────────────────────────────────
    # 5️. GPU-CPU Transfer Bandwidth (PCIe)
    # ───────────────────────────────────────────────────────────────
    @allure.feature("NVIDIA GPU")
    @allure.story("GPU ↔ CPU Transfer Bandwidth")
    @pytest.mark.benchmark
    def test_gpu_cpu_transfer_bandwidth(self, setup_device, benchmark):
        """Measure GPU to CPU data transfer bandwidth."""
        device = setup_device
        data = torch.randn((1024, 1024, 1024), device=device)

        def transfer_op():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = data.cpu()
            torch.cuda.synchronize()
            return time.perf_counter() - start

        duration = benchmark(transfer_op)
        size_gb = data.numel() * 4 / (1024**3)
        bandwidth = size_gb / duration
        allure.attach(f"{bandwidth:.2f} GB/s", name="PCIe Transfer Bandwidth")
        assert bandwidth > 8, f"Low PCIe bandwidth: {bandwidth:.2f} GB/s"
