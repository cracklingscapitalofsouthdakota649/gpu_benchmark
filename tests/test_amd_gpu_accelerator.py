# tests/test_amd_gpu_accelerator.py
# AMD GPU accelerator benchmarks (ROCm / HIP) for real-world workloads

import os
import time
import pytest
import torch
import allure
import numpy as np
from torch import nn

def detect_amd_gpu():
    """Return True if an AMD GPU with ROCm/HIP support is available."""
    try:
        return torch.cuda.is_available() and "amd" in torch.cuda.get_device_name(0).lower()
    except Exception:
        return False

@pytest.mark.gpu
@pytest.mark.amd
class TestAmdGPUAccelerator:
    @pytest.fixture(scope="class", autouse=True)
    def setup_device(self):
        if not detect_amd_gpu():
            pytest.skip("No AMD GPU detected.")
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        allure.attach(torch.cuda.get_device_name(0), name="AMD GPU Device", attachment_type=allure.attachment_type.TEXT)
        return device

    # 1. GEMM (matrix multiplication) FP16 throughput benchmark
    @allure.feature("AMD GPU Accelerator")
    @allure.story("Matrix Multiplication Throughput FP16")
    @pytest.mark.benchmark
    def test_fp16_gemm_throughput(self, setup_device, benchmark):
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
        tflops = (2 * (8192**3)) / (duration * 1e12)
        allure.attach(f"{tflops:.2f} TFLOPs", name="FP16 GEMM TFLOPs")
        assert tflops > 20, f"Low FP16 GEMM throughput: {tflops:.2f} TFLOPs"

    # 2. CNN forward pass throughput (ResNet-style)
    @allure.feature("AMD GPU Accelerator")
    @allure.story("CNN Forward Pass Throughput")
    @pytest.mark.accelerator
    def test_cnn_forward_throughput(self, setup_device, benchmark):
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
        data = torch.randn((64, 3, 224, 224), device=device)

        def forward_pass():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(data)
            torch.cuda.synchronize()
            return time.perf_counter() - start

        duration = benchmark(forward_pass)
        images_per_sec = 64 / duration
        allure.attach(f"{images_per_sec:.2f} img/sec", name="CNN Throughput")
        assert images_per_sec > 300, f"CNN throughput too low: {images_per_sec:.2f} img/sec"

    # 3. Transformer encoder latency (BERT-style)
    @allure.feature("AMD GPU Accelerator")
    @allure.story("Transformer Encoder Latency")
    @pytest.mark.benchmark
    def test_transformer_encoder_latency(self, setup_device, benchmark):
        device = setup_device
        model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8).to(device),
            num_layers=4
        )
        data = torch.randn((16, 32, 512), device=device)

        def transformer_op():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(data)
            torch.cuda.synchronize()
            return time.perf_counter() - start

        duration = benchmark(transformer_op)
        latency_ms = duration * 1000
        allure.attach(f"{latency_ms:.2f} ms", name="Transformer Latency")
        assert latency_ms < 150, f"Transformer latency too high: {latency_ms:.2f} ms"

    # 4. Memory bandwidth test (device-to-device)
    @allure.feature("AMD GPU Accelerator")
    @allure.story("Device Memory Bandwidth")
    @pytest.mark.benchmark
    def test_device_memory_bandwidth(self, setup_device, benchmark):
        device = setup_device
        size_mb = 1024
        x = torch.randn(size_mb * 256, device=device)
        y = torch.randn(size_mb * 256, device=device)

        def bandwidth_op():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = x + y
            torch.cuda.synchronize()
            return time.perf_counter() - start

        duration = benchmark(bandwidth_op)
        gbps = (size_mb * 2) / (duration * 1024)
        allure.attach(f"{gbps:.2f} GB/s", name="Device Memory Bandwidth")
        assert gbps > 150, f"Low device memory bandwidth: {gbps:.2f} GB/s"

    # 5. GPU↔CPU transfer bandwidth (PCIe) test
    @allure.feature("AMD GPU Accelerator")
    @allure.story("GPU-CPU Transfer Bandwidth")
    @pytest.mark.benchmark
    def test_gpu_cpu_transfer_bandwidth(self, setup_device, benchmark):
        device = setup_device
        data = torch.randn((512, 512, 512), device=device)

        def transfer_op():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = data.cpu()
            torch.cuda.synchronize()
            return time.perf_counter() - start

        duration = benchmark(transfer_op)
        size_gb = data.numel() * 4 / (1024**3)
        bandwidth = size_gb / duration
        allure.attach(f"{bandwidth:.2f} GB/s", name="PCIe Bandwidth")
        assert bandwidth > 6, f"Low GPU-CPU transfer bandwidth: {bandwidth:.2f} GB/s"

    # 6. Mixed precision (BF16) speedup test
    @allure.feature("AMD GPU Accelerator")
    @allure.story("Mixed Precision (BF16) Speedup")
    @pytest.mark.accelerator
    def test_mixed_precision_bf16_speedup(self, setup_device, benchmark):
        device = setup_device
        model_fp32 = nn.Conv2d(128, 256, 3, stride=1, padding=1).to(device)
        model_bf16 = model_fp32.bfloat16()
        data_fp32 = torch.randn((32, 128, 112, 112), device=device, dtype=torch.float32)
        data_bf16 = data_fp32.bfloat16()

        def run_fp32():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model_fp32(data_fp32)
            torch.cuda.synchronize()
            return time.perf_counter() - start

        def run_bf16():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model_bf16(data_bf16)
            torch.cuda.synchronize()
            return time.perf_counter() - start

        t32 = benchmark(run_fp32)
        t16 = benchmark(run_bf16)
        speedup = t32 / t16 if t16 > 0 else 0
        allure.attach(f"FP32: {t32:.4f}s | BF16: {t16:.4f}s | Speedup: {speedup:.2f}×",
                      name="BF16 Speedup", attachment_type=allure.attachment_type.TEXT)
        assert speedup >= 1.3, f"Expected ≥1.3× BF16 speedup, got {speedup:.2f}×"

    # 7. VRAM fragmentation / stability test
    @allure.feature("AMD GPU Accelerator")
    @allure.story("VRAM Stability Test")
    @pytest.mark.stress
    def test_vram_fragmentation_stability(self, setup_device):
        device = setup_device
        free_before = torch.cuda.mem_get_info(device)[0]

        for _ in range(50):
            tensors = [torch.randn((256,256,256), device=device) for _ in range(4)]
            del tensors
            torch.cuda.empty_cache()

        free_after = torch.cuda.mem_get_info(device)[0]
        diff_mb = abs(free_before - free_after) / (1024**2)
        allure.attach(f"Free memory delta: {diff_mb:.2f} MB", name="VRAM Stability")
        assert diff_mb < 100, f"Potential memory leak or fragmentation >100 MB: {diff_mb:.2f} MB"

    # 8. Multi-GPU scaling & synchronization (if >1 GPU)
    @allure.feature("AMD GPU Accelerator")
    @allure.story("Multi-GPU Sync & Scaling")
    @pytest.mark.stress
    def test_multi_gpu_scaling(self, setup_device):
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires at least 2 AMD GPUs for multi-GPU scaling.")
        count = torch.cuda.device_count()
        tensors = [torch.randn((2048,2048), device=f"cuda:{i}") for i in range(count)]
        torch.cuda.synchronize()
        start = time.perf_counter()
        results = [t.sum().item() for t in tensors]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        allure.attach(f"{elapsed:.3f}s across {count} GPUs", name="Multi-GPU Sync Time")
        assert elapsed < 4.0, f"Multi-GPU sync exceeded threshold: {elapsed:.3f}s"

    # 9. End-to-end model inference throughput (vision classification)
    @allure.feature("AMD GPU Accelerator")
    @allure.story("End-to-end Inference Throughput")
    @pytest.mark.accelerator
    def test_end_to_end_inference_throughput(self, setup_device, benchmark):
        device = setup_device
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64, 100)
        ).to(device)
        model.eval()
        batch = torch.randn((128, 3, 224, 224), device=device)

        def inference_op():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(batch)
            torch.cuda.synchronize()
            return time.perf_counter() - start

        duration = benchmark(inference_op)
        samples_per_sec = 128 / duration
        allure.attach(f"{samples_per_sec:.2f} samples/sec", name="End-to-End Throughput")
        assert samples_per_sec > 250, f"Inference throughput too low: {samples_per_sec:.2f} samples/sec"
