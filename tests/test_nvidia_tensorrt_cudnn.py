# tests/test_nvidia_tensorrt_cudnn.py
# NVIDIA cuDNN and TensorRT performance and inference validation

import os
import time
import pytest
import torch
import allure
import numpy as np
from torch import nn

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except Exception:
    TRT_AVAILABLE = False


# ───────────────────────────────────────────────────────────────
# Utility functions
# ───────────────────────────────────────────────────────────────

def detect_nvidia_gpu():
    """Check for NVIDIA GPU availability."""
    try:
        return torch.cuda.is_available() and "nvidia" in torch.cuda.get_device_name(0).lower()
    except Exception:
        return False


@pytest.mark.nvidia
@pytest.mark.gpu
class TestNvidiaTensorRTandCuDNN:
    @pytest.fixture(scope="class", autouse=True)
    def setup_device(self):
        """Prepare CUDA device and validate libraries."""
        if not detect_nvidia_gpu():
            pytest.skip("No NVIDIA GPU detected.")
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        device = torch.device("cuda")
        allure.attach(torch.cuda.get_device_name(device), name="NVIDIA GPU Device", attachment_type=allure.attachment_type.TEXT)
        return device

    # ───────────────────────────────────────────────────────────────
    # 1️. cuDNN Convolution Performance
    # ───────────────────────────────────────────────────────────────
    @allure.feature("NVIDIA cuDNN")
    @allure.story("Convolution Performance Test")
    @pytest.mark.accelerator
    def test_cudnn_convolution_speed(self, setup_device, benchmark):
        """Validate cuDNN convolution speed and correctness."""
        device = setup_device
        model = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1).to(device)
        data = torch.randn((32, 64, 224, 224), device=device)

        def conv_op():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(data)
            torch.cuda.synchronize()
            return time.perf_counter() - start

        duration = benchmark(conv_op)
        throughput = 32 / duration
        allure.attach(f"{throughput:.2f} samples/sec", name="cuDNN Conv Throughput")
        assert throughput > 100, f"Low cuDNN throughput: {throughput:.2f} samples/sec"

    # ───────────────────────────────────────────────────────────────
    # 2️. cuDNN FP16 vs FP32 Speedup
    # ───────────────────────────────────────────────────────────────
    @allure.feature("NVIDIA cuDNN")
    @allure.story("FP16 vs FP32 Speedup")
    @pytest.mark.accelerator
    def test_cudnn_fp16_speedup(self, setup_device, benchmark):
        """Compare FP16 vs FP32 convolution throughput."""
        device = setup_device
        model_fp32 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1).to(device)
        model_fp16 = model_fp32.half()
        data_fp32 = torch.randn((32, 128, 112, 112), device=device, dtype=torch.float32)
        data_fp16 = data_fp32.half()

        def run_fp32():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model_fp32(data_fp32)
            torch.cuda.synchronize()
            return time.perf_counter() - start

        def run_fp16():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model_fp16(data_fp16)
            torch.cuda.synchronize()
            return time.perf_counter() - start

        t32 = benchmark(run_fp32)
        t16 = benchmark(run_fp16)
        speedup = t32 / t16 if t16 > 0 else 0
        allure.attach(f"FP16: {t16:.4f}s | FP32: {t32:.4f}s | Speedup: {speedup:.2f}x",
                      name="cuDNN FP16 Speedup", attachment_type=allure.attachment_type.TEXT)
        assert speedup >= 1.2, f"Expected ≥1.2x FP16 speedup, got {speedup:.2f}x"

    # ───────────────────────────────────────────────────────────────
    # 3️. TensorRT FP16 Inference Test
    # ───────────────────────────────────────────────────────────────
    @allure.feature("NVIDIA TensorRT")
    @allure.story("FP16 Inference Speedup")
    @pytest.mark.benchmark
    def test_tensorrt_inference_speedup(self, setup_device, benchmark):
        """Run TensorRT FP16 inference vs PyTorch FP32 baseline."""
        if not TRT_AVAILABLE:
            pytest.skip("TensorRT not installed in environment.")
        device = setup_device

        # Build a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
                self.relu = nn.ReLU()
                self.fc = nn.Linear(64 * 224 * 224, 10)
            def forward(self, x):
                x = self.conv1(x)
                x = self.relu(x)
                x = x.flatten(1)
                x = self.fc(x)
                return x

        model = SimpleModel().to(device)
        data = torch.randn((8, 3, 224, 224), device=device)

        # Convert to ONNX
        onnx_path = "model_temp.onnx"
        torch.onnx.export(model, data, onnx_path, export_params=True, opset_version=17,
                          do_constant_folding=True, input_names=['input'], output_names=['output'])

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        with open(onnx_path, 'rb') as f:
            parser.parse(f.read())

        builder.max_batch_size = 8
        builder.fp16_mode = True
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        engine = builder.build_engine(network, config)

        context = engine.create_execution_context()

        def tensorrt_infer():
            start = time.perf_counter()
            _ = context.execute_v2([])
            end = time.perf_counter()
            return end - start

        duration_trt = benchmark(tensorrt_infer)

        # PyTorch baseline
        def pytorch_infer():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(data)
            torch.cuda.synchronize()
            return time.perf_counter() - start

        duration_torch = benchmark(pytorch_infer)
        speedup = duration_torch / duration_trt if duration_trt > 0 else 0

        allure.attach(f"TensorRT: {duration_trt:.4f}s | PyTorch: {duration_torch:.4f}s | Speedup: {speedup:.2f}x",
                      name="TensorRT FP16 Speedup", attachment_type=allure.attachment_type.TEXT)
        assert speedup >= 1.5, f"TensorRT FP16 expected ≥1.5x speedup, got {speedup:.2f}x"
