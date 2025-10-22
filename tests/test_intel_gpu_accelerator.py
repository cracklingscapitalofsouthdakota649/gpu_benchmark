# tests/test_intel_gpu_accelerator.py
# Intel GPU accelerator benchmarks (oneAPI / SYCL / OpenCL)
# Works with Intel Arc, Xe MAX, or integrated GPUs that expose OpenCL or oneAPI.

import os
import time
import pytest
import allure
import numpy as np
import torch

try:
    import pyopencl as cl
except ImportError:
    cl = None


def detect_intel_gpu():
    """Detect Intel GPU either via PyTorch (oneAPI/SYCL) or PyOpenCL."""
    try:
        if torch.cuda.is_available() and "intel" in torch.cuda.get_device_name(0).lower():
            return True
    except Exception:
        pass
    if cl:
        try:
            for platform in cl.get_platforms():
                for device in platform.get_devices():
                    if "intel" in device.name.lower():
                        return True
        except Exception:
            pass
    return False


@pytest.mark.gpu
@pytest.mark.intel
class TestIntelGPUAccelerator:
    @pytest.fixture(scope="class", autouse=True)
    def setup_device(self):
        if not detect_intel_gpu():
            pytest.skip("No Intel GPU detected on this system.")
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        allure.attach("Intel GPU detected", name="Intel GPU Device", attachment_type=allure.attachment_type.TEXT)
        return device

    # 1. FP32 matrix multiplication throughput
    @allure.feature("Intel GPU Accelerator")
    @allure.story("Matrix Multiplication Throughput FP32")
    @pytest.mark.benchmark
    def test_fp32_gemm_throughput(self, setup_device, benchmark):
        device = setup_device
        a = torch.randn(4096, 4096, device=device)
        b = torch.randn(4096, 4096, device=device)

        def matmul_op():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            return time.perf_counter() - start

        duration = benchmark(matmul_op)
        tflops = (2 * (4096**3)) / (duration * 1e12)
        allure.attach(f"{tflops:.2f} TFLOPs", name="FP32 GEMM TFLOPs")
        assert tflops > 1, f"Low FP32 GEMM throughput: {tflops:.2f} TFLOPs"

    # 2. FP16 matrix multiplication throughput
    @allure.feature("Intel GPU Accelerator")
    @allure.story("Matrix Multiplication Throughput FP16")
    @pytest.mark.benchmark
    def test_fp16_gemm_throughput(self, setup_device, benchmark):
        device = setup_device
        a = torch.randn(4096, 4096, device=device, dtype=torch.float16)
        b = torch.randn(4096, 4096, device=device, dtype=torch.float16)

        def matmul_op():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            return time.perf_counter() - start

        duration = benchmark(matmul_op)
        tflops = (2 * (4096**3)) / (duration * 1e12)
        allure.attach(f"{tflops:.2f} TFLOPs", name="FP16 GEMM TFLOPs")
        assert tflops > 2, f"Low FP16 GEMM throughput: {tflops:.2f} TFLOPs"

    # 3. Convolution throughput (ResNet-style)
    @allure.feature("Intel GPU Accelerator")
    @allure.story("CNN Forward Pass Throughput")
    @pytest.mark.accelerator
    def test_cnn_forward_throughput(self, setup_device, benchmark):
        device = setup_device
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 7, stride=2, padding=3),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 1000)
        ).to(device)
        data = torch.randn((32, 3, 224, 224), device=device)

        def forward_pass():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(data)
            torch.cuda.synchronize()
            return time.perf_counter() - start

        duration = benchmark(forward_pass)
        throughput = 32 / duration
        allure.attach(f"{throughput:.2f} img/s", name="CNN Throughput")
        assert throughput > 200, f"CNN throughput too low: {throughput:.2f} img/s"

    # 4. Transformer encoder latency
    @allure.feature("Intel GPU Accelerator")
    @allure.story("Transformer Encoder Latency")
    @pytest.mark.benchmark
    def test_transformer_encoder_latency(self, setup_device, benchmark):
        device = setup_device
        model = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=256, nhead=8).to(device),
            num_layers=2
        )
        data = torch.randn((8, 16, 256), device=device)

        def transformer_op():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(data)
            torch.cuda.synchronize()
            return time.perf_counter() - start

        duration = benchmark(transformer_op)
        latency_ms = duration * 1000
        allure.attach(f"{latency_ms:.2f} ms", name="Transformer Latency")
        assert latency_ms < 250, f"Transformer latency too high: {latency_ms:.2f} ms"

    # 5. Device memory bandwidth (approx)
    @allure.feature("Intel GPU Accelerator")
    @allure.story("Device Memory Bandwidth")
    @pytest.mark.benchmark
    def test_memory_bandwidth(self, setup_device, benchmark):
        device = setup_device
        size_mb = 256
        x = torch.randn(size_mb * 256, device=device)
        y = torch.randn(size_mb * 256, device=device)

        def add_op():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = x + y
            torch.cuda.synchronize()
            return time.perf_counter() - start

        duration = benchmark(add_op)
        gbps = (size_mb * 2) / (duration * 1024)
        allure.attach(f"{gbps:.2f} GB/s", name="Memory Bandwidth")
        assert gbps > 50, f"Low memory bandwidth: {gbps:.2f} GB/s"

    # 6. Data transfer CPU<->GPU
    @allure.feature("Intel GPU Accelerator")
    @allure.story("GPU-CPU Transfer Bandwidth")
    @pytest.mark.benchmark
    def test_transfer_bandwidth(self, setup_device, benchmark):
        device = setup_device
        data = torch.randn((256, 256, 256), device=device)

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
        assert bandwidth > 4, f"Low transfer bandwidth: {bandwidth:.2f} GB/s"

    # 7. Mixed precision performance (BF16)
    @allure.feature("Intel GPU Accelerator")
    @allure.story("Mixed Precision BF16 Speedup")
    @pytest.mark.accelerator
    def test_bf16_speedup(self, setup_device, benchmark):
        device = setup_device
        model_fp32 = torch.nn.Linear(2048, 2048).to(device)
        model_bf16 = model_fp32.bfloat16()
        data_fp32 = torch.randn((1024, 2048), device=device)
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
        allure.attach(f"{speedup:.2f}× speedup", name="BF16 Speedup")
        assert speedup >= 1.2, f"Expected ≥1.2× speedup, got {speedup:.2f}×"

    # 8. Multi-threaded OpenCL throughput (if available)
    @allure.feature("Intel GPU Accelerator")
    @allure.story("OpenCL Parallel Throughput")
    @pytest.mark.benchmark
    def test_opencl_parallel(self, benchmark):
        if not cl:
            pytest.skip("PyOpenCL not available.")
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags
        a_np = np.random.rand(1024 * 1024).astype(np.float32)
        b_np = np.random.rand(1024 * 1024).astype(np.float32)
        a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
        b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
        res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
        prg = cl.Program(ctx, """
        __kernel void sum(__global const float *a, __global const float *b, __global float *res) {
            int gid = get_global_id(0);
            res[gid] = a[gid] + b[gid];
        }
        """).build()

        def opencl_sum():
            start = time.perf_counter()
            prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)
            queue.finish()
            return time.perf_counter() - start

        duration = benchmark(opencl_sum)
        allure.attach(f"{duration:.6f}s", name="OpenCL Kernel Duration")
        assert duration < 0.05, f"OpenCL sum kernel too slow: {duration:.6f}s"

    # 9. End-to-end inference throughput
    @allure.feature("Intel GPU Accelerator")
    @allure.story("End-to-End Inference Throughput")
    @pytest.mark.accelerator
    def test_end_to_end_inference_throughput(self, setup_device, benchmark):
        device = setup_device
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 10)
        ).to(device)
        batch = torch.randn((64, 3, 224, 224), device=device)

        def inference_op():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(batch)
            torch.cuda.synchronize()
            return time.perf_counter() - start

        duration = benchmark(inference_op)
        samples_per_sec = 64 / duration
        allure.attach(f"{samples_per_sec:.2f} samples/sec", name="End-to-End Throughput")
        assert samples_per_sec > 150, f"Low inference throughput: {samples_per_sec:.2f} samples/sec"
