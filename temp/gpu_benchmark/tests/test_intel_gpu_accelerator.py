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
        
        # If PyTorch is not using an Intel device (i.e., it's a "cpu" fallback
        # or another backend like OpenCL is used), we need to handle that.
        # This test class primarily focuses on the oneAPI/SYCL path via PyTorch-XPU,
        # but uses the fixture to setup the environment.
        if torch.cuda.is_available() and "intel" in torch.cuda.get_device_name(0).lower():
            device_name = torch.cuda.get_device_name(0)
            allure.attach(f"PyTorch-XPU backend detected: {device_name}", 
                          name="Intel GPU Device", attachment_type=allure.attachment_type.TEXT)
            torch.cuda.empty_cache()
            return torch.device("cuda")
        
        # Fallback to a CPU device for cases where only OpenCL is detected 
        # but the test relies on PyTorch. These tests will be slow but will
        # still run unless explicitly skipped in the test method.
        # OpenCL-specific tests will be run separately (e.g., test_opencl_kernel_execution).
        allure.attach("Intel GPU detected via OpenCL only. PyTorch tests will run on CPU or skip.", 
                      name="Intel GPU Device (OpenCL Fallback)", attachment_type=allure.attachment_type.TEXT)
        
        # For PyTorch tests, return 'cpu' if no XPU is found, forcing them to run slow or fail.
        # Only the OpenCL-specific test will run on the OpenCL backend.
        return torch.device("cpu")
        

    # 1. FP32 matrix multiplication throughput (PyTorch or CPU fallback)
    @allure.feature("Intel GPU Accelerator")
    @allure.story("FP32 Matrix Multiplication")
    @pytest.mark.benchmark
    def test_fp32_gemm_throughput(self, setup_device, benchmark):
        device = setup_device
        if str(device) == "cpu":
            pytest.skip("Skipping PyTorch-specific benchmark on CPU fallback.")

        size = 4096
        a = torch.rand(size, size, device=device)
        b = torch.rand(size, size, device=device)

        def matmul():
            c = torch.matmul(a, b)
            torch.cuda.synchronize() 
            return c

        duration = benchmark(matmul).mean
        gflops = (2 * size**3) / (duration * 1e9)
        allure.attach(f"{gflops:.2f} GFLOPS", name="FP32 GEMM GFLOPS", attachment_type=allure.attachment_type.TEXT)
        assert gflops > 10.0, "FP32 GEMM performance is too low."

    # 2. Memory Bandwidth Benchmark (PyTorch or CPU fallback)
    @allure.feature("Intel GPU Accelerator")
    @allure.story("Device Memory Read/Write Bandwidth")
    @pytest.mark.benchmark
    def test_memory_bandwidth(self, setup_device, benchmark):
        device = setup_device
        if str(device) == "cpu":
            pytest.skip("Skipping PyTorch-specific benchmark on CPU fallback.")

        size = 256 * 1024 * 1024  # 256MB
        a = torch.rand(size, device=device, dtype=torch.float32)

        def copy_op():
            # Simulate a read and write operation
            b = a + 1 
            torch.cuda.synchronize()
            return b

        duration = benchmark(copy_op).mean
        # 3 read/write operations (a, 1, b) of 256MB each, total 768MB
        bandwidth = (3 * size * 4) / (duration * 1024**3) # GB/s (4 bytes per float)
        allure.attach(f"{bandwidth:.2f} GB/s", name="Memory Bandwidth", attachment_type=allure.attachment_type.TEXT)
        assert bandwidth > 5.0, "Memory bandwidth is too low."

    # 3. Host-to-Device Transfer Latency (PyTorch or CPU fallback)
    @allure.feature("Intel GPU Accelerator")
    @allure.story("Host-to-Device Latency")
    @pytest.mark.benchmark
    def test_h2d_latency(self, setup_device, benchmark):
        device = setup_device
        if str(device) == "cpu":
            pytest.skip("Skipping PyTorch-specific benchmark on CPU fallback.")

        size = 1024  # Small tensor for latency
        cpu_tensor = torch.rand(size)

        def h2d_transfer():
            gpu_tensor = cpu_tensor.to(device)
            torch.cuda.synchronize()
            return gpu_tensor

        duration = benchmark(h2d_transfer).mean
        allure.attach(f"{duration * 1e6:.2f} µs", name="H2D Latency (µs)")
        assert duration * 1e6 < 200.0, "Host-to-Device latency is too high (>200µs)."
    
    # 4. Kernel Launch Overhead (PyTorch or CPU fallback)
    @allure.feature("Intel GPU Accelerator")
    @allure.story("Kernel Launch Overhead")
    @pytest.mark.benchmark
    def test_kernel_launch_overhead(self, setup_device, benchmark):
        device = setup_device
        if str(device) == "cpu":
            pytest.skip("Skipping PyTorch-specific benchmark on CPU fallback.")

        x = torch.zeros(1, device=device)

        def launch():
            # Launch a simple element-wise kernel
            _ = x + 1
            torch.cuda.synchronize()
        
        duration = benchmark(launch).mean
        allure.attach(f"{duration * 1e6:.2f} µs", name="Kernel Launch Overhead (µs)")
        assert duration * 1e6 < 50.0, "Kernel launch overhead is too high (>50µs)."

    # 5. OpenCL-specific kernel execution (for non-oneAPI devices)
    @allure.feature("Intel GPU Accelerator")
    @allure.story("Raw OpenCL Kernel Execution")
    @pytest.mark.opencl
    def test_opencl_kernel_execution(self, setup_device, benchmark):
        if not cl:
            pytest.skip("pyopencl not installed or OpenCL not available.")

        # Try to find an Intel GPU platform
        platform = None
        for p in cl.get_platforms():
            if "Intel" in p.name:
                platform = p
                break
        
        if not platform:
            pytest.skip("No Intel OpenCL platform detected.")
        
        # Try to find a GPU device
        devices = platform.get_devices(cl.device_type.GPU)
        if not devices:
            pytest.skip("No OpenCL GPU device found on Intel platform.")

        device = devices[0]
        context = cl.Context([device])
        queue = cl.CommandQueue(context)
        
        allure.attach(f"Using OpenCL Device: {device.name}", name="OpenCL Device", attachment_type=allure.attachment_type.TEXT)

        # Simple element-wise addition kernel (A + B = C)
        kernel_code = """
        __kernel void sum(__global const float *a,
                          __global const float *b,
                          __global float *res)
        {
          int gid = get_global_id(0);
          res[gid] = a[gid] + b[gid];
        }
        """
        prg = cl.Program(context, kernel_code).build()
        
        size = 1000000 
        a_np = np.random.rand(size).astype(np.float32)
        b_np = np.random.rand(size).astype(np.float32)
        res_np = np.empty_like(a_np)

        mf = cl.mem_flags
        a_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
        b_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
        res_g = cl.Buffer(context, mf.WRITE_ONLY, res_np.nbytes)
        
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
        # Skip if device is 'cpu' and PyTorch XPU is not available, as this is a high-load test
        if str(device) == "cpu":
            pytest.skip("Skipping high-load PyTorch inference on CPU fallback.")
            
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

        duration = benchmark(inference_op).mean
        allure.attach(f"{duration:.3f}s", name="E2E Inference Duration")
        assert duration < 0.5, "End-to-end inference is too slow (>0.5s)."