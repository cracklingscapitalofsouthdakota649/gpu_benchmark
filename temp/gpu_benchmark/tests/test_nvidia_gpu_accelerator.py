# tests/test_nvidia_gpu_accelerator.py
# NVIDIA GPU validation, kernel performance, and tensor core benchmarks

import os
import time
import torch
import pytest
import allure
import numpy as np
from torch import nn
from torch.optim import SGD
import json
from supports.gpu_monitor import collect_gpu_metrics

# Benchmarks a full training step using DistributedDataParallel (DDP). This is the modern standard for multi-GPU training and heavily tests 
# the NCCL library for high-speed all-reduce gradient synchronization.
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam

# ───────────────────────────────────────────────────────────────
# Detection utilities
# ───────────────────────────────────────────────────────────────

def detect_nvidia_gpu():
    """Return True if an NVIDIA GPU with CUDA support is available."""
    try:
        return torch.cuda.is_available() and torch.cuda.get_device_name(0).lower().startswith("nvidia")
    except Exception:
        return False


# Helper function to be spawned for DDP benchmark
def _ddp_benchmark_func(rank, world_size, duration_queue, batch_size):
    """Internal function to be spawned for DDP benchmark."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # Use a free port
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    except Exception as e:
        print(f"Error initializing DDP: {e}")
        return

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    # Simple model and data
    model = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64)).to(device)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = Adam(ddp_model.parameters())
    criterion = nn.MSELoss()
    
    # Fake dataset
    dummy_data = torch.randn(batch_size, 256, device=device)
    dummy_target = torch.randn(batch_size, 64, device=device)
    
    # Warmup
    for _ in range(5):
        output = ddp_model(dummy_data)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    # Benchmark loop (10 steps)
    start_time = time.perf_counter()
    for _ in range(10):
        output = ddp_model(dummy_data)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    duration = time.perf_counter() - start_time
    
    if rank == 0:
        duration_queue.put(duration)

    dist.destroy_process_group()

# ───────────────────────────────────────────────────────────────
# Test Class
# ───────────────────────────────────────────────────────────────

@pytest.mark.gpu
@pytest.mark.nvidia
class TestNvidiaGPUAccelerator:
    @pytest.fixture(scope="class", autouse=True)
    def setup_device(self):
        if not detect_nvidia_gpu():
            pytest.skip("No NVIDIA GPU detected.")
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        allure.attach(torch.cuda.get_device_name(0), name="NVIDIA GPU Device", attachment_type=allure.attachment_type.TEXT)
        return device

    # 1. Device check
    @allure.feature("NVIDIA GPU Accelerator") 
    @allure.story("Device Detection and Info") 
    @pytest.mark.validation
    def test_device_detection(self, setup_device):
        """Validate CUDA/NVIDIA detection and info logging."""
        assert torch.cuda.is_available()
        assert torch.cuda.get_device_name(0).lower().startswith("nvidia")
        
        # Log device properties
        properties = torch.cuda.get_device_properties(0)
        info = {
            "Total Memory (GB)": round(properties.total_memory / (1024**3), 2),
            "Major/Minor Compute Capability": f"{properties.major}.{properties.minor}",
            "Number of Multiprocessors": properties.multi_processor_count,
            "Max Threads/Multiprocessor": properties.max_threads_per_multiprocessor
        }
        allure.attach(json.dumps(info, indent=2), name="GPU Properties", attachment_type=allure.attachment_type.JSON)
        
    # 2. CUDA Kernel Launch Latency
    @allure.feature("NVIDIA GPU Accelerator") 
    @allure.story("Kernel Launch Latency") 
    @pytest.mark.latency
    def test_kernel_launch_latency(self, setup_device, benchmark):
        """Benchmark the overhead of launching a minimal CUDA kernel."""
        device = setup_device
        # A simple CUDA operation (noop)
        x = torch.zeros(1, device=device)
        def launch_kernel():
            _ = x + 1
            torch.cuda.synchronize()
            
        # The result is mean duration of one operation
        duration = benchmark(launch_kernel).mean
        allure.attach(f"{duration * 1e6:.3f} µs", name="Kernel Launch Latency")
        assert duration * 1e6 < 100 # Expect latency < 100 microseconds (depends on driver and OS)

    # 3. Memory Bandwidth (Host to Device)
    @allure.feature("NVIDIA GPU Accelerator") 
    @allure.story("Host-to-Device Memory Bandwidth") 
    @pytest.mark.bandwidth
    def test_h2d_bandwidth(self, setup_device, benchmark):
        """Benchmark Host-to-Device (CPU to GPU) memory copy bandwidth."""
        device = setup_device
        size_bytes = 256 * 1024 * 1024 # 256 MB
        host_tensor = torch.randn(size_bytes // 4, dtype=torch.float32) # Float32 is 4 bytes
        
        def h2d_copy():
            _ = host_tensor.to(device, non_blocking=True)
            torch.cuda.synchronize()

        duration = benchmark(h2d_copy).mean
        bandwidth = (size_bytes / duration) / (1024**3) # GB/s
        allure.attach(f"{bandwidth:.2f} GB/s", name="H2D Bandwidth")
        assert bandwidth > 5, "H2D bandwidth is too low."

    # 4. Memory Bandwidth (Device to Device)
    @allure.feature("NVIDIA GPU Accelerator") 
    @allure.story("Device-to-Device Memory Bandwidth") 
    @pytest.mark.bandwidth
    def test_d2d_bandwidth(self, setup_device, benchmark):
        """Benchmark Device-to-Device (GPU to GPU) memory copy bandwidth."""
        device = setup_device
        size_bytes = 512 * 1024 * 1024 # 512 MB
        x = torch.randn(size_bytes // 4, dtype=torch.float32, device=device)
        y = torch.empty_like(x)
        
        def d2d_copy():
            y.copy_(x)
            torch.cuda.synchronize()
            
        duration = benchmark(d2d_copy).mean
        bandwidth = (size_bytes / duration) / (1024**3) # GB/s
        allure.attach(f"{bandwidth:.2f} GB/s", name="D2D Bandwidth")
        assert bandwidth > 100, "D2D bandwidth is too low." # Expect very high speed

    # 5. FP32 Matrix Multiplication Throughput (non-Tensor Core)
    @allure.feature("NVIDIA GPU Accelerator") 
    @allure.story("FP32 Matrix Multiplication Throughput") 
    @pytest.mark.benchmark
    def test_fp32_gemm_throughput(self, setup_device, benchmark):
        """Benchmark FP32 GEMM performance (non-Tensor Core)."""
        device = setup_device
        size = 4096
        a = torch.randn(size, size, device=device, dtype=torch.float32)
        b = torch.randn(size, size, device=device, dtype=torch.float32)
        
        flops = 2 * size**3 # ~137 billion FLOPS

        def matmul():
            _ = a @ b
            torch.cuda.synchronize()

        duration = benchmark(matmul).mean
        gflops = (flops / duration) / 1e9
        allure.attach(f"{gflops:.2f} GFLOPS", name="FP32 GFLOPS")
        assert gflops > 1000 # Expect at least 1 TFLOPS (1000 GFLOPS) on modern cards.

    # 6. FP16 Tensor Core Throughput
    @allure.feature("NVIDIA GPU Accelerator") 
    @allure.story("FP16 Tensor Core Throughput") 
    @pytest.mark.benchmark
    @pytest.mark.tensorcore
    def test_fp16_tensor_core_throughput(self, setup_device, benchmark):
        """Benchmark FP16 Tensor Core performance (requires recent NVIDIA GPU)."""
        device = setup_device
        if not torch.cuda.get_device_properties(0).major >= 7:
             pytest.skip("Test requires Volta (SM 7.0) or newer GPU for Tensor Cores.")

        size = 4096
        a = torch.randn(size, size, device=device, dtype=torch.float16)
        b = torch.randn(size, size, device=device, dtype=torch.float16)
        
        flops = 2 * size**3 # FLOPs approximation for comparison

        def matmul_fp16():
            _ = a @ b
            torch.cuda.synchronize()

        duration = benchmark(matmul_fp16).mean
        # A simple GFLOPS approximation using the FP32 calculation for comparison
        gflops_approx = (flops / duration) / 1e9
        allure.attach(f"{gflops_approx:.2f} GFLOPS (Approx)", name="FP16 Tensor Core GFLOPS")
        assert gflops_approx > 5000 # Expect significantly higher throughput than FP32.

    # 7. Multi-GPU P2P Bandwidth (if multiple GPUs available)
    @allure.feature("NVIDIA GPU Accelerator") 
    @allure.story("Multi-GPU Peer-to-Peer Bandwidth") 
    @pytest.mark.multigpu
    def test_multi_gpu_p2p_bandwidth(self, setup_device, benchmark):
        """Benchmark P2P memory copy between two GPUs (requires NVLink/PCIe link)."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Test requires at least 2 NVIDIA GPUs.")
            
        # Check P2P capability
        if not torch.cuda.can_device_access_peer(0, 1):
            pytest.skip("P2P access disabled between GPU 0 and GPU 1.")
            
        size_bytes = 256 * 1024 * 1024 # 256 MB
        x = torch.randn(size_bytes // 4, dtype=torch.float32, device='cuda:0')
        y = torch.empty_like(x, device='cuda:1')
        
        def p2p_copy():
            y.copy_(x, non_blocking=True)
            torch.cuda.synchronize()
            
        duration = benchmark(p2p_copy).mean
        bandwidth = (size_bytes / duration) / (1024**3) # GB/s
        allure.attach(f"{bandwidth:.2f} GB/s", name="P2P Bandwidth (GPU 0 -> GPU 1)")
        assert bandwidth > 10, "P2P bandwidth is too low."

    # 8. Multi-GPU All-Reduce Latency (e.g., NCCL)
    @allure.feature("NVIDIA GPU Accelerator") 
    @allure.story("Multi-GPU All-Reduce Latency") 
    @pytest.mark.multigpu
    def test_multi_gpu_all_reduce_latency(self):
        """Benchmark the latency of NCCL all-reduce operation."""
        world_size = torch.cuda.device_count()
        if world_size < 2:
            pytest.skip("Test requires at least 2 NVIDIA GPUs.")

        # This complex test uses multi-processing to safely run a distributed benchmark
        def _all_reduce_wrapper(rank, world_size, queue):
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12356'
            try:
                dist.init_process_group("nccl", rank=rank, world_size=world_size)
            except:
                queue.put(float('inf'))
                return

            torch.cuda.set_device(rank)
            device = torch.device("cuda", rank)
            
            tensor_size = 1 * 1024 * 1024 # 1MB
            tensor = torch.ones(tensor_size, device=device)
            
            # Warmup
            for _ in range(5):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()

            # Benchmark
            start_time = time.perf_counter()
            for _ in range(100):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            duration = time.perf_counter() - start_time
            
            if rank == 0:
                queue.put(duration / 100) # Average latency

            dist.destroy_process_group()

        mp_context = mp.get_context('spawn')
        latency_queue = mp_context.Queue()
        
        mp.spawn(
            _all_reduce_wrapper,
            args=(world_size, latency_queue),
            nprocs=world_size,
            join=True
        )
        
        latency = latency_queue.get()
        allure.attach(f"{latency * 1e6:.3f} µs", name="All-Reduce Latency per step")
        assert latency * 1e6 < 200 # Expect low latency for a small tensor all-reduce

    # 9. End-to-end model inference throughput (vision classification)
    @allure.feature("NVIDIA GPU Accelerator") 
    @allure.story("End-to-end Inference Throughput") 
    @pytest.mark.benchmark
    def test_end_to_end_inference_throughput(self, setup_device, benchmark):
        """Benchmark end-to-end inference throughput for a vision model."""
        device = setup_device
        # Simple CNN model
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
            with torch.no_grad():
                _ = model(batch)
                torch.cuda.synchronize()
        
        duration = benchmark(inference_op).mean
        # Throughput in samples/sec
        throughput = 128 / duration
        allure.attach(f"{throughput:.2f} samples/sec", name="Inference Throughput")
        assert throughput > 1000 # Expect high throughput

    # 10. DistributedDataParallel Training Step
    @allure.feature("NVIDIA GPU Accelerator")
    @allure.story("Multi-GPU DDP (NCCL) Throughput")
    @pytest.mark.stress
    def test_distributeddataparallel_step(self):
        """Benchmark a DDP training step, testing NCCL all-reduce."""
        world_size = torch.cuda.device_count()
        if world_size < 2:
            pytest.skip("Test requires at least 2 NVIDIA GPUs.")
            
        # DDP requires spawning processes
        mp_context = mp.get_context('spawn')
        duration_queue = mp_context.Queue()
        
        # Batch size per GPU
        batch_size_per_gpu = 128
        total_batch_size = batch_size_per_gpu * world_size
        
        mp.spawn(
            _ddp_benchmark_func,
            args=(world_size, duration_queue, batch_size_per_gpu),
            nprocs=world_size,
            join=True
        )
        
        # Check if benchmark ran
        if duration_queue.empty():
            pytest.fail("DDP benchmark function failed to execute.")

        duration = duration_queue.get()
        
        # We ran 10 steps
        avg_step_time = duration / 10
        throughput = total_batch_size / avg_step_time
        
        allure.attach(f"{throughput:.2f} samples/sec", 
                      name=f"DDP Throughput (Batch Size: {total_batch_size})", 
                      attachment_type=allure.attachment_type.TEXT)
        assert avg_step_time < 0.5, f"DDP training step is too slow: {avg_step_time:.3f}s"
    
    # ───────────────────────────────────────────────────────────────
    # New Tests Start Here
    # ───────────────────────────────────────────────────────────────

    # 11. Concurrent Kernel Launch Stress Test (NEW TEST)
    @allure.feature("NVIDIA CUDA Parallelism")
    @allure.story("Concurrent Stream Execution")
    @pytest.mark.stress
    @pytest.mark.parallel
    def test_concurrent_kernel_streams(self, setup_device, benchmark):
        """Benchmark concurrent execution using multiple CUDA streams."""
        device = setup_device
        size = 1024
        num_streams = 8
        # Create separate streams
        streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
        
        # Create tensors for operations
        tensors = [torch.randn(size, size, device=device) for _ in range(num_streams)]
        results = [torch.empty(size, size, device=device) for _ in range(num_streams)]
        
        def concurrent_matmul():
            # Enqueue operations on different streams
            for i in range(num_streams):
                with torch.cuda.stream(streams[i]):
                    # Simple matrix multiplication
                    results[i].copy_(tensors[i] @ tensors[i])
            
            # Wait for all streams to finish
            torch.cuda.synchronize()

        duration = benchmark(concurrent_matmul).mean
        allure.attach(f"{duration * 1000:.3f} ms", name=f"Time for {num_streams} Concurrent Streams")
        
        # Simple assertion: The concurrent time should be relatively low and typically faster 
        # than the sequential sum of all operations if there is parallelism.
        assert duration < 0.1 

    # 12. Maximum VRAM Allocation Test (Stability) (NEW TEST)
    @allure.feature("NVIDIA VRAM Management")
    @allure.story("Maximum Usable Memory Allocation")
    @pytest.mark.memory
    @pytest.mark.stability
    def test_max_vram_allocation(self, setup_device):
        """Attempt to allocate the maximum available VRAM to test stability and fragmentation."""
        device = setup_device
        
        # Get total VRAM
        properties = torch.cuda.get_device_properties(0)
        total_memory = properties.total_memory # Bytes
        
        # Target 95% of total memory to leave some for OS/driver overhead
        target_allocation_bytes = int(total_memory * 0.95)
        # Tensor size (assuming Float32 = 4 bytes)
        target_elements = target_allocation_bytes // 4
        
        # Clear any existing small allocations
        torch.cuda.empty_cache()

        try:
            with allure.step(f"Attempting to allocate {target_allocation_bytes / (1024**3):.2f} GB"):
                # Allocate a tensor slightly below the total memory
                large_tensor = torch.empty(target_elements, device=device, dtype=torch.float32)
                
                # Ensure the allocation is successful
                allocated_mem = torch.cuda.memory_allocated()
                allure.attach(f"{allocated_mem / (1024**3):.2f} GB", name="Successfully Allocated VRAM")
                
                # Perform a simple operation to force memory access and kernel launch
                _ = large_tensor + 1
                torch.cuda.synchronize()
                
                # Cleanup
                del large_tensor
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                allure.attach(str(e), name="OOM Error Details", attachment_type=allure.attachment_type.TEXT)
                pytest.fail(f"OOM when allocating 95% of VRAM. Allocation is unstable or fragmented. Error: {e}")
            else:
                raise e # Re-raise other errors
        
        # Assert cleanup was successful (less than 10MB residual)
        final_mem = torch.cuda.memory_allocated()
        assert final_mem < 10 * 1024 * 1024, "VRAM cleanup failed after large allocation stress test."