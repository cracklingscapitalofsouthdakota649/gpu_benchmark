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
        if rank == 0:
            print(f"Failed to init DDP: {e}")
        return

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    model = nn.Sequential(
        nn.Linear(1024, 4096), nn.ReLU(),
        nn.Linear(4096, 1024)
    ).to(device)
    model_ddp = DDP(model, device_ids=[rank])
    
    criterion = nn.MSELoss()
    optimizer = Adam(model_ddp.parameters())
    
    data = torch.randn(batch_size, 1024, device=device)
    target = torch.randn(batch_size, 1024, device=device)
    
    # Warmup
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        output = model_ddp(data)
        loss = criterion(output, target)
        loss.backward() # All-reduce gradients
        optimizer.step()
        
    torch.cuda.synchronize(device)
    dist.barrier()
    
    # Timed run
    start = time.perf_counter()
    
    # Run 10 steps
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        output = model_ddp(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
    torch.cuda.synchronize(device)
    dist.barrier()
    end = time.perf_counter()
    
    if rank == 0:
        duration_queue.put(end - start)
        
    dist.destroy_process_group()
    

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

    # ───────────────────────────────────────────────────────────────
    # 5️. Automatic Mixed Precision (AMP) Training Step
    # ───────────────────────────────────────────────────────────────
    @allure.feature("NVIDIA GPU Accelerator")
    @allure.story("AMP Training Step Throughput")
    @pytest.mark.accelerator
    def test_amp_training_step_speed(self, setup_device, benchmark):
        """Benchmark a full forward/backward/optimizer step using AMP."""
        device = setup_device
        model = nn.Sequential(
            nn.Linear(1024, 4096), nn.ReLU(),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, 1024)
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters())
        scaler = GradScaler()
        
        data = torch.randn(256, 1024, device=device)
        target = torch.randn(256, 1024, device=device)

        def amp_training_step():
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                output = model(data)
                loss = criterion(output, target)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            torch.cuda.synchronize()

        duration = benchmark(amp_training_step)
        steps_per_sec = 1.0 / duration
        allure.attach(f"{steps_per_sec:.2f} steps/sec", name="AMP Training Throughput")
        assert steps_per_sec > 100, f"Low training throughput: {steps_per_sec:.2f} steps/sec"

    # ───────────────────────────────────────────────────────────────
    # 6️. Multi-GPU Peer-to-Peer (P2P) Bandwidth
    # ───────────────────────────────────────────────────────────────
    @allure.feature("NVIDIA GPU Accelerator")
    @allure.story("Multi-GPU P2P Bandwidth (NVLink)")
    @pytest.mark.benchmark
    def test_multi_gpu_p2p_bandwidth(self, benchmark):
        """Measure P2P bandwidth between GPU 0 and GPU 1."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Test requires at least 2 NVIDIA GPUs.")
        
        dev0 = torch.device("cuda:0")
        dev1 = torch.device("cuda:1")
        
        if not torch.cuda.can_device_access_peer(0, 1):
            pytest.skip("Peer-to-peer memory access not supported between cuda:0 and cuda:1.")
            
        # 1GB tensor
        data_gb = 1
        tensor_size = (data_gb * 1024**3) // 4 
        a = torch.randn(tensor_size, device=dev0, dtype=torch.float32)
        b = torch.empty_like(a, device=dev1)

        def p2p_transfer():
            torch.cuda.synchronize(dev0)
            torch.cuda.synchronize(dev1)
            start = time.perf_counter()
            
            # Asynchronously copy from GPU 0 to GPU 1
            b.copy_(a, non_blocking=True)
            
            torch.cuda.synchronize(dev1)
            return time.perf_counter() - start

        duration = benchmark(p2p_transfer)
        bandwidth_gbps = data_gb / duration
        
        allure.attach(f"{bandwidth_gbps:.2f} GB/s", name="GPU 0->1 P2P Bandwidth")
        # A good NVLink connection should be > 20 GB/s. PCIe P2P is slower.
        assert bandwidth_gbps > 10, f"Low P2P bandwidth: {bandwidth_gbps:.2f} GB/s"
        
    # ───────────────────────────────────────────────────────────────
    # 7️. JIT Compiler Fusion Speedup
    # ───────────────────────────────────────────────────────────────
    @allure.feature("NVIDIA GPU Accelerator")
    @allure.story("JIT Fusion Speedup")
    @pytest.mark.benchmark
    def test_jit_fusion_speedup(self, setup_device, benchmark):
        """Compare a standard Python op vs. a TorchScript JIT-fused kernel."""
        device = setup_device
        x = torch.randn(1024, 4096, device=device)
        y = torch.randn(1024, 4096, device=device)
        z = torch.randn(1024, 4096, device=device)
        
        # Operation: (x * y) + z
        
        # 1. Standard (eager mode) execution
        # This launches two separate kernels (mul and add)
        def eager_op():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = (x * y) + z
            torch.cuda.synchronize()
            return time.perf_counter() - start

        # 2. JIT-scripted (fused) execution
        @torch.jit.script
        def fused_op(x, y, z):
            return (x * y) + z

        def jit_op():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = fused_op(x, y, z)
            torch.cuda.synchronize()
            return time.perf_counter() - start

        t_eager = benchmark(eager_op)
        t_jit = benchmark(jit_op)
        
        speedup = t_eager / t_jit if t_jit > 0 else 0
        allure.attach(f"Eager: {t_eager*1e6:.2f} µs | JIT (Fused): {t_jit*1e6:.2f} µs | Speedup: {speedup:.2f}x",
                      name="JIT Fusion Speedup")
        # Fused kernels should be significantly faster by reducing memory bandwidth
        assert speedup > 1.2, f"Expected >1.2x speedup from JIT fusion, got {speedup:.2f}x"
        
    # ───────────────────────────────────────────────────────────────
    # 8️. nn.DataParallel Training Step Overhead
    # ───────────────────────────────────────────────────────────────
    @allure.feature("NVIDIA GPU Accelerator")
    @allure.story("Multi-GPU DataParallel Throughput")
    @pytest.mark.stress
    def test_dataparallel_training_step(self, benchmark):
        """Benchmark a full training step using nn.DataParallel."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Test requires at least 2 NVIDIA GPUs.")
            
        device_ids = list(range(torch.cuda.device_count()))
        primary_device = torch.device("cuda:0")
        
        # A simple model to be replicated
        model = nn.Sequential(
            nn.Linear(1024, 2048), nn.ReLU(),
            nn.Linear(2048, 1024)
        )
        
        # Wrap model in DataParallel
        model_dp = nn.DataParallel(model, device_ids=device_ids).to(primary_device)
        criterion = nn.MSELoss()
        optimizer = SGD(model_dp.parameters(), lr=0.01)
        
        # Batch size should be divisible by number of GPUs
        batch_size = 64 * torch.cuda.device_count()
        data = torch.randn(batch_size, 1024, device=primary_device)
        target = torch.randn(batch_size, 1024, device=primary_device)

        def dp_training_step():
            torch.cuda.synchronize(primary_device)
            start = time.perf_counter()
            
            optimizer.zero_grad()
            output = model_dp(data) # Scatters data to all GPUs
            loss = criterion(output, target)
            loss.backward()         # Reduces gradients back to primary
            optimizer.step()
            
            torch.cuda.synchronize(primary_device)
            return time.perf_counter() - start

        duration = benchmark(dp_training_step)
        throughput = batch_size / duration
        
        allure.attach(f"{throughput:.2f} samples/sec", 
                      name=f"DataParallel Throughput ({torch.cuda.device_count()} GPUs)")
        assert throughput > 1000, f"Low DataParallel throughput: {throughput:.2f} samples/sec"
        
    # ───────────────────────────────────────────────────────────────
    # 9️. DistributedDataParallel (DDP) Training Step
    # ───────────────────────────────────────────────────────────────
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
                      name=f"DDP Throughput ({world_size} GPUs)")
        allure.attach(f"{avg_step_time*1000:.2f} ms/step", 
                      name="Average Step Time")
        
        assert throughput > 1000, f"Low DDP throughput: {throughput:.2f} samples/sec"        