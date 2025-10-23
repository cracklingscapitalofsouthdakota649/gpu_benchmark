# tests/test_nvidia_real_gpu.py
# Real-world NVIDIA GPU compute and memory benchmarks

import os
import time
import pytest
import torch
import allure
import numpy as np
import json # <-- ADDED
from supports.gpu_monitor import collect_gpu_metrics # <-- ADDED

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler # Added from second file


def detect_nvidia_gpu():
    """Check for NVIDIA GPU availability."""
    try:
        # Check for CUDA availability and confirm it's an NVIDIA device
        return torch.cuda.is_available() and "nvidia" in torch.cuda.get_device_name(0).lower()
    except Exception:
        return False


@pytest.mark.nvidia
@pytest.mark.gpu
class TestRealNvidiaGPU:
    @pytest.fixture(scope="class", autouse=True)
    def setup_device(self):
        """Skip test if no NVIDIA GPU is detected, otherwise set up the device."""
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
        # Use large matrices for maximum utilization
        a = torch.randn(8192, 8192, device=device, dtype=torch.float16)
        b = torch.randn(8192, 8192, device=device, dtype=torch.float16)

        def matmul_op():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            return time.perf_counter() - start

        duration = benchmark(matmul_op)
        # Calculate TFLOPs: 2 * N^3 / time
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
        batch_size = 128
        data = torch.randn((batch_size, 3, 224, 224), device=device)

        def forward_pass():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(data)
            torch.cuda.synchronize()
            return time.perf_counter() - start

        duration = benchmark(forward_pass)
        samples_per_sec = batch_size / duration
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
            # Allocate 8 large tensors
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
        # 4GB tensor (1024^3 elements * 4 bytes/float)
        data = torch.randn((1024, 1024, 1024), device=device) 

        def transfer_op():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = data.cpu() # Transfer from GPU to CPU
            torch.cuda.synchronize()
            return time.perf_counter() - start

        duration = benchmark(transfer_op)
        size_gb = data.numel() * 4 / (1024**3)
        bandwidth = size_gb / duration
        allure.attach(f"{bandwidth:.2f} GB/s", name="PCIe Transfer Bandwidth")
        assert bandwidth > 8, f"Low PCIe bandwidth: {bandwidth:.2f} GB/s"

    # ───────────────────────────────────────────────────────────────
    # 6️. DataLoader to GPU Throughput (Pinned Memory) - Unique from File 1
    # ───────────────────────────────────────────────────────────────
    @allure.feature("NVIDIA GPU")
    @allure.story("DataLoader to GPU Transfer")
    @pytest.mark.accelerator
    def test_dataloader_to_gpu_throughput(self, setup_device, benchmark):
        """Benchmark data transfer from a parallel DataLoader to GPU."""
        device = setup_device
        # Create a 2GB dummy dataset (500 samples, ~4MB/sample)
        num_samples = 500
        data = torch.randn(num_samples, 3, 1024, 1024) 
        labels = torch.randn(num_samples, 1)
        dataset = TensorDataset(data, labels)
        
        # Use pin_memory=True for faster CPU->GPU async transfers
        loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)
        
        def transfer_loop():
            total_bytes = 0
            for x, y in loader:
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                # non_blocking=True works with pin_memory
                x = x.to(device, non_blocking=True) 
                y = y.to(device, non_blocking=True)
                
                torch.cuda.synchronize()
                duration = time.perf_counter() - start
                
                # Track bytes transferred per loop iteration
                total_bytes += (x.nelement() * x.element_size()) + (y.nelement() * y.element_size())
            
            # This benchmark variant returns the computed throughput
            return total_bytes
        
        # We benchmark the entire epoch loop
        total_bytes_transferred = benchmark(transfer_loop)
        total_duration = benchmark.stats.last_duration
        
        bandwidth_gbps = (total_bytes_transferred / total_duration) / (1024**3)
        
        allure.attach(f"{bandwidth_gbps:.2f} GB/s", name="DataLoader to GPU Bandwidth")
        assert bandwidth_gbps > 10.0, f"Low DataLoader transfer speed: {bandwidth_gbps:.2f} GB/s"
        
    # ───────────────────────────────────────────────────────────────
    # 7️. Mixed Precision (AMP) Acceleration Validation - Unique from File 2
    # ───────────────────────────────────────────────────────────────
    @allure.feature("NVIDIA GPU")
    @allure.story("Automatic Mixed Precision (AMP) Speedup")
    @pytest.mark.accelerator
    def test_mixed_precision_speedup(self, setup_device, benchmark):
        """Compare FP32 vs AMP performance on a CNN forward pass."""
        device = setup_device
        # A deeper model to showcase AMP benefits
        model = nn.Sequential(
            nn.Conv2d(3, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 1000)
        ).to(device)
        model.eval()

        data = torch.randn((64, 3, 224, 224), device=device)

        def forward_fp32():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(data)
            torch.cuda.synchronize()
            return time.perf_counter() - start

        def forward_amp():
            torch.cuda.synchronize()
            start = time.perf_counter()
            with autocast():
                _ = model(data) # Operations inside autocast will use FP16 where possible
            torch.cuda.synchronize()
            return time.perf_counter() - start

        fp32_time = benchmark(forward_fp32)
        amp_time = benchmark(forward_amp)
        speedup = fp32_time / amp_time

        allure.attach(
            f"FP32: {fp32_time:.4f}s\nAMP: {amp_time:.4f}s\nSpeedup: {speedup:.2f}×",
            name="AMP Speedup Report",
            attachment_type=allure.attachment_type.TEXT
        )

        assert speedup > 1.5, f"Expected AMP speedup ≥ 1.5×, got {speedup:.2f}×"

    # ───────────────────────────────────────────────────────────────
    # 8️. Sparse EmbeddingBag Lookup Throughput - Unique from File 1
    # ───────────────────────────────────────────────────────────────
    @allure.feature("NVIDIA GPU")
    @allure.story("Sparse EmbeddingBag Lookup")
    @pytest.mark.benchmark
    def test_sparse_embedding_lookup_speed(self, setup_device, benchmark):
        """Benchmark sparse embedding lookups, common in recommender systems."""
        device = setup_device
        num_embeddings = 1_000_000  # 1 Million embeddings
        embedding_dim = 128
        batch_size = 512
        lookups_per_sample = 20
        
        # Initialize a large embedding table on the GPU
        embedding_bag = nn.EmbeddingBag(
            num_embeddings, 
            embedding_dim, 
            mode='mean', 
            sparse=True  # Use sparse gradients
        ).to(device)
        
        # Create a batch of indices to look up.
        indices = torch.randint(low=0, high=num_embeddings, 
                                size=(batch_size, lookups_per_sample), 
                                device=device)
        
        def lookup_op():
            # This operation is heavily memory-bandwidth bound (gather)
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = embedding_bag(indices)
            torch.cuda.synchronize()
            return time.perf_counter() - start

        duration = benchmark(lookup_op)
        lookups_per_sec = (batch_size * lookups_per_sample) / duration
        
        allure.attach(f"{lookups_per_sec / 1e6:.2f} M-lookups/sec", name="EmbeddingBag Throughput")
        # This assertion is highly dependent on the GPU model
        assert lookups_per_sec > 10_000_000, f"Low embedding lookup throughput: {lookups_per_sec / 1e6:.2f} M-lookups/sec"
        
    # ───────────────────────────────────────────────────────────────
    # 9️. cuDNN-Accelerated RNN (GRU) Throughput - Unique from File 1
    # ───────────────────────────────────────────────────────────────
    @allure.feature("NVIDIA GPU")
    @allure.story("cuDNN RNN (GRU) Throughput")
    @pytest.mark.benchmark
    def test_rnn_gru_throughput(self, setup_device, benchmark):
        """Benchmark a multi-layer GRU, common in sequence modeling."""
        device = setup_device
        input_size = 256
        hidden_size = 1024
        num_layers = 4
        batch_size = 128
        seq_len = 100
        
        # Initialize a GRU layer (relies on cuDNN)
        model = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        ).to(device)
        model.eval() 
        
        # (Batch, Sequence, Features)
        data = torch.randn(batch_size, seq_len, input_size, device=device)
        
        # Hidden state (Layers, Batch, Hidden)
        h0 = torch.randn(num_layers, batch_size, hidden_size, device=device)
        
        def rnn_forward_pass():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(data, h0)
            torch.cuda.synchronize()
            return time.perf_counter() - start
            
        duration = benchmark(rnn_forward_pass)
        # Calculate throughput in total samples processed per second
        throughput = (batch_size * seq_len) / duration
        
        allure.attach(f"{throughput / 1e6:.2f} M-samples/sec", name="GRU Forward Throughput")
        assert throughput > 1_000_000, f"Low GRU throughput: {throughput / 1e6:.2f} M-samples/sec"        
        
    # ───────────────────────────────────────────────────────────────
    # 10. cuFFT Throughput (Fast Fourier Transform)
    # ───────────────────────────────────────────────────────────────
    @allure.feature("NVIDIA GPU")
    @allure.story("cuFFT (FFT) Throughput")
    @pytest.mark.benchmark
    def test_fft_throughput(self, setup_device, benchmark):
        """Benchmark cuFFT by performing a large 3D FFT and collecting telemetry."""
        device = setup_device
        # 512^3 complex tensor
        shape = (512, 512, 512)
        # ... (error handling)
            
        def fft_op():
            torch.cuda.synchronize()
            _ = torch.fft.fftn(data)
            torch.cuda.synchronize()

        # --- 1. Run the benchmark ---
        result = benchmark(fft_op)
        duration_mean = result.stats.mean
            
        # 2. Calculate throughput
        giga_elements = (data.numel()) / 1e9
        throughput_geps = giga_elements / duration_mean # Giga-elements processed per second
        
        # 3. Collect GPU Telemetry (3-second sample)
        telemetry = collect_gpu_metrics(duration=3, interval=0.1) 

        # 4. Attach Metrics to Allure
        with allure.step("Performance Metrics"):
            allure.attach(f"{throughput_geps:.2f}", 
                          name="FFT Throughput (GEPS)", 
                          attachment_type=allure.attachment_type.TEXT)
            allure.attach(
                json.dumps(telemetry, indent=2),
                name="GPU Utilization (JSON)",
                attachment_type=allure.attachment_type.JSON,
            )
            
        assert throughput_geps > 10.0, f"FFT throughput too low: {throughput_geps:.2f} GEPS"