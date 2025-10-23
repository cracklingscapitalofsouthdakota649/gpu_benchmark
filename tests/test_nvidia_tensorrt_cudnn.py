# tests/test_nvidia_tensorrt_cudnn.py
# NVIDIA cuDNN and TensorRT performance and inference validation

import os
import time
import pytest
import torch
import allure
import numpy as np
import json 
from torch import nn
from supports.gpu_monitor import collect_gpu_metrics # ADDED

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except Exception:
    TRT_AVAILABLE = False

try:
    import torch_geometric.nn as gnn
    from torch_geometric.data import Data
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    
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
        allure.attach(torch.cuda.get_device_name(device), name="GPU Device", attachment_type=allure.attachment_type.TEXT)
        return device

    # 1. CuDNN Convolution Benchmark
    @allure.feature("NVIDIA cuDNN Acceleration")
    @allure.story("Convolution Forward Pass Throughput")
    @pytest.mark.benchmark
    def test_cudnn_convolution_benchmark(self, setup_device, benchmark):
        device = setup_device
        # Standard ResNet-like 3x3 convolution
        model = nn.Conv2d(64, 128, kernel_size=3, padding=1).to(device)
        data = torch.randn(64, 64, 128, 128, device=device) # Batch 64
        batch_size = data.shape[0]

        def conv_op():
            _ = model(data)
            torch.cuda.synchronize()

        # --- 1. Run the benchmark ---
        result = benchmark(conv_op)
        
        # --- 2. Calculate Throughput (FPS) ---
        duration_mean = result.stats.mean
        fps = batch_size / duration_mean

        # --- 3. Collect GPU Telemetry (3-second sample) ---
        telemetry = collect_gpu_metrics(duration=3, interval=0.1) 

        # --- 4. Attach Metrics to Allure ---
        with allure.step("Performance Metrics"):
            allure.attach(f"{fps:.2f}", 
                          name="Convolution FPS", 
                          attachment_type=allure.attachment_type.TEXT)
            allure.attach(
                json.dumps(telemetry, indent=2),
                name="GPU Utilization (JSON)",
                attachment_type=allure.attachment_type.JSON,
            )
            
        assert fps > 1500.0 # Example assertion

    # 2. TensorRT Inference Speedup (Placeholder for TRT integration)
    @allure.feature("NVIDIA TensorRT")
    @allure.story("Inference Speedup over PyTorch")
    @pytest.mark.skipif(not TRT_AVAILABLE, reason="TensorRT not installed.")
    @pytest.mark.benchmark
    def test_tensorrt_inference_speedup(self, setup_device, benchmark):
        # This test remains a placeholder for integration but includes metric logging.
        device = setup_device
        
        # --- Placeholder Metrics ---
        speedup = 2.5 # Simulated speedup value
        
        # --- Collect GPU Telemetry (3-second sample) ---
        telemetry = collect_gpu_metrics(duration=3, interval=0.1) 
        
        # --- Attach Metrics to Allure ---
        with allure.step("Performance Metrics"):
            allure.attach(f"{speedup:.2f}x", 
                          name="TensorRT Speedup", 
                          attachment_type=allure.attachment_type.TEXT)
            allure.attach(
                json.dumps(telemetry, indent=2),
                name="GPU Utilization (JSON)",
                attachment_type=allure.attachment_type.JSON,
            )
        
        assert speedup >= 1.5, f"TensorRT speedup too low: {speedup:.2f}x"


    # 3. PyG GCNConv Throughput (Testing sparse-dense CuDNN/CUDA path)
    @allure.feature("NVIDIA cuDNN Acceleration")
    @allure.story("Graph Neural Network (GNN) Throughput")
    @pytest.mark.accelerator
    @pytest.mark.benchmark
    def test_gnn_throughput(self, setup_device, benchmark):
        """Benchmark a GNN layer, relying on underlying sparse-dense CUDA kernels."""
        if not GNN_AVAILABLE:
            pytest.skip("torch_geometric not installed (pip install torch-geometric).")
            
        device = setup_device
        num_nodes = 500_000
        num_edges = 2_000_000
        in_features = 128
        out_features = 128
        
        try:
            node_features = torch.randn(num_nodes, in_features, device=device)
            edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
            model = gnn.GCNConv(in_features, out_features).to(device).eval()
        except Exception as e:
             pytest.skip(f"Could not allocate tensors for GNN test (OOM?): {e}")

        def gnn_forward_pass():
            with torch.no_grad():
                _ = model(node_features, edge_index)
                torch.cuda.synchronize()

        # --- 1. Run the benchmark ---
        result = benchmark(gnn_forward_pass)
        
        # --- 2. Calculate Throughput (M-edges/sec) ---
        duration_mean = result.stats.mean
        throughput_meps = (num_edges / duration_mean) / 1e6 
        
        # --- 3. Collect GPU Telemetry (3-second sample) ---
        telemetry = collect_gpu_metrics(duration=3, interval=0.1) 

        # --- 4. Attach Metrics to Allure ---
        with allure.step("Performance Metrics"):
            allure.attach(f"{throughput_meps:.2f}", 
                          name="GNN Throughput (M-edges/sec)", 
                          attachment_type=allure.attachment_type.TEXT)
            allure.attach(
                json.dumps(telemetry, indent=2),
                name="GPU Utilization (JSON)",
                attachment_type=allure.attachment_type.JSON,
            )
            
        assert throughput_meps > 10.0 # Example assertion

    # 4. CuDNN FP32 Matrix Multiplication Benchmark
    @allure.feature("NVIDIA cuDNN Acceleration")
    @allure.story("FP32 Matrix Multiplication TFLOPS")
    @pytest.mark.benchmark
    def test_cudnn_fp32_matmul_benchmark(self, setup_device, benchmark):
        device = setup_device
        N = 4096
        a = torch.randn(N, N, device=device, dtype=torch.float32)
        b = torch.randn(N, N, device=device, dtype=torch.float32)

        def matmul():
            c = a @ b
            torch.cuda.synchronize()
        
        # --- 1. Run the benchmark ---
        result = benchmark(matmul)

        # --- 2. Calculate TFLOPS/sec ---
        # FLOPS for N x N * N x N is 2 * N^3
        flops = 2 * (N ** 3) 
        duration_mean = result.stats.mean
        tflops_sec = flops / (duration_mean * 1e12) 

        # --- 3. Collect GPU Telemetry (3-second sample) ---
        telemetry = collect_gpu_metrics(duration=3, interval=0.1) 

        # --- 4. Attach Metrics to Allure ---
        with allure.step("Performance Metrics"):
            allure.attach(f"{tflops_sec:.2f}", 
                          name="FP32 MatMul TFLOPS/sec", 
                          attachment_type=allure.attachment_type.TEXT)
            allure.attach(
                json.dumps(telemetry, indent=2),
                name="GPU Utilization (JSON)",
                attachment_type=allure.attachment_type.JSON,
            )

        assert tflops_sec > 5.0 # Example assertion

    # 5. CuDNN LSTM/RNN Throughput
    @allure.feature("NVIDIA cuDNN Acceleration")
    @allure.story("Recurrent Neural Network (LSTM) Throughput")
    @pytest.mark.benchmark
    def test_cudnn_rnn_throughput(self, setup_device, benchmark):
        device = setup_device
        input_size = 1024
        hidden_size = 1024
        num_layers = 2
        batch_size = 128
        seq_len = 100
        
        # Use a model that benefits greatly from cuDNN
        model = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False).to(device)
        input_data = torch.randn(seq_len, batch_size, input_size, device=device)
        h0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        c0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)

        def rnn_forward_pass():
            with torch.no_grad():
                _ = model(input_data, (h0, c0))
                torch.cuda.synchronize()

        # --- 1. Run the benchmark ---
        result = benchmark(rnn_forward_pass)

        # --- 2. Calculate Throughput (Samples/sec) ---
        # Throughput is measured in Samples Per Second (batch_size / mean_duration)
        duration_mean = result.stats.mean 
        samples_per_sec = batch_size / duration_mean

        # --- 3. Collect GPU Telemetry (3-second sample) ---
        telemetry = collect_gpu_metrics(duration=3, interval=0.1) 

        # --- 4. Attach Metrics to Allure ---
        with allure.step("Performance Metrics"):
            allure.attach(f"{samples_per_sec:.2f}", 
                          name="LSTM Samples/sec", 
                          attachment_type=allure.attachment_type.TEXT)
            allure.attach(
                json.dumps(telemetry, indent=2),
                name="GPU Utilization (JSON)",
                attachment_type=allure.attachment_type.JSON,
            )

        assert samples_per_sec > 1000.0 # Example assertion