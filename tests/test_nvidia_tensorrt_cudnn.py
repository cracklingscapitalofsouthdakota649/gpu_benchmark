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

    # ───────────────────────────────────────────────────────────────
    # 4️. CUDA Graph Latency Speedup
    # ───────────────────────────────────────────────────────────────
    @allure.feature("NVIDIA CUDA Graphs")
    @allure.story("Kernel Launch Overhead Reduction")
    @pytest.mark.accelerator
    def test_cuda_graph_latency_speedup(self, setup_device, benchmark):
        """Compare looped inference latency vs. CUDA Graph 'replay' latency."""
        device = setup_device
        model = nn.Sequential(
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 512)
        ).to(device)
        model.eval()
        data = torch.randn(128, 512, device=device)
        
        # 1. Baseline: Run in a standard Python loop (incurs launch overhead)
        def baseline_loop():
            for _ in range(100):
                _ = model(data)
            torch.cuda.synchronize()
        
        t_baseline = benchmark(baseline_loop)

        # 2. CUDA Graph: Capture the graph once
        graph = torch.cuda.CUDAGraph()
        # Warmup
        for _ in range(5):
            _ = model(data)
        
        # Capture
        with torch.cuda.graph(graph):
            static_output = model(data)
        
        # 3. Replay: Run the captured graph (minimal overhead)
        def graph_replay_loop():
            for _ in range(100):
                graph.replay()
            torch.cuda.synchronize()

        t_graph = benchmark(graph_replay_loop)
        
        speedup = t_baseline / t_graph if t_graph > 0 else 0
        allure.attach(f"Baseline: {t_baseline*1000:.2f}ms | Graph: {t_graph*1000:.2f}ms | Speedup: {speedup:.2f}x",
                      name="CUDA Graph Speedup (100 iterations)")
        assert speedup > 2.0, f"Expected >2.0x speedup from CUDA Graphs, got {speedup:.2f}x"
        
    # ───────────────────────────────────────────────────────────────
    # 5️. cuDNN BatchNorm Throughput
    # ───────────────────────────────────────────────────────────────
    @allure.feature("NVIDIA cuDNN")
    @allure.story("BatchNorm Throughput Test")
    @pytest.mark.accelerator
    def test_cudnn_batchnorm_throughput(self, setup_device, benchmark):
        """Benchmark cuDNN-accelerated BatchNorm2d forward/backward pass."""
        device = setup_device
        num_features = 256
        batch_size = 128
        img_size = 112
        
        # NCHW format
        data = torch.randn(batch_size, num_features, img_size, img_size, 
                           device=device, requires_grad=True)
        bn_layer = nn.BatchNorm2d(num_features).to(device)
        
        def forward_backward_pass():
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            # Forward pass
            output = bn_layer(data)
            
            # Backward pass
            grad_output = torch.randn_like(output)
            output.backward(grad_output, retain_graph=True)
            
            torch.cuda.synchronize()
            return time.perf_counter() - start

        duration = benchmark(forward_backward_pass)
        # Calculate in Giga-elements per second (processed twice: fwd + bwd)
        total_elements = data.numel() * 2
        throughput_geps = (total_elements / duration) / 1e9
        
        allure.attach(f"{throughput_geps:.2f} Giga-elements/sec", name="BatchNorm Fwd/Bwd Throughput")
        assert throughput_geps > 50, f"Low BatchNorm throughput: {throughput_geps:.2f} Giga-elements/sec"
        
    # ───────────────────────────────────────────────────────────────
    # 6️. torch.compile (Dynamo) Inference Speedup
    # ───────────────────────────────────────────────────────────────
    @allure.feature("NVIDIA torch.compile")
    @allure.story("torch.compile vs Eager Speedup")
    @pytest.mark.accelerator
    def test_torch_compile_inference_speedup(self, setup_device, benchmark):
        """Compare eager-mode inference vs. torch.compile() (Dynamo)."""
        device = setup_device
        
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(64, 128, 3, padding=1)
                self.relu = nn.ReLU()
                self.bn = nn.BatchNorm2d(128)
            
            def forward(self, x):
                x = self.conv(x)
                x = self.relu(x)
                x = self.bn(x)
                return x

        model = SimpleNet().to(device).eval()
        
        # 1. Compile the model
        # mode="max-autotune" takes longer but gives best performance
        try:
            compiled_model = torch.compile(model, mode="max-autotune")
        except Exception as e:
            pytest.skip(f"torch.compile() failed, likely unsupported on this platform: {e}")
            
        data = torch.randn(64, 64, 56, 56, device=device)

        # 2. Benchmark eager (standard) model
        def eager_infer():
            with torch.no_grad():
                torch.cuda.synchronize()
                start = time.perf_core()
                _ = model(data)
                torch.cuda.synchronize()
                return time.perf_counter() - start
        
        t_eager = benchmark(eager_infer)

        # 3. Benchmark compiled model
        def compiled_infer():
            with torch.no_grad():
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = compiled_model(data)
                torch.cuda.synchronize()
                return time.perf_counter() - start
        
        t_compiled = benchmark(compiled_infer)
        
        speedup = t_eager / t_compiled if t_compiled > 0 else 0
        allure.attach(f"Eager: {t_eager*1e6:.2f} µs | Compiled: {t_compiled*1e6:.2f} µs | Speedup: {speedup:.2f}x",
                      name="torch.compile (Dynamo) Speedup")
        assert speedup > 1.1, f"Expected >1.1x speedup from torch.compile, got {speedup:.2f}x"   

    # ───────────────────────────────────────────────────────────────
    # 7️. Graph Neural Network (GNN) Inference
    # ───────────────────────────────────────────────────────────────
    @allure.feature("NVIDIA GNN")
    @allure.story("GNN (GCN) Inference Throughput")
    @pytest.mark.accelerator
    def test_gnn_inference_throughput(self, setup_device, benchmark):
        """Benchmark a GCN layer forward pass, testing sparse gather ops."""
        if not GNN_AVAILABLE:
            pytest.skip("torch_geometric not installed (pip install torch-geometric).")
            
        device = setup_device
        num_nodes = 500_000
        num_edges = 2_000_000
        in_features = 128
        out_features = 128
        
        try:
            # Create a large random graph and move it to GPU
            node_features = torch.randn(num_nodes, in_features, device=device)
            edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
            
            # Create a GCNConv layer
            model = gnn.GCNConv(in_features, out_features).to(device).eval()
        except Exception as e:
             pytest.skip(f"Could not allocate tensors for GNN test (OOM?): {e}")

        def gnn_forward_pass():
            with torch.no_grad():
                torch.cuda.synchronize()
                start = time.perf_counter()
                # This performs sparse-dense matrix multiplication (SpMM)
                _ = model(node_features, edge_index)
                torch.cuda.synchronize()
                return time.perf_counter() - start
        
        duration = benchmark(gnn_forward_pass)
        # Throughput in M-edges/sec
        throughput_meps = (num_edges / duration) / 1e6
        
        allure.attach(f"{throughput_meps:.2f} M-edges/sec", name="GCNConv Inference Throughput")
        assert throughput_meps > 100, f"Low GNN throughput: {throughput_meps:.2f} M-edges/sec"        