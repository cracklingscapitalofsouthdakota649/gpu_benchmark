# tests/test_multi_gpu.py
import pytest
import time
import psutil
import numpy as np
import allure

from scripts.plot_gpu_metrics import attach_chart_to_allure


# --------------------------------------------------------------------
# 1. CUDA Tensor Core Performance (PyTorch)
# --------------------------------------------------------------------
@allure.feature("CUDA Tensor Cores")
@allure.story("Matrix Multiply–Accumulate (MMA) TensorCore Benchmark")
@pytest.mark.gpu
@pytest.mark.accelerator
@pytest.mark.benchmark
def test_cuda_tensorcore_mma(benchmark):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TensorCore acceleration

    A = torch.randn(1024, 1024, dtype=torch.float16, device=device)
    B = torch.randn(1024, 1024, dtype=torch.float16, device=device)

    def run_tensorcore():
        start = time.time()
        for _ in range(20):
            torch.matmul(A, B)
        torch.cuda.synchronize()
        return time.time() - start

    duration = benchmark(run_tensorcore)
    allure.attach(f"TensorCore matrix multiply in {duration:.4f}s", name="TensorCore Duration", attachment_type=allure.attachment_type.TEXT)
    assert duration < 5


# --------------------------------------------------------------------
# 2. cuDNN Convolution Benchmark
# --------------------------------------------------------------------
@allure.feature("cuDNN")
@allure.story("Convolution Benchmark")
@pytest.mark.gpu
@pytest.mark.accelerator
@pytest.mark.benchmark
def test_cudnn_conv_benchmark(benchmark):
    import torch
    import torch.nn as nn

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    conv = nn.Conv2d(64, 128, kernel_size=3, padding=1).to(device)
    x = torch.randn(16, 64, 256, 256, device=device)

    def run_conv():
        torch.cuda.synchronize()
        start = time.time()
        y = conv(x)
        torch.cuda.synchronize()
        return time.time() - start

    duration = benchmark(run_conv)
    allure.attach(f"cuDNN convolution executed in {duration:.4f}s", name="cuDNN Convolution Time", attachment_type=allure.attachment_type.TEXT)
    assert duration < 3


# --------------------------------------------------------------------
# 3. PyTorch Automatic Mixed Precision (AMP)
# --------------------------------------------------------------------
@allure.feature("PyTorch AMP")
@allure.story("Automatic Mixed Precision Matrix Ops")
@pytest.mark.gpu
@pytest.mark.accelerator
@pytest.mark.benchmark
def test_amp_performance(benchmark):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    A = torch.randn(4096, 4096, device=device)
    B = torch.randn(4096, 4096, device=device)
    scaler = torch.cuda.amp.GradScaler()

    def run_amp():
        start = time.time()
        with torch.cuda.amp.autocast():
            C = torch.matmul(A, B)
        torch.cuda.synchronize()
        return time.time() - start

    duration = benchmark(run_amp)
    allure.attach(f"AMP matmul completed in {duration:.4f}s", name="AMP Time", attachment_type=allure.attachment_type.TEXT)
    assert duration < 6


# --------------------------------------------------------------------
# 4. RAPIDS cuML (GPU Machine Learning)
# --------------------------------------------------------------------
@allure.feature("RAPIDS cuML")
@allure.story("GPU Accelerated Linear Regression")
@pytest.mark.gpu
@pytest.mark.accelerator
@pytest.mark.benchmark
def test_cuml_linear_regression():
    try:
        from cuml.linear_model import LinearRegression
        import cudf
    except ImportError:
        pytest.skip("RAPIDS cuML not installed")

    n_samples = 50000
    X = cudf.DataFrame({"x1": np.random.rand(n_samples), "x2": np.random.rand(n_samples)})
    y = 3.5 * X["x1"] + 2.1 * X["x2"] + 0.5 + np.random.rand(n_samples)

    start = time.time()
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    elapsed = time.time() - start

    allure.attach(f"cuML Linear Regression fitted in {elapsed:.4f}s", name="RAPIDS cuML Time", attachment_type=allure.attachment_type.TEXT)
    assert elapsed < 2


# --------------------------------------------------------------------
# 5. CuPy Vectorized Computation (NumPy on GPU)
# --------------------------------------------------------------------
@allure.feature("Python GPU Libraries") 
@allure.story("CuPy Vectorized Operations") 
@pytest.mark.gpu
@pytest.mark.accelerator
@pytest.mark.benchmark
def test_cupy_vectorized_ops():
    try:
        import cupy as cp
    except ImportError:
        pytest.skip("CuPy not installed")

    n = 10_000_000
    a = cp.random.rand(n)
    b = cp.random.rand(n)

    start = time.time()
    c = cp.sin(a) * cp.sqrt(b)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - start

    allure.attach(f"CuPy vectorized ops took {elapsed:.4f}s", name="CuPy Vector Ops", attachment_type=allure.attachment_type.TEXT)
    assert elapsed < 1.0


# --------------------------------------------------------------------
# 6. TensorFlow Multi-GPU Training Simulation
# --------------------------------------------------------------------
@allure.feature("TensorFlow Distributed")
@allure.story("Multi-GPU Training Simulation")
@pytest.mark.gpu
@pytest.mark.accelerator
@pytest.mark.benchmark
def test_tensorflow_multi_gpu_train():
    try:
        import tensorflow as tf
    except ImportError:
        pytest.skip("TensorFlow not installed")

    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) < 2:
        pytest.skip("Need at least 2 GPUs for distributed test")

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        x = np.random.rand(1000, 1024)
        y = np.random.randint(0, 10, 1000)

        start = time.time()
        model.fit(x, y, epochs=1, batch_size=128, verbose=0)
        elapsed = time.time() - start

    allure.attach(f"TensorFlow multi-GPU training took {elapsed:.4f}s", name="TF Multi-GPU", attachment_type=allure.attachment_type.TEXT)
    assert elapsed < 10
