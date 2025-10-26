"""
TensorFlow GPU Benchmark Suite
Comprehensive tests for NVIDIA GPU compute and memory performance using TensorFlow.
Each test is tagged with BDD-style Allure metadata.
"""

import pytest
import tensorflow as tf
import time
import numpy as np

# --- Allure compatibility handling ---
try:
    from allure_commons._allure import feature, story, severity
    from allure_commons.types import Severity
    severity_level = Severity  # alias for compatibility
except ImportError:
    from allure_commons._allure import feature, story, severity, severity_level

pytestmark = pytest.mark.tensorflow

# --- GPU Setup ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print(f"[WARN] Failed to set memory growth for {gpu}: {e}")
else:
    pytest.skip("No GPU detected; skipping TensorFlow GPU tests.", allow_module_level=True)


# --- Utility ---
def measure_bandwidth(bytes_transferred, seconds):
    """Convert bytes/sec to GB/s"""
    return (bytes_transferred / seconds) / (1024 ** 3)


@feature("TensorFlow GPU Benchmark")
@story("Matrix Multiplication Performance")
@severity(severity_level.CRITICAL)
@pytest.mark.io_accelerator
@pytest.mark.gpu
@pytest.mark.nvidia
def test_tensorflow_matmul_benchmark(allure_attach):
    """Benchmark large matrix multiplication on GPU."""
    size = 4096
    a = tf.random.uniform((size, size))
    b = tf.random.uniform((size, size))

    start = time.perf_counter()
    c = tf.matmul(a, b)
    _ = c.numpy()
    elapsed = time.perf_counter() - start

    allure_attach(f"Matrix multiplication of {size}x{size} took {elapsed:.2f}s")
    assert elapsed < 10.0, f"Matrix multiplication too slow: {elapsed:.2f}s"


@feature("TensorFlow GPU Benchmark")
@story("Conv2D Operation Performance")
@severity(severity_level.NORMAL)
@pytest.mark.io_accelerator
@pytest.mark.gpu
@pytest.mark.nvidia
def test_tensorflow_conv2d_benchmark(allure_attach):
    """Benchmark Conv2D operation on GPU."""
    input_shape = (32, 128, 128, 3)
    filters = 64
    kernel_size = (3, 3)

    x = tf.random.normal(input_shape)
    conv = tf.keras.layers.Conv2D(filters, kernel_size, activation='relu')

    start = time.perf_counter()
    y = conv(x)
    _ = y.numpy()
    elapsed = time.perf_counter() - start

    allure_attach(f"Conv2D with {filters} filters took {elapsed:.2f}s")
    assert elapsed < 5.0, f"Conv2D operation too slow: {elapsed:.2f}s"


@feature("TensorFlow GPU Benchmark")
@story("Mixed Precision Inference Speed")
@severity(severity_level.NORMAL)
@pytest.mark.io_accelerator
@pytest.mark.gpu
@pytest.mark.nvidia
def test_tensorflow_mixed_precision_inference(allure_attach):
    """Benchmark mixed precision inference."""
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    model = tf.keras.applications.MobileNetV2(weights=None, input_shape=(128, 128, 3))
    x = tf.random.normal((16, 128, 128, 3))

    start = time.perf_counter()
    _ = model(x)
    elapsed = time.perf_counter() - start

    allure_attach(f"Mixed precision inference (MobileNetV2) took {elapsed:.2f}s")
    assert elapsed < 8.0, f"Inference too slow: {elapsed:.2f}s"


@feature("TensorFlow GPU Benchmark")
@story("Random Number Generation Performance")
@severity(severity_level.MINOR)
@pytest.mark.io_accelerator
@pytest.mark.gpu
@pytest.mark.nvidia
def test_tensorflow_random_generation(allure_attach):
    """Benchmark random number generation."""
    start = time.perf_counter()
    data = tf.random.uniform((8192, 8192))
    _ = data.numpy()
    elapsed = time.perf_counter() - start

    allure_attach(f"Random number generation took {elapsed:.2f}s")
    assert elapsed < 4.0, f"Random generation too slow: {elapsed:.2f}s"


@feature("TensorFlow GPU Benchmark")
@story("GPU Memory Throughput (GB/s)")
@severity(severity_level.CRITICAL)
@pytest.mark.io_accelerator
@pytest.mark.gpu
@pytest.mark.nvidia
def test_tensorflow_memory_bandwidth(allure_attach):
    """Measure approximate GPU memory bandwidth."""
    size = 8192
    a = tf.random.uniform((size, size))
    b = tf.random.uniform((size, size))
    bytes_transferred = (a.numpy().nbytes + b.numpy().nbytes) * 2  # read + write

    start = time.perf_counter()
    c = a + b
    _ = c.numpy()
    elapsed = time.perf_counter() - start
    bandwidth = measure_bandwidth(bytes_transferred, elapsed)

    allure_attach(f"Memory bandwidth: {bandwidth:.2f} GB/s over {elapsed:.2f}s")
    assert bandwidth > 100, f"Low memory bandwidth detected: {bandwidth:.2f} GB/s"


@feature("TensorFlow GPU Benchmark")
@story("Multi-GPU Parallelism Performance")
@severity(severity_level.BLOCKER)
@pytest.mark.io_accelerator
@pytest.mark.gpu
@pytest.mark.nvidia
def test_tensorflow_multi_gpu_parallelism(allure_attach):
    """Benchmark multi-GPU parallel computation."""
    gpus = tf.config.list_logical_devices('GPU')
    if len(gpus) < 2:
        pytest.skip("Less than 2 GPUs available.")

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
        x = tf.random.normal((2048, 512))
        start = time.perf_counter()
        _ = model(x)
        elapsed = time.perf_counter() - start

    allure_attach(f"Multi-GPU forward pass took {elapsed:.2f}s on {len(gpus)} GPUs")
    assert elapsed < 5.0, f"Multi-GPU parallel computation too slow: {elapsed:.2f}s"
