# tests/test_gpu_mixed_precision.py
import torch
import pytest

@pytest.mark.gpu
def test_mixed_precision_matmul(benchmark):
    """Benchmark matrix multiplication using mixed precision."""
    if not torch.cuda.is_available():
        pytest.skip("Mixed precision requires CUDA.")
    device = "cuda"
    x = torch.randn(2048, 2048, device=device, dtype=torch.float16)
    y = torch.randn(2048, 2048, device=device, dtype=torch.float16)

    def matmul():
        with torch.cuda.amp.autocast():
            z = x @ y
            torch.cuda.synchronize()

    benchmark(matmul)
