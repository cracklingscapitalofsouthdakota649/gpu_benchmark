# tests/test_gpu_matrix_mul.py
import torch
import pytest

@pytest.mark.gpu
def test_matrix_multiplication(benchmark):
    """Benchmark dense matrix multiplication."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.rand(4096, 4096, device=device)
    b = torch.rand(4096, 4096, device=device)

    def multiply():
        c = torch.matmul(a, b)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        return c

    result = benchmark(multiply)
    assert result is not None
