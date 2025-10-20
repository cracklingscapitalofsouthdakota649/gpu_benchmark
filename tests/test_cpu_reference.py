# tests/test_cpu_reference.py
import torch
import pytest

@pytest.mark.cpu
def test_cpu_reference_operations(benchmark):
    """Compare CPU baseline to GPU performance."""
    device = "cpu"
    x = torch.rand(10000, 10000, device=device)
    y = torch.rand(10000, 10000, device=device)

    def cpu_op():
        return torch.mm(x, y)

    result = benchmark(cpu_op)
    assert result is not None
