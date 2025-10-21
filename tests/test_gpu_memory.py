# tests/test_gpu_memory.py
import torch
import pytest
import psutil
import allure # Import allure is generally good practice when using decorators

@allure.feature("GPU Resource Management")      # ⬅️ ADDED
@allure.story("Memory Allocation and Cleanup")  # ⬅️ ADDED
@pytest.mark.gpu
def test_gpu_memory_allocation():
    """Validate GPU memory allocation and cleanup."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    initial_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else psutil.virtual_memory().used
    x = torch.randn(2048, 2048, device=device)
    del x
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    final_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else psutil.virtual_memory().used
    assert final_mem <= initial_mem * 1.1