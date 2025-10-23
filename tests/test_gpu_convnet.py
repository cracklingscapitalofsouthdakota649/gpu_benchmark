# tests/test_gpu_convnet.py

import torch
import torch.nn as nn
import pytest
import allure # Import allure is generally good practice when using decorators

@allure.feature("GPU Deep Learning Workloads") # ⬅️ ADDED
@allure.story("Convolutional Network Forward Pass") # ⬅️ ADDED
@pytest.mark.gpu
def test_convnet_forward_pass(benchmark):
    """Benchmark simple CNN forward pass."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1)
    ).to(device)

    data = torch.randn(32, 3, 224, 224, device=device)

    def forward():
        with torch.no_grad():
            _ = model(data)
            torch.cuda.synchronize() if torch.cuda.is_available() else None

    # FAILED LINE ⬇️ (Implicitly returns None, which external tooling tries to access)
    benchmark(forward)