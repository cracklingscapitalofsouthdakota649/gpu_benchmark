# tests/test_gpu_transformer.py

import torch
import torch.nn as nn
import pytest
import allure

class MiniTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, src):
        return self.encoder(src)

@allure.feature("GPU Deep Learning Workloads") # ⬅️ ADDED
@allure.story("Transformer Encoder Inference") # ⬅️ ADDED
@pytest.mark.gpu
def test_transformer_inference(benchmark):
    """Benchmark transformer encoder inference on GPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniTransformer().to(device)
    src = torch.rand(100, 32, 64, device=device)

    def run_inference():
        with torch.no_grad():
            _ = model(src)
            torch.cuda.synchronize() if torch.cuda.is_available() else None

    benchmark(run_inference)
