# tests/test_gpu_memory.py
import torch
import pytest
import psutil
import allure
import json # <-- ADDED
from supports.gpu_monitor import collect_gpu_metrics # <-- ADDED

@allure.feature("GPU Resource Management")
@allure.story("Memory Allocation and Cleanup")
@pytest.mark.gpu
def test_gpu_memory_allocation():
    """Validate GPU memory allocation, cleanup, and capture memory usage."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 1. Telemetry Snapshot: Before Allocation ---
    telemetry_before = collect_gpu_metrics(duration=1, interval=0.2)
    
    initial_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else psutil.virtual_memory().used
    
    x = torch.randn(2048, 2048, device=device)
    
    # --- 2. Current Usage and Snapshot: After Allocation ---
    current_mem_used = torch.cuda.memory_allocated() if torch.cuda.is_available() else psutil.virtual_memory().used
    telemetry_during = collect_gpu_metrics(duration=1, interval=0.2)
    
    with allure.step("VRAM Usage"):
        allure.attach(f"{current_mem_used / (1024**2):.2f} MB", 
                      name="VRAM Used After Allocation", 
                      attachment_type=allure.attachment_type.TEXT)
        
    del x
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    final_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else psutil.virtual_memory().used
    
    # --- 3. Telemetry Snapshot: After Cleanup ---
    telemetry_after = collect_gpu_metrics(duration=1, interval=0.2)
    
    with allure.step("Utilization Snapshots"):
        allure.attach(json.dumps(telemetry_before, indent=2), name="GPU/CPU Before Allocation", attachment_type=allure.attachment_type.JSON)
        allure.attach(json.dumps(telemetry_during, indent=2), name="GPU/CPU During Allocation", attachment_type=allure.attachment_type.JSON)
        allure.attach(json.dumps(telemetry_after, indent=2), name="GPU/CPU After Cleanup", attachment_type=allure.attachment_type.JSON)

    assert final_mem <= initial_mem * 1.1