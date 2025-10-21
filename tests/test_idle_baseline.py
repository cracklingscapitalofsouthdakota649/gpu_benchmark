# tests/test_idle_baseline.py
import time
import json
import allure
import pytest

from scripts.plot_gpu_metrics import attach_chart_to_allure
from scripts.system_metrics import SystemMetrics
from tests.device_utils import pick_device, get_device

@allure.feature("System Resource Monitoring") # ⬅️ ADDED
@allure.story("Idle System Baseline Metrics") # ⬅️ ADDED
@pytest.mark.gpu
@pytest.mark.cpu
def test_idle_baseline():
    dev_type = pick_device()
    device = get_device(dev_type)
    metrics_monitor = SystemMetrics(device)

    results = []
    for i in range(10):
        metrics = metrics_monitor.get_all_metrics()
        metrics["step"] = i
        results.append(metrics)
        time.sleep(0.3)

    allure.attach(json.dumps(results, indent=2), "Idle Baseline", allure.attachment_type.JSON)
    attach_chart_to_allure(results)
