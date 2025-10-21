# tests/test_parallel_training.py
import json
import time
import threading
import allure
import pytest
import torch

from scripts.plot_gpu_metrics import attach_chart_to_allure
from scripts.system_metrics import SystemMetrics
from tests.device_utils import pick_device, get_device, synchronize


def worker(thread_id, results, lock, metrics_monitor, dev_type, device):
    model = torch.nn.Linear(256, 256).to(device)
    x = torch.randn(1024, 256, dtype=torch.float32).to(device)
    for _ in range(5):
        _ = model(x)
        synchronize(dev_type)
        metrics = metrics_monitor.get_all_metrics()
        metrics["thread"] = thread_id
        with lock:
            results.append(metrics)
        time.sleep(0.1)


@allure.feature("System Parallelism") # ⬅️ ADDED
@allure.story("Multi-Threaded Load Test") # ⬅️ ADDED
@pytest.mark.gpu
@pytest.mark.cpu
@pytest.mark.benchmark
def test_parallel_training():
    dev_type = pick_device()
    device = get_device(dev_type)
    metrics_monitor = SystemMetrics(device)

    results = []
    lock = threading.Lock()
    threads = []

    for i in range(4):
        t = threading.Thread(target=worker, args=(i, results, lock, metrics_monitor, dev_type, device))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Sort & add step
    results.sort(key=lambda m: m.get("thread", 0))
    for i, m in enumerate(results):
        m["step"] = i

    allure.attach(json.dumps(results, indent=2), "Parallel Training Metrics", allure.attachment_type.JSON)
    attach_chart_to_allure(results)
