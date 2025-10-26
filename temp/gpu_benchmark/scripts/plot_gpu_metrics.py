# scripts/plot_gpu_metrics.py
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing

import matplotlib.pyplot as plt
import io
import base64
import allure

def attach_chart_to_allure(metrics):
    """
    Attach GPU/CPU usage chart to Allure report.
    """
    if not metrics:
        return

    steps = [m.get("step", i) for i, m in enumerate(metrics)]
    gpu_util = [m.get("gpu_util", 0) for m in metrics]
    cpu_util = [m.get("cpu_util", 0) for m in metrics]

    plt.figure(figsize=(7, 4))
    plt.plot(steps, gpu_util, label="GPU Util (%)")
    plt.plot(steps, cpu_util, label="CPU Util (%)")
    plt.xlabel("Step")
    plt.ylabel("Utilization (%)")
    plt.title("GPU/CPU Utilization over Time")
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    allure.attach(buf.read(), name="GPU_CPU_Util", attachment_type=allure.attachment_type.PNG)
