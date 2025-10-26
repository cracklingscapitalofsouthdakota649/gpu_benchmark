# supports/performance_trend.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


def compute_averages_from_telemetry(build_dir):
    """Compute avg CPU/GPU/RAM usage per build from telemetry JSONs."""
    telemetry_files = glob(os.path.join(build_dir, "telemetry_*.json"))
    if not telemetry_files:
        return None

    cpu_vals, ram_vals, gpu_vals = [], [], []

    for path in telemetry_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            cpu_vals += [d.get("cpu_percent", 0) for d in data if d.get("cpu_percent") is not None]
            ram_vals += [d.get("ram_percent", 0) for d in data if d.get("ram_percent") is not None]
            gpu_vals += [d.get("gpu_util_percent", 0) or 0 for d in data]

    if not cpu_vals:
        return None

    return {
        "cpu_avg": round(float(np.mean(cpu_vals)), 2),
        "ram_avg": round(float(np.mean(ram_vals)), 2),
        "gpu_avg": round(float(np.mean(gpu_vals)), 2),
    }


def update_performance_trend(build_number, results_dir="allure-results", history_dir=None, max_builds=20):
    """Append new build averages to Allure's trend history JSON."""
    build_dir = os.path.join(results_dir, str(build_number))
    history_dir = history_dir or os.path.join(results_dir, "latest", "history")
    os.makedirs(history_dir, exist_ok=True)
    trend_path = os.path.join(history_dir, "performance-trend.json")

    current = compute_averages_from_telemetry(build_dir)
    if not current:
        print("[WARN] No telemetry data to build performance trend.")
        return

    trend = []
    if os.path.exists(trend_path):
        try:
            with open(trend_path, "r", encoding="utf-8") as f:
                trend = json.load(f)
        except Exception:
            trend = []

    # Append new record
    entry = {
        "build_number": str(build_number),
        "cpu_avg": current["cpu_avg"],
        "ram_avg": current["ram_avg"],
        "gpu_avg": current["gpu_avg"],
    }
    trend.append(entry)
    trend = trend[-max_builds:]  # keep last N builds

    with open(trend_path, "w", encoding="utf-8") as f:
        json.dump(trend, f, indent=2)

    print(f"[INFO] Updated performance trend ({len(trend)} builds tracked).")

    return trend


def plot_trend_chart(trend, dest_dir, build_number):
    """Generate PNG chart visualizing performance trend."""
    if not trend:
        return None

    os.makedirs(dest_dir, exist_ok=True)
    builds = [t["build_number"] for t in trend]
    cpu = [t["cpu_avg"] for t in trend]
    ram = [t["ram_avg"] for t in trend]
    gpu = [t["gpu_avg"] for t in trend]

    plt.figure(figsize=(8, 5))
    plt.plot(builds, cpu, marker="o", label="CPU (%)")
    plt.plot(builds, ram, marker="o", label="RAM (%)")
    plt.plot(builds, gpu, marker="o", label="GPU (%)")
    plt.title("Performance Trend (Last Builds)")
    plt.xlabel("Build Number")
    plt.ylabel("Average Utilization (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(dest_dir, f"performance_trend_{build_number}.png")
    plt.savefig(path)
    plt.close()
    print(f"[INFO] Trend chart saved: {path}")
    return path
