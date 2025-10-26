# supports/telemetry_trend.py
import os
import json
import numpy as np
from glob import glob
from datetime import datetime


def compute_module_average(path):
    """Compute average CPU/GPU/RAM usage for a telemetry JSON."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            return None

        cpu = np.mean([d.get("cpu_percent", 0) for d in data])
        ram = np.mean([d.get("ram_percent", 0) for d in data])
        gpu = np.mean([
            d.get("gpu_util_percent", 0) or 0
            for d in data
            if d.get("gpu_util_percent") is not None
        ])
        return {
            "cpu_avg": round(cpu, 2),
            "ram_avg": round(ram, 2),
            "gpu_avg": round(gpu, 2),
        }
    except Exception as e:
        print(f"[WARN] Failed to compute telemetry avg for {path}: {e}")
        return None


def build_trend_summary(build_number, results_dir="allure-results"):
    """Aggregate all telemetry JSON files and update trend summary."""
    build_dir = os.path.join(results_dir, str(build_number))
    telemetry_files = glob(os.path.join(build_dir, "telemetry_*.json"))
    summary = {
        "build_number": str(build_number),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "modules": {},
        "totals": {"cpu_avg": 0, "ram_avg": 0, "gpu_avg": 0},
    }

    if not telemetry_files:
        print("[WARN] No telemetry data found.")
        return None

    for path in telemetry_files:
        mod_name = os.path.basename(path).replace("telemetry_", "").replace(".json", "")
        avg = compute_module_average(path)
        if avg:
            summary["modules"][mod_name] = avg

    # Compute overall suite averages
    if summary["modules"]:
        cpu_vals = [v["cpu_avg"] for v in summary["modules"].values()]
        ram_vals = [v["ram_avg"] for v in summary["modules"].values()]
        gpu_vals = [v["gpu_avg"] for v in summary["modules"].values()]
        summary["totals"] = {
            "cpu_avg": round(float(np.mean(cpu_vals)), 2),
            "ram_avg": round(float(np.mean(ram_vals)), 2),
            "gpu_avg": round(float(np.mean(gpu_vals)), 2),
        }

    # Save trend summary for the build
    out_path = os.path.join(build_dir, "telemetry_trend.json")
    os.makedirs(build_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Saved telemetry trend summary to {out_path}")

    # Copy to latest history folder
    latest_hist = os.path.join(results_dir, "latest", "history")
    os.makedirs(latest_hist, exist_ok=True)
    with open(os.path.join(latest_hist, "telemetry_trend.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[INFO] Updated latest trend history for Allure.")

    return summary


if __name__ == "__main__":
    import sys
    build_number = sys.argv[1] if len(sys.argv) > 1 else "local"
    build_trend_summary(build_number)
