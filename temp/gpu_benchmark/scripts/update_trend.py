# scripts/update_trend.py
import json
import os
import time
import glob
from datetime import datetime

RESULTS_DIR = "allure-results"
HISTORY_DIR = os.path.join(RESULTS_DIR, "history")
HISTORY_FILE = os.path.join(HISTORY_DIR, "history-trend.json")
SUMMARY_GLOB = os.path.join(RESULTS_DIR, "summary_*.json")
MAX_RUNS = 20

def read_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not read {path}. Error: {e}")
        return None

def main():
    os.makedirs(HISTORY_DIR, exist_ok=True)
    
    # Find all individual test summaries
    summary_files = glob.glob(SUMMARY_GLOB)
    if not summary_files:
        print("Warning: No 'summary_*.json' files found. Allure trend will not be updated.")
        return

    all_summaries = []
    for f in summary_files:
        summary_data = read_json(f)
        if summary_data:
            all_summaries.append(summary_data)
    
    if not all_summaries:
        print("Error: Found summary files, but failed to read any. Trend not updated.")
        return

    # Calculate average metrics across all summaries
    num_summaries = len(all_summaries)
    avg_gpu = sum(s.get("avg_util", 0) for s in all_summaries) / num_summaries
    avg_mem = sum(s.get("avg_mem", 0) for s in all_summaries) / num_summaries
    avg_cpu = sum(s.get("avg_cpu_util", 0) for s in all_summaries) / num_summaries
    avg_fps = sum(s.get("fps", 0) for s in all_summaries) / num_summaries
    
    # Use GPU name from the first summary file
    gpu_name = all_summaries[0].get("gpu_name", "Unknown GPU")

    # Create a single aggregated entry for this run
    run_entry = {
        "buildOrder": int(time.time()),
        "reportName": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gpu_avg_util": round(avg_gpu, 2),
        "gpu_avg_mem": round(avg_mem, 2),
        "cpu_avg_util": round(avg_cpu, 2),
        "fps": round(avg_fps, 2),
        "gpu_name": gpu_name
    }

    history = read_json(HISTORY_FILE) or []
    history.append(run_entry)
    history = history[-MAX_RUNS:]  # Keep only the last MAX_RUNS

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"✅ Allure trend updated with aggregated metrics from {num_summaries} test(s) for {gpu_name}")

if __name__ == "__main__":
    main()