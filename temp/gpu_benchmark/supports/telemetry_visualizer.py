# supports/telemetry_visualizer.py
"""
Utility to generate time-series plots for per-test telemetry data.
Requires matplotlib (pip install matplotlib).
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_telemetry_json(json_path: str, test_name: str, dest_dir: str) -> str or None:
    """
    Reads a telemetry JSON file and generates a time-series plot of 
    CPU, RAM, GPU, and VRAM utilization.

    Args:
        json_path: Full path to the telemetry JSON file.
        test_name: The name of the test (used for the plot title and filename).
        dest_dir: The directory where the plot image should be saved.

    Returns:
        The path to the generated PNG image, or None if plotting fails.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not read telemetry JSON at {json_path}: {e}")
        return None

    if not data:
        return None

    # --- 1. Prepare Data ---
    # Normalize timestamps to start from 0 for the plot
    start_time = data[0]['timestamp'] if data else 0
    timestamps = [d['timestamp'] - start_time for d in data]
    
    # Extract metrics. Use get() and ensure numeric data.
    cpu_data = np.array([d.get('cpu_percent', 0) for d in data])
    ram_data = np.array([d.get('ram_percent', 0) for d in data])
    gpu_data = np.array([d.get('gpu_util_percent', 0) for d in data])
    vram_data = np.array([d.get('vram_percent', 0) for d in data])

    # --- 2. Generate Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Primary Y-axis for CPU, RAM, GPU Utilization
    ax.plot(timestamps, cpu_data, label='CPU Util (%)', color='blue', marker='.', linestyle='-')
    ax.plot(timestamps, ram_data, label='RAM Util (%)', color='green', marker='.', linestyle='-')
    ax.plot(timestamps, gpu_data, label='GPU Util (%)', color='red', marker='.', linestyle='-')

    # Secondary Y-axis for VRAM (since it's also a percentage, we can keep it here, 
    # but a second axis helps if the scales are vastly different)
    ax.plot(timestamps, vram_data, label='VRAM Util (%)', color='purple', marker='.', linestyle='--')

    # Styling and Labels
    ax.set_title(f"Telemetry for: {test_name}", fontsize=14)
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Resource Utilization (%)", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105) # Cap at 100% + buffer

    # Save the plot
    image_filename = f"telemetry_{test_name}.png"
    image_path = os.path.join(dest_dir, image_filename)
    
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close(fig) # Close the figure to free memory

    return image_path

if __name__ == '__main__':
    # Simple test case (requires a dummy JSON file)
    print("This module is a utility. To test, ensure you have a telemetry JSON file.")
    # Example usage:
    # plot_telemetry_json(
    #     'allure-results/manual/telemetry_test_gpu_op.json', 
    #     'test_gpu_op', 
    #     'allure-results/manual'
    # )
