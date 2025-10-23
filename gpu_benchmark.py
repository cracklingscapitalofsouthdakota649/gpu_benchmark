#!/usr/bin/env python3
"""
gpu_benchmark.py — Unified benchmark runner with proper Allure trend support.
"""

import os
import sys
import subprocess
import platform
import psutil
import json
import time
import shutil

try:
    import pyopencl as cl
except ImportError:
    cl = None

# --------------------------
# Configuration
# --------------------------
ALLURE_RESULTS = "allure-results"
ALLURE_REPORT = "allure-report"
SUPPORT_DIR = "supports"
TARGET_PYTHON_VERSION = "3.10"


# --------------------------
# Find & Switch Python
# --------------------------
def find_python_executable(version=TARGET_PYTHON_VERSION):
    """Locate a Python executable matching the version."""
    try:
        if sys.platform.startswith("win"):
            cmd = ['py', f'-{version}', '-c', 'import sys;print(sys.executable)']
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            path = result.stdout.strip()
            if os.path.exists(path):
                return path
        else:
            name = f"python{version}"
            path = shutil.which(name)
            if path:
                return path
            for alt in [f"/usr/bin/{name}", f"/usr/local/bin/{name}"]:
                if os.path.exists(alt):
                    return alt
    except Exception:
        pass
    return sys.executable


def ensure_python310():
    """Re-run silently with Python 3.10 (no CMD echo)."""
    if not sys.version.startswith(TARGET_PYTHON_VERSION):
        py_path = find_python_executable(TARGET_PYTHON_VERSION)
        if py_path and py_path != sys.executable:
            os.environ["PYTHONUNBUFFERED"] = "1"
            # use os.execve instead of execv to suppress command echo
            os.execve(py_path, [py_path] + sys.argv, os.environ)


# --------------------------
# System Info Detection
# --------------------------
def detect_cpu_info():
    try:
        if sys.platform.startswith("win"):
            cpu = platform.processor()
            if not cpu:
                cpu = subprocess.check_output(
                    ["wmic", "cpu", "get", "name"], text=True
                ).split("\n")[1].strip()
            return cpu
        elif sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        return platform.processor()
    except Exception:
        return "Unknown CPU"


def detect_memory_info():
    try:
        return round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        return 0


def detect_gpu_info():
    vendor, name = "None", "None"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            return "NVIDIA", result.stdout.strip().split("\n")[0]
    except Exception:
        pass

    if cl:
        try:
            for p in cl.get_platforms():
                for d in p.get_devices():
                    if d.type & cl.device_type.GPU:
                        vendor = d.vendor.strip()
                        name = d.name.strip()
                        if "Intel" in vendor:
                            vendor = "Intel"
                        elif "AMD" in vendor or "Advanced Micro" in vendor:
                            vendor = "AMD"
                        return vendor, name
        except Exception:
            pass
    return vendor, name


# --------------------------
# Allure Environment Writer
# --------------------------
def write_environment_properties(build_number):
    dest_dir = os.path.join(ALLURE_RESULTS, str(build_number))
    os.makedirs(dest_dir, exist_ok=True)
    cpu = detect_cpu_info()
    gpu_vendor, gpu_model = detect_gpu_info()
    mem_gb = detect_memory_info()
    os_name = f"{platform.system()} {platform.release()}"
    path = os.path.join(dest_dir, "environment.properties")

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"CPU={cpu}\nGPU Vendor={gpu_vendor}\nGPU Model={gpu_model}\n")
        f.write(f"Total System Memory={mem_gb} GB\nOS={os_name}\n")

    print(f"[INFO] CPU: {cpu}")
    print(f"[INFO] GPU: {gpu_vendor} {gpu_model}")
    print(f"[INFO] Memory: {mem_gb} GB")
    print(f"[INFO] OS: {os_name}")


# --------------------------
# Allure Support
# --------------------------

def copy_categories(dest_dir):
    shutil.copy(os.path.join(SUPPORT_DIR, "categories.json"), dest_dir)
    print("[INFO] Copied categories.json to allure-results/")
    
def update_executor_json(build_number):
    path = os.path.join(ALLURE_RESULTS, str(build_number), "executor.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "name": "GPU Benchmark CI",
        "type": "local",
        "buildOrder": str(build_number),
        "buildName": f"GPU Benchmark #{build_number}",
        "reportUrl": "",
        "buildUrl": "",
        "description": f"Automated GPU/CPU benchmark build {build_number}"
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    

def copy_history(build_number):
    latest = os.path.join(ALLURE_RESULTS, "latest", "history")
    dest = os.path.join(ALLURE_RESULTS, str(build_number), "history")
    if os.path.exists(latest):
        shutil.copytree(latest, dest, dirs_exist_ok=True)
    else:
        os.makedirs(dest, exist_ok=True)


def save_latest(build_number):
    """Preserve latest results & history for Allure trend."""
    latest_dir = os.path.join(ALLURE_RESULTS, "latest")
    build_dir = os.path.join(ALLURE_RESULTS, str(build_number))
    report_history = os.path.join(ALLURE_REPORT, "history")

    if os.path.exists(latest_dir):
        shutil.rmtree(latest_dir)
    shutil.copytree(build_dir, latest_dir, dirs_exist_ok=True)

    hist_dir = os.path.join(latest_dir, "history")
    os.makedirs(hist_dir, exist_ok=True)

    # Copy Allure's own trend JSONs if they exist
    if os.path.exists(report_history):
        for file in os.listdir(report_history):
            if file.endswith(".json"):
                shutil.copy(os.path.join(report_history, file), hist_dir)


# --------------------------
# Pytest Runner
# --------------------------
def run_pytest(build_number, suite):
    results_dir = os.path.join(ALLURE_RESULTS, str(build_number))
    os.makedirs(results_dir, exist_ok=True)
    pytest_cmd = [
        sys.executable, "-m", "pytest", "-v",
        "-m", suite,
        "-p", "supports.telemetry_hook",
        f"--alluredir={results_dir}"
    ]
    print(f"[INFO] Running pytest suite: {suite.upper()}")
    return subprocess.run(pytest_cmd).returncode


# --------------------------
# Allure Report
# --------------------------
def generate_allure_report(build_number):
    results_dir = os.path.join(ALLURE_RESULTS, str(build_number))
    allure_bin = shutil.which("allure") or shutil.which("allure.cmd")
    if not allure_bin:
        print("[WARN] Allure CLI not found. Install with Scoop or npm.")
        return
    subprocess.run([allure_bin, "generate", results_dir, "-o", ALLURE_REPORT, "--clean"],
                   shell=False, check=False)


def open_allure_report():
    allure_bin = shutil.which("allure") or shutil.which("allure.cmd")
    if allure_bin:
        subprocess.Popen([allure_bin, "open", ALLURE_REPORT], shell=False)


# --------------------------
# Main Entrypoint
# --------------------------
def main():
    ensure_python310()
    if len(sys.argv) < 2:
        print("Usage: python gpu_benchmark.py <Build_Number> [suite]")
        sys.exit(1)

    build = sys.argv[1]
    suite = sys.argv[2] if len(sys.argv) > 2 else "gpu"

    print("=" * 80)
    print(f" Starting GPU Benchmark Suite | Build #{build} | Suite: {suite.upper()} ")
    print("=" * 80)

    results_dir = os.path.join(ALLURE_RESULTS, str(build))
    os.makedirs(results_dir, exist_ok=True)

    copy_categories(results_dir)
    copy_history(build)
    update_executor_json(build)
    write_environment_properties(build)

    start = time.time()
    rc = run_pytest(build, suite)
    dur = round(time.time() - start, 2)
    print(f"[INFO] Test execution completed in {dur}s (exit={rc}).")

    generate_allure_report(build)
    save_latest(build)
    open_allure_report()
    sys.exit(rc)


if __name__ == "__main__":
    main()
