#!/usr/bin/env python3
"""
gpu_benchmark.py — Unified benchmark runner with Allure telemetry trend support.
"""

import os
import sys
import subprocess
import platform
import psutil
import json
import time
import shutil
import signal

from supports.performance_trend import update_performance_trend, plot_trend_chart

try:
    import pyopencl as cl
except ImportError:
    cl = None

ALLURE_RESULTS = "allure-results"
ALLURE_REPORT = "allure-report"
SUPPORT_DIR = "supports"
TARGET_PYTHON_VERSION = "3.10"


# --------------------------
# Python Version Management
# --------------------------
def find_python_executable(version=TARGET_PYTHON_VERSION):
    """Find specific Python version on Windows or *nix."""
    try:
        if sys.platform.startswith("win"):
            result = subprocess.run(
                ["py", f"-{version}", "-c", "import sys;print(sys.executable)"],
                capture_output=True, text=True, check=True, timeout=5
            )
            path = result.stdout.strip()
            if path and os.path.exists(path):
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
    """Re-run under Python 3.10 quietly and wait until it finishes."""
    if not sys.version.startswith(TARGET_PYTHON_VERSION):
        py_path = find_python_executable(TARGET_PYTHON_VERSION)
        if not py_path or py_path == sys.executable:
            return

        print(f"[INFO] Switching to Python {TARGET_PYTHON_VERSION}: {py_path}")

        if sys.platform.startswith("win"):
            # Build quoted argument list safely
            args_str = " ".join([f'"{a}"' for a in sys.argv])
            ps_cmd = f'& "{py_path}" {args_str}'

            # Run invisibly but wait until completion
            subprocess.run(
                ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps_cmd],
                check=False,
            )
            sys.exit(0)
        else:
            os.execve(py_path, [py_path] + sys.argv, os.environ)

# --------------------------
# Hardware Info
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
                        return line.split(":", 1)[1].strip()
        return platform.processor() or "Unknown CPU"
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
# Allure Utilities
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


def ensure_categories(dest_dir):
    src = os.path.join(SUPPORT_DIR, "categories.json")
    dest = os.path.join(dest_dir, "categories.json")
    os.makedirs(dest_dir, exist_ok=True)
    if os.path.exists(src):
        shutil.copy(src, dest)
    else:
        default = [
            {"name": "Performance", "matchedStatuses": ["passed"], "messageRegex": ".*"},
            {"name": "Failures", "matchedStatuses": ["failed"], "messageRegex": ".*"},
        ]
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(default, f, indent=2)


def update_executor_json(build_number):
    path = os.path.join(ALLURE_RESULTS, str(build_number), "executor.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "name": "GPU Benchmark CI",
        "type": "local",
        "buildOrder": str(build_number),
        "buildName": f"GPU Benchmark #{build_number}",
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


# --------------------------
# Pytest Runner (with signal forwarding)
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

    # Graceful SIGINT handling
    process = subprocess.Popen(pytest_cmd)
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n[WARN] Ctrl+C detected — terminating benchmark...")
        process.send_signal(signal.SIGINT)
        process.wait()
    return process.returncode


# --------------------------
# Allure Reporting
# --------------------------
def generate_allure_report(build_number):
    results_dir = os.path.join(ALLURE_RESULTS, str(build_number))
    allure_bin = shutil.which("allure") or shutil.which("allure.cmd")
    if not allure_bin:
        print("[WARN] Allure CLI not found. Install it via Scoop or npm.")
        return
    subprocess.run([allure_bin, "generate", results_dir, "-o", ALLURE_REPORT, "--clean"])
    print(f"[INFO] Report generated to {ALLURE_REPORT}")


def open_allure_report():
    allure_bin = shutil.which("allure") or shutil.which("allure.cmd")
    if allure_bin:
        subprocess.Popen([allure_bin, "open", ALLURE_REPORT])


# --------------------------
# Main Entrypoint
# --------------------------
def main():
    ensure_python310()
    if len(sys.argv) < 2:
        print("Usage: python gpu_benchmark.py <Build_Number> [suite]")
        sys.exit(1)

    build_number = sys.argv[1]
    suite = sys.argv[2] if len(sys.argv) > 2 else "gpu"

    print("=" * 80)
    print(f" Starting GPU Benchmark Suite | Build #{build_number} | Suite: {suite.upper()} ")
    print("=" * 80)

    results_dir = os.path.join(ALLURE_RESULTS, str(build_number))
    os.makedirs(results_dir, exist_ok=True)

    ensure_categories(results_dir)
    copy_history(build_number)
    update_executor_json(build_number)
    write_environment_properties(build_number)

    start = time.time()
    rc = run_pytest(build_number, suite)
    print(f"[INFO] Test execution completed in {round(time.time() - start, 2)}s (exit={rc}).")

    generate_allure_report(build_number)

    # --- Performance Trend ---
    trend = update_performance_trend(
        build_number,
        results_dir=ALLURE_RESULTS,
        history_dir=os.path.join(ALLURE_REPORT, "history")
    )
    if trend:
        chart = plot_trend_chart(trend, os.path.join(ALLURE_RESULTS, str(build_number)), build_number)
        if chart and os.path.exists(chart):
            print(f"[INFO] Trend chart generated: {chart}")
    else:
        print("[WARN] No telemetry data found for trend (check supports/telemetry_hook).")

    open_allure_report()
    sys.exit(rc)


if __name__ == "__main__":
    main()
