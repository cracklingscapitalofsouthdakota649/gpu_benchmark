import os
import sys
import subprocess
import shutil
import datetime
import platform
import venv
import json

# -----------------------
# Configuration
# -----------------------
ALLURE_RESULTS = "allure-results"
ALLURE_REPORT = "allure-report"
SUPPORT_DIR = "supports"

COMMON_PACKAGES = [
    "pytest==8.4.2",
    "pytest-benchmark==5.1.0",
    "allure-pytest==2.15.0",
    "psutil==7.1.0",
    "matplotlib==3.10.7",
    "numpy==2.2.6",
]

TORCH_PACKAGES = {
    "cuda": ["torch==2.2.0", "torchvision==0.18.1", "torchaudio==2.2.0"],
    "rocm": ["torch==2.2.0", "torchvision==0.18.1", "torchaudio==2.2.0"],
    "intel_opencl": ["torch==2.2.0", "torchvision==0.18.1", "torchaudio==2.2.0"],
    "intel_xpu": ["torch==2.2.0", "torchvision==0.18.1", "torchaudio==2.2.0"],
    "directml": ["torch==2.2.0", "torchvision==0.18.1", "torchaudio==2.2.0"],
    "cpu": ["torch==2.2.0", "torchvision==0.18.1", "torchaudio==2.2.0"],
}

# -----------------------
# Utilities
# -----------------------
def run_cmd(cmd, check=True):
    """Run a command, resolving allure.cmd if needed on Windows."""
    if platform.system().lower() == "windows" and cmd[0] == "allure":
        allure_path = shutil.which("allure")
        if allure_path is None:
            print("[ERROR] Allure not found in PATH.")
            sys.exit(1)
        cmd[0] = allure_path
        return subprocess.run(" ".join(cmd), check=check, shell=True)
    else:
        return subprocess.run(cmd, check=check)


# -----------------------
# GPU Detection
# -----------------------
def detect_gpu_flavor():
    """Detect CUDA, ROCm, Intel XPU/OpenCL, or fallback to CPU."""
    try:
        # Try running supports/gpu_check.py if available
        if os.path.exists(os.path.join(SUPPORT_DIR, "gpu_check.py")):
            ret = subprocess.run(
                [sys.executable, os.path.join(SUPPORT_DIR, "gpu_check.py")],
                capture_output=True, text=True
            )
            if ret.returncode == 0:
                data = json.loads(ret.stdout.strip())
                summary = data.get("summary", {})
                if summary.get("cuda"):
                    return "cuda"
                elif summary.get("rocm"):
                    return "rocm"
                elif summary.get("directml"):
                    return "directml"
                elif summary.get("intel_gpu") or summary.get("opencl_gpu_devices", 0) > 0:
                    return "intel_opencl"
                else:
                    return "cpu"
    except Exception as e:
        print(f"[WARN] gpu_check.py failed: {e}")

    # Fallback detection without gpu_check.py
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "intel_xpu"
    except Exception:
        pass

    # OpenCL check
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        if any("Intel" in d.name for p in platforms for d in p.get_devices()):
            return "intel_opencl"
    except Exception:
        pass

    return "cpu"


# -----------------------
# Package Installation
# -----------------------
def install_packages(packages):
    python_exec = sys.executable
    for pkg in packages:
        print(f"[INFO] Installing {pkg}")
        subprocess.run(
            f'"{python_exec}" -m pip install --no-deps -q {pkg}',
            shell=True,
            check=False
        )


# -----------------------
# Allure & Reporting
# -----------------------
def update_executor_json(build_number):
    """Dynamically update executor.json for each run."""
    executor_template = os.path.join(SUPPORT_DIR, "executor.json")
    if not os.path.exists(executor_template):
        print("[WARN] executor.json template missing; skipping update.")
        return
    with open(executor_template, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["buildOrder"] = str(build_number)
    data["buildName"] = f"GPU Benchmark #{build_number}"
    data["description"] = f"Benchmark suite for GPU performance testing (Build {build_number})"

    dest = os.path.join(ALLURE_RESULTS, str(build_number), "executor.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Updated executor.json for build {build_number}")


def copy_support_files(dest_dir):
    shutil.copy(os.path.join(SUPPORT_DIR, "categories.json"), dest_dir)
    print("[INFO] Copied categories.json to allure-results/")

    shutil.copy(os.path.join(SUPPORT_DIR, "executor.json"), dest_dir)
    print("[INFO] Copied executor.json to allure-results/")

    env_file = "windows.properties" if platform.system() == "Windows" else "ubuntu.properties"
    shutil.copy(
        os.path.join(SUPPORT_DIR, env_file),
        os.path.join(dest_dir, "environment.properties"),
    )
    print(f"[INFO] Copied {env_file} as environment.properties")


def copy_history(build_number):
    latest_dir = os.path.join(ALLURE_RESULTS, "latest")
    history_src = os.path.join(latest_dir, "history")
    history_dest = os.path.join(ALLURE_RESULTS, str(build_number), "history")

    if os.path.exists(history_src):
        shutil.copytree(history_src, history_dest, dirs_exist_ok=True)
        print(f"[INFO] Imported trend history into build {build_number}")
    else:
        os.makedirs(history_dest, exist_ok=True)
        print(f"[INFO] Initialized new history for build {build_number}")


def save_latest(build_number):
    latest_dir = os.path.join(ALLURE_RESULTS, "latest")
    build_dir = os.path.join(ALLURE_RESULTS, str(build_number))
    report_history = os.path.join(ALLURE_REPORT, "history")

    if os.path.exists(latest_dir):
        shutil.rmtree(latest_dir)
    shutil.copytree(build_dir, latest_dir, dirs_exist_ok=True)

    if os.path.exists(report_history):
        shutil.copytree(report_history, os.path.join(latest_dir, "history"), dirs_exist_ok=True)
        print("[INFO] Captured updated history from Allure report.")
    else:
        print("[WARN] No history found in report.")

    print("[INFO] Updated latest results for next trend comparison")


def generate_allure_report(results_dir):
    print("[INFO] Generating Allure report...")
    run_cmd(["allure", "generate", results_dir, "-o", ALLURE_REPORT, "--clean"], check=False)
    print("[INFO] Report successfully generated to allure-report")


def open_allure_report():
    print("Starting Allure web server...")
    allure_cmd = "allure"
    shell_required = False
    if platform.system().lower() == "windows":
        allure_path = shutil.which("allure")
        if allure_path:
            allure_cmd = allure_path
            shell_required = True
        else:
            print("[ERROR] Allure not found in PATH.")
            sys.exit(1)
    try:
        process = subprocess.Popen([allure_cmd, "open", ALLURE_REPORT], shell=shell_required)
        print("Server started. Press <Ctrl+C> to stop.")
        process.wait()
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped by user.")
        try:
            process.terminate()
        except Exception:
            pass
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Failed to open Allure report: {e}")


# -----------------------
# Main
# -----------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python gpu_benchmark.py <Build_Number> [suite]")
        sys.exit(1)

    build_number = sys.argv[1]
    suite = sys.argv[2] if len(sys.argv) > 2 else None

    flavor = detect_gpu_flavor()
    print(f"[INFO] Detected GPU flavor: {flavor}")

    install_packages(COMMON_PACKAGES)
    install_packages(TORCH_PACKAGES.get(flavor, TORCH_PACKAGES["cpu"]))

    results_dir = os.path.join(ALLURE_RESULTS, str(build_number))
    os.makedirs(results_dir, exist_ok=True)

    copy_history(build_number)
    copy_support_files(results_dir)
    update_executor_json(build_number)

    pytest_cmd = [sys.executable, "-m", "pytest", "-v", "--alluredir", results_dir]
    if suite:
        pytest_cmd.extend(["-m", suite])

    print("[INFO] Running pytest benchmark...")
    run_cmd(pytest_cmd, check=False)
    print("[INFO] Pytest run completed.")

    generate_allure_report(results_dir)
    save_latest(build_number)
    open_allure_report()


if __name__ == "__main__":
    main()
