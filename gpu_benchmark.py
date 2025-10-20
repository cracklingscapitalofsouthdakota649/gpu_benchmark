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
VENV_DIR = "venv310"
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
    "intel": ["torch==2.2.0", "torchvision==0.18.1", "torchaudio==2.2.0"],
    "directml": ["torch==2.2.0", "torchvision==0.18.1", "torchaudio==2.2.0"],
    "cpu": ["torch==2.2.0", "torchvision==0.18.1", "torchaudio==2.2.0"],
}

# -----------------------
# Utilities
# -----------------------
def run_cmd(cmd, check=True):
    """Run a command. On Windows, resolve allure.cmd if needed."""
    if platform.system().lower() == "windows" and cmd[0] == "allure":
        allure_path = shutil.which("allure")
        if allure_path is None:
            print("[ERROR] Allure not found in PATH.")
            sys.exit(1)
        cmd[0] = allure_path
        return subprocess.run(" ".join(cmd), check=check, shell=True)
    else:
        return subprocess.run(cmd, check=check)


def create_venv():
    if not os.path.exists(VENV_DIR):
        print(f"[INFO] Creating virtual environment: {VENV_DIR}")
        venv.create(VENV_DIR, with_pip=True)
    else:
        print(f"[INFO] Using existing virtual environment: {VENV_DIR}")


def detect_gpu_flavor():
    """Detect GPU flavor"""
    try:
        ret = subprocess.run(
            [sys.executable, os.path.join(SUPPORT_DIR, "gpu_check.py")],
            capture_output=True,
            text=True,
        )
        if ret.returncode != 0:
            print(f"[WARN] gpu_check.py error: {ret.stderr.strip()}")
            return "cpu"
        info = ret.stdout.strip()
        if not info:
            print(f"[WARN] gpu_check.py returned no output")
            return "cpu"
        data = json.loads(info)
        summary = data.get("summary", {})
        if summary.get("cuda"):
            return "cuda"
        elif summary.get("rocm"):
            return "rocm"
        elif summary.get("directml"):
            return "directml"
        elif summary.get("opencl_gpu_devices", 0) > 0:
            return "intel"
        else:
            return "cpu"
    except Exception as e:
        print(f"[WARN] Failed to detect GPU: {e}")
        return "cpu"


def install_packages(pip_exe, packages):
    for pkg in packages:
        run_cmd([pip_exe, "install", "--no-deps", pkg])


def update_executor_json(build_number):
    """Dynamically update executor.json for each run"""
    executor_template = os.path.join(SUPPORT_DIR, "executor.json")
    if not os.path.exists(executor_template):
        print("[WARN] executor.json template missing; skipping update.")
        return
    with open(executor_template, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["buildOrder"] = str(build_number)
    data["buildName"] = f"GPU Benchmark #{build_number}"
    data["description"] = (
        f"Benchmark suite for GPU performance testing (Build {build_number})"
    )

    dest = os.path.join(ALLURE_RESULTS, str(build_number), "executor.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Updated executor.json for build {build_number}")


def copy_support_files(dest_dir):
    """Copy categories, executor, environment.properties"""
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


# -----------------------
# Trend Support
# -----------------------
def copy_history(build_number):
    """Pull previous run's history into this build"""
    latest_dir = os.path.join(ALLURE_RESULTS, "latest")
    history_src = os.path.join(latest_dir, "history")
    history_dest = os.path.join(ALLURE_RESULTS, str(build_number), "history")

    if os.path.exists(history_src):
        shutil.copytree(history_src, history_dest, dirs_exist_ok=True)
        print(f"[INFO] Imported trend history from previous run into build {build_number}")
    else:
        os.makedirs(history_dest, exist_ok=True)
        history_json = {
            "reportName": f"GPU Benchmark Build {build_number}",
            "statistic": {"passed": 0, "failed": 0, "broken": 0, "skipped": 0, "unknown": 0},
            "time": {"start": int(datetime.datetime.now().timestamp() * 1000)},
        }
        with open(os.path.join(history_dest, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history_json, f, indent=2)
        print(f"[INFO] Initialized new history for first trend run (build {build_number})")


def save_latest(build_number):
    """Save current results and update 'latest' including new history"""
    latest_dir = os.path.join(ALLURE_RESULTS, "latest")
    build_dir = os.path.join(ALLURE_RESULTS, str(build_number))
    report_history = os.path.join(ALLURE_REPORT, "history")

    if os.path.exists(latest_dir):
        shutil.rmtree(latest_dir)
    shutil.copytree(build_dir, latest_dir, dirs_exist_ok=True)

    if os.path.exists(report_history):
        shutil.copytree(report_history, os.path.join(latest_dir, "history"), dirs_exist_ok=True)
        print("[INFO] Captured updated history from Allure report for next trend.")
    else:
        print("[WARN] No history found in report; trend may be incomplete.")

    print("[INFO] Updated latest results for next trend comparison")


def generate_allure_report(results_dir):
    """Generate Allure report"""
    print("[INFO] Generating Allure report...")
    run_cmd(["allure", "generate", results_dir, "-o", ALLURE_REPORT, "--clean"], check=False)
    print("[INFO] Report successfully generated to allure-report")


def open_allure_report():
    """Open the generated Allure report in a browser"""
    print("Starting web server to open report...")

    # Use the same logic from run_cmd to find allure path on Windows
    allure_cmd = "allure"
    shell_required = False
    if platform.system().lower() == "windows":
        allure_path = shutil.which("allure")
        if allure_path:
            allure_cmd = allure_path
            shell_required = True  # .cmd files often require shell=True
        else:
            print("[ERROR] Allure not found in PATH.")
            sys.exit(1)

    try:
        process = subprocess.Popen([allure_cmd, "open", ALLURE_REPORT], shell=shell_required)
        print("Server started. Press <Ctrl+C> in this terminal to stop the server and exit.")
        process.wait()
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped by user (Ctrl+C). Exiting gracefully...")
        try:
            process.terminate()
        except Exception:
            pass
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Failed to open Allure report: {e}")
        print("Please open the report manually by running: allure open allure-report")


# -----------------------
# Main
# -----------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python gpu_benchmark.py <Build_Number> [suite]")
        sys.exit(1)

    build_number = sys.argv[1]
    suite = sys.argv[2] if len(sys.argv) > 2 else None

    create_venv()
    pip_exe = os.path.join(VENV_DIR, "Scripts", "pip.exe")
    python_exe = os.path.join(VENV_DIR, "Scripts", "python.exe")

    flavor = detect_gpu_flavor()
    print(f"[INFO] Detected GPU flavor: {flavor}")

    install_packages(pip_exe, COMMON_PACKAGES)
    install_packages(pip_exe, TORCH_PACKAGES.get(flavor, TORCH_PACKAGES["cpu"]))

    results_dir = os.path.join(ALLURE_RESULTS, str(build_number))
    os.makedirs(results_dir, exist_ok=True)

    copy_history(build_number)
    copy_support_files(results_dir)
    update_executor_json(build_number)

    pytest_cmd = [python_exe, "-m", "pytest", "-v", "--alluredir", results_dir]
    if suite:
        pytest_cmd.extend(["-m", suite])
    run_cmd(pytest_cmd)

    # 1. Generate the report
    generate_allure_report(results_dir)
    
    # 2. Save the new history *before* opening the report
    save_latest(build_number)
    
    # 3. Now, open the report for viewing
    open_allure_report()


if __name__ == "__main__":
    main()