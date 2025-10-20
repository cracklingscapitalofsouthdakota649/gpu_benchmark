# run_docker.py
import os
import sys
import shutil
import subprocess
import time
import platform
import json
import psutil  # pip install psutil

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGE_NAME = "gpu-benchmark:latest"

# Results & reports
ALLURE_RESULTS_DIR = os.path.join(PROJECT_ROOT, "allure-results")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
ALLURE_REPORT_DIR = os.path.join(REPORTS_DIR, "allure-report")

# Optional metadata folder
SUPPORTS_DIR = os.path.join(PROJECT_ROOT, "supports")

# Python version / venv
PYTHON_BIN = "/usr/bin/python3.10"
VENV_NAME = "venv310"


def run_cmd(cmd, check=True, shell=False):
    """Run a command on host."""
    try:
        proc = subprocess.run(cmd, check=check, text=True, capture_output=True, shell=shell)
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {' '.join(cmd)}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        if check:
            raise
        return e.returncode, e.stdout, e.stderr


def check_docker_running():
    """Check Docker daemon is running."""
    try:
        subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print("✅ Docker is running and responsive.")
    except Exception:
        print("❌ Docker is not running. Start Docker and retry.")
        sys.exit(1)


def ensure_image_built(image_name: str, context_dir: str = PROJECT_ROOT, dockerfile: str = "Dockerfile"):
    """Ensure Docker image exists, build if missing."""
    result = subprocess.run(["docker", "images", "-q", image_name], capture_output=True, text=True)
    if result.stdout.strip():
        print(f"✅ Found image {image_name}")
        return
    print(f"⚠️ Image '{image_name}' not found. Building...")
    subprocess.run(["docker", "build", "-t", image_name, "-f", dockerfile, context_dir], check=True)
    print(f"✅ Successfully built image {image_name}")


def container_has_pytest_benchmark():
    """Check if pytest-benchmark exists inside Docker."""
    try:
        res = subprocess.run(
            ["docker", "run", "--rm", IMAGE_NAME, PYTHON_BIN, "-m", "pip", "show", "pytest-benchmark"],
            capture_output=True, text=True
        )
        return res.returncode == 0 and "Name: pytest-benchmark" in (res.stdout or "")
    except Exception:
        return False


def prepare_allure_metadata(run_results_dir):
    """Copy environment and categories metadata + executor.json"""
    # Copy environment + categories
    for src_name, dest_name in [("windows.properties", "environment.properties"), ("categories.json", "categories.json")]:
        src_path = os.path.join(SUPPORTS_DIR, src_name)
        dest_path = os.path.join(run_results_dir, dest_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            print(f"Copied {src_name} -> {dest_name}")
        else:
            print(f"Warning: {src_name} not found, skipping.")

    # executor.json
    try:
        git_branch = subprocess.getoutput("git rev-parse --abbrev-ref HEAD")
        git_commit = subprocess.getoutput("git rev-parse --short HEAD")
        executor = {
            "name": "GPU Benchmark Runner",
            "type": "Local",
            "buildOrder": build_number,
            "buildName": f"GPU Benchmark Build #{build_number}",
            "data": {
                "Git Branch": git_branch,
                "Git Commit": git_commit,
                "OS": platform.system(),
                "Python": platform.python_version(),
                "Docker Image": IMAGE_NAME,
            },
        }
        with open(os.path.join(run_results_dir, "executor.json"), "w", encoding="utf-8") as f:
            json.dump(executor, f, indent=2)
        print("Wrote executor.json")
    except Exception as e:
        print(f"Warning: Failed to write executor.json: {e}")


def generate_allure_report(results_dir):
    """Generate Allure report."""
    os.makedirs(ALLURE_REPORT_DIR, exist_ok=True)
    run_cmd(["allure", "generate", results_dir, "-o", ALLURE_REPORT_DIR, "--clean"], check=False)
    print(f"[INFO] Allure report generated at {ALLURE_REPORT_DIR}")


def save_latest(build_number):
    """Save history and latest results for Allure trend."""
    latest_dir = os.path.join(ALLURE_RESULTS_DIR, "latest")
    build_dir = os.path.join(ALLURE_RESULTS_DIR, str(build_number))
    report_history = os.path.join(ALLURE_REPORT_DIR, "history")

    if os.path.exists(latest_dir):
        shutil.rmtree(latest_dir)
    shutil.copytree(build_dir, latest_dir, dirs_exist_ok=True)

    if os.path.exists(report_history):
        shutil.copytree(report_history, os.path.join(latest_dir, "history"), dirs_exist_ok=True)
        print("[INFO] Captured updated history for next trend.")
    else:
        print("[WARN] No history found; trend may be incomplete.")


def run_docker_tests(suite: str, run_results_dir: str):
    """Run Docker tests with optional benchmark plugin."""
    has_benchmark = container_has_pytest_benchmark()

    docker_test_command = [
        "docker", "run", "--rm",
        "-v", f"{ALLURE_RESULTS_DIR}:/app/allure-results",
        IMAGE_NAME,
        "pytest",
        "--alluredir=/app/allure-results",
        "-m", suite
    ]

    if has_benchmark:
        docker_test_command += [
            "--benchmark-json=/app/allure-results/benchmark.json",
            "--benchmark-save=latest",
            "--benchmark-autosave",
            "--benchmark-min-rounds=5",
        ]

    print(f"--- Running Docker tests for suite '{suite}' ---")
    subprocess.run(docker_test_command, check=True)


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)

    if len(sys.argv) < 2:
        print("Usage: python run_docker.py <BUILD_NUMBER> [suite]")
        sys.exit(1)

    build_number = sys.argv[1].strip()
    suite = sys.argv[2].strip() if len(sys.argv) > 2 else "preprocess"

    print(f"=== GPU Benchmark Build #{build_number}, Suite: {suite} ===")

    # Step 0: Docker
    check_docker_running()

    # Step 1: Workspace
    shutil.rmtree(os.path.join(ALLURE_RESULTS_DIR, build_number), ignore_errors=True)
    run_results_dir = os.path.join(ALLURE_RESULTS_DIR, build_number)
    os.makedirs(run_results_dir, exist_ok=True)

    # Seed history
    prev_history = os.path.join(ALLURE_RESULTS_DIR, "latest", "history")
    if os.path.exists(prev_history):
        shutil.copytree(prev_history, os.path.join(run_results_dir, "history"), dirs_exist_ok=True)
        print("Seeded previous history for trend.")
    else:
        print("No previous history found; trend will start fresh.")

    # Step 2: Docker image
    ensure_image_built(IMAGE_NAME)

    # Step 3: Metadata
    prepare_allure_metadata(run_results_dir)

    # Step 4: Docker tests
    run_docker_tests(suite, run_results_dir)

    # Step 5: Allure report
    generate_allure_report(run_results_dir)
    save_latest(build_number)

    print(f"\n=== GPU Benchmark Build #{build_number} ({suite}) Complete ===")
