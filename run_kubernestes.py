# filename: run_kubernestes.py
# purpose: Automation script to build the Docker image, run GPU benchmarks inside the container, 
#          and generate a local Allure report using command-line arguments (BUILD_NUMBER and SUITE).

import sys
import subprocess
import os
import argparse
import platform
import shutil
import json
import time
import webbrowser
import signal

# --- Constants ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_IMAGE_TAG = "luckyjoy/gpu-benchmark-local:latest"

ALLURE_RESULTS_DIR = os.path.join(PROJECT_ROOT, "allure-results")
ALLURE_REPORT_DIR = os.path.join(PROJECT_ROOT, "allure-report")
SUPPORTS_DIR = os.path.join(PROJECT_ROOT, "supports")

# FIX 1: Define the explicit path to the Python executable inside the Docker image
CONTAINER_PYTHON_BIN = "/usr/bin/python3.10" 

# Windows Process Group Creation Flag for robust Ctrl+C handling
CREATE_NEW_PROCESS_GROUP = 0x00000200 if platform.system() == "Windows" else 0

# --- Helper Functions ---

def execute_command(command, error_message, check_output=False, exit_on_error=True, suppress_output=False, shell=True, timeout=None):
    """
    Executes a shell command using subprocess.Popen for better cross-platform Ctrl+C handling.
    """
    stdout_target = subprocess.PIPE if (check_output or suppress_output) else sys.stdout
    stderr_target = subprocess.PIPE if (check_output or suppress_output) else sys.stderr
    
    process = subprocess.Popen(
        command,
        text=True,
        stdout=stdout_target,
        stderr=stderr_target,
        shell=shell,
        # Required on Windows for process group management
        creationflags=CREATE_NEW_PROCESS_GROUP 
    )

    try:
        process.wait(timeout=timeout)
        stdout, stderr = (process.communicate() if check_output or suppress_output else (None, None))
        if check_output:
            return stdout.strip()

    except subprocess.CalledProcessError as e:
        print("\n==========================================================")
        print(f"FATAL UNHANDLED ERROR during command execution: {error_message}")
        print(f"Command failed: {command}")
        print(f"Return Code: {e.returncode}")
        if e.output:
            print(f"Output: {e.output.strip()}")
        print("==========================================================")
        if exit_on_error:
            sys.exit(e.returncode)
        raise
    except KeyboardInterrupt:
        # Ctrl+C was pressed. Immediately terminate the process tree.
        print("\n[WARN] Ctrl+C detected in synchronous command — terminating process group...", file=sys.stderr)
        
        if platform.system() == "Windows":
             # Use taskkill to kill the entire process tree (/T) by PID
             subprocess.call(
                 ['taskkill', '/F', '/T', '/PID', str(process.pid)],
                 creationflags=CREATE_NEW_PROCESS_GROUP,
                 stdout=subprocess.DEVNULL,
                 stderr=subprocess.DEVNULL
             )
        else:
            # Standard POSIX: Send SIGINT to the process group
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
            except OSError:
                process.terminate()

        # Re-raise the interrupt to stop the main script execution
        raise
    
    if process.returncode != 0 and exit_on_error:
        # If output was suppressed, print it now for debugging the failure
        if suppress_output or check_output:
            print("\n==========================================================")
            print(f"FATAL UNHANDLED ERROR during command execution: {error_message}")
            print(f"Command failed: {command}")
            print(f"Return Code: {process.returncode}")
            if stdout:
                print(f"Output: {stdout.strip()}")
            print("==========================================================")
        sys.exit(process.returncode)
    
    return 0 if process.returncode == 0 else process.returncode


def check_docker_running():
    """Checks if the Docker daemon is running."""
    try:
        # Use suppress_output=True to keep output clean, but keep exit_on_error=True
        execute_command("docker info", "Docker daemon is not running.", suppress_output=True, exit_on_error=True)
        print("✅ Docker is running and responsive.")
    except Exception:
        sys.exit(1)


def ensure_image_built(tag):
    """
    Builds the Docker image if it doesn't exist.
    Forces a rebuild with --no-cache if the image is missing to ensure all complex dependencies
    (like PyTorch/TensorFlow) install correctly, preventing 'No module named pytest' errors.
    """
    image_exists = execute_command(f"docker images -q {tag}", "", check_output=True, exit_on_error=False)
    
    if image_exists:
        print(f"✅ Found image {tag}. Skipping build.")
        return

    # FIX 3: Add --no-cache to force a clean dependency installation
    print(f"⚠️ Docker image '{tag}' not found. Starting clean build (using --no-cache)...")
    execute_command(
        f"docker build --no-cache -t {tag} -f Dockerfile {PROJECT_ROOT}",
        "Docker image build failed. Check the Docker logs for dependency installation errors (e.g., PyTorch/TensorFlow compile failures).",
        exit_on_error=True
    )
    print(f"✅ Successfully built image {tag}")


def write_executor_json(build_number: str, suite: str):
    """Generates the Allure executor.json file using defined variables."""
    try:
        executor_path = os.path.join(ALLURE_RESULTS_DIR, build_number, "executor.json")
        executor_data = {
            "name": "Local Docker Runner",
            "type": "job",
            "url": "N/A",  # Not running on a CI server
            "buildOrder": int(build_number),
            "buildName": f"Build #{build_number} - Suite: {suite}",
            "buildUrl": "N/A",
            "reportUrl": "N/A",
            "nodeLabels": [platform.system(), "Docker", "GPU"],
            "parameters": {
                "SUITE": suite,
                "BUILD_NUMBER": build_number
            }
        }
        os.makedirs(os.path.dirname(executor_path), exist_ok=True)
        with open(executor_path, 'w') as f:
            json.dump(executor_data, f, indent=4)
        
        print(f"  ✅ Generated dynamic: executor.json")
    except Exception as e:
        print(f"  ⚠️ Failed to generate executor.json: {e}")

def copy_allure_metadata(build_number: str):
    """Copies required static files for the Allure report and environment properties."""
    
    results_dir = os.path.join(ALLURE_RESULTS_DIR, build_number)
    
    # 1. Copy categories.json
    src_categories = os.path.join(SUPPORTS_DIR, "categories.json")
    dst_categories = os.path.join(results_dir, "categories.json")
    try:
        if os.path.exists(src_categories):
            shutil.copy(src_categories, dst_categories)
            print(f"  Copied: categories.json")
        else:
            pass 
    except Exception as e:
        print(f"  ⚠️ Failed to copy categories.json: {e}")
        
    # 2. Copy environment properties (platform-specific)
    try:
        os_name = "windows" if platform.system() == "Windows" else "linux"
        env_src = os.path.join(SUPPORTS_DIR, f"{os_name}.properties")
        env_dst = os.path.join(results_dir, "environment.properties")
        
        if os.path.exists(env_src):
            shutil.copy(env_src, env_dst)
            print(f"  Copied: {os_name}.properties → environment.properties")
        else:
            pass

    except Exception as e:
        print(f"  ⚠️ Failed to copy environment properties: {e}")


def run_docker_tests(build_number: str, suite: str):
    """Constructs and executes the corrected docker run command."""
    
    results_dir_name = str(build_number)
    
    # --- Conditional Device Mapping (Handles /dev/dri missing on Windows) ---
    device_flags = ""
    if platform.system() == "Windows":
        # Use /dev/dxg for Windows/WSL2/Intel GPU support
        print("  Detected Windows OS. Using /dev/dxg for GPU acceleration.")
        device_flags = "--device /dev/dxg:/dev/dxg"
    else:
        # Assume standard Linux/WSL1/WSL2 setups that might support /dev/dri
        print("  Detected non-Windows OS. Using /dev/dri for GPU acceleration.")
        device_flags = "--device /dev/dri:/dev/dri"
    
    # Allure results volume mapping (Absolute path required for Windows mounts)
    volume_flag = f"-v {os.path.abspath(ALLURE_RESULTS_DIR)}:/app/allure-results"
    
    # Explicitly call Python and the pytest module runner
    pytest_command = f"{CONTAINER_PYTHON_BIN} -m pytest --alluredir=/app/allure-results/{results_dir_name} -m {suite}"
    
    # Add optional flags for pytest
    optional_flags = " --ignore=features/manual_tests"

    docker_test_command = (
        f"docker run --rm {device_flags} {volume_flag} {LOCAL_IMAGE_TAG} "
        f"{pytest_command} {optional_flags}"
    )

    print("\n--- Step 4: Running Docker Tests ---")
    print(f"Executing: {docker_test_command}")
    
    # Execute the command with error handling
    rc = execute_command(
        docker_test_command, 
        "Docker Test Run failed.", 
        shell=True,
        exit_on_error=False
    )
    return rc


def main():
    if len(sys.argv) < 3:
        print("Usage: python run_kubernestes.py <BUILD_NUMBER> <suite>")
        sys.exit(1)

    build_number = sys.argv[1].strip()
    suite = sys.argv[2].strip()

    print("=" * 50)
    print(f"Running GPU Benchmark Test Workflow for Build #{build_number} | Suite: {suite}")
    print("=" * 50)

    # Step 0: Docker check
    print("\n--- Checking Docker Daemon Status ---")
    check_docker_running()
    
    # Step 1: Prepare Workspace
    print("\n--- Step 1: Prepare Workspace and History ---")
    run_results_dir = os.path.join(ALLURE_RESULTS_DIR, build_number)
    shutil.rmtree(run_results_dir, ignore_errors=True)
    os.makedirs(run_results_dir, exist_ok=True)
    
    # Copy previous history for Allure trend
    prev_history = os.path.join(ALLURE_RESULTS_DIR, "latest", "history")
    if os.path.exists(prev_history):
        shutil.copytree(prev_history, os.path.join(run_results_dir, "history"), dirs_exist_ok=True)
        print("Seeded previous history for trend.")
    else:
        print("No previous history found; trends will start fresh.")


    # Step 2: Docker image (Will force rebuild if missing)
    print("\n--- Step 2: Checking Local Docker Image ---")
    ensure_image_built(LOCAL_IMAGE_TAG)

    # Step 3: Allure Metadata
    print("\n--- Step 3: Preparing Allure Metadata ---")
    copy_allure_metadata(build_number)
    write_executor_json(build_number, suite)

    # Step 4: Run Docker Tests
    rc = run_docker_tests(build_number, suite)

    if rc != 0:
        print("\n❌ FAIL POLICY: Test execution failed. Stopping.")
        sys.exit(rc)
    
    # --- Step 5: Generate Report ---
    print("\n--- Step 5: Generating Allure Report ---")
    allure_bin = shutil.which("allure") or shutil.which("allure.cmd")
    if allure_bin:
        execute_command(
            f"{allure_bin} generate {run_results_dir} -o {ALLURE_REPORT_DIR} --clean",
            "Allure report generation failed."
        )
        print(f"✅ Report generated to {ALLURE_REPORT_DIR}")
    else:
        print("⚠️ Allure CLI not found on host. Skipping report generation.", file=sys.stderr)
        
    # --- Step 6: Save Latest History ---
    print("\n--- Step 6: Saving Latest History ---")
    latest_dir = os.path.join(ALLURE_RESULTS_DIR, "latest")
    shutil.rmtree(latest_dir, ignore_errors=True)
    shutil.copytree(run_results_dir, latest_dir, dirs_exist_ok=True)
    print("✅ Saved results to 'latest' directory.")
    
    # --- Step 7: Open Allure Report Locally ---
    print("\n--- Step 7: Opening Allure Report Locally ---")
    allure_bin = shutil.which("allure") or shutil.which("allure.cmd")
    if allure_bin:
        try:
            # Using Popen to run in background
            subprocess.Popen([allure_bin, "open", ALLURE_REPORT_DIR], creationflags=CREATE_NEW_PROCESS_GROUP)
            print(f"🚀 Opening via Allure CLI.")
        except Exception as e:
            print(f"⚠️ CLI open failed: {e}")
            webbrowser.open(os.path.join(ALLURE_REPORT_DIR, "index.html"))
    else:
        print("⚠️ Cannot open report: Allure CLI is not available on host.")
        
    print("\n" + "=" * 50)
    print(f"Build #{build_number} ({suite}) Complete (RC: 0)")
    print("=" * 50)
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user (Ctrl+C). Exiting.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        # Final catch for any unexpected errors
        print(f"\nFATAL UNHANDLED SCRIPT ERROR: {e}", file=sys.stderr)
        sys.exit(1)
