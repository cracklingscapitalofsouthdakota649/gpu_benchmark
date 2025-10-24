# run_docker.py
import os
import sys
import shutil
import subprocess
import time
import platform
import json
import psutil  # pip install psutil
import datetime
import signal

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGE_NAME = "gpu-benchmark:latest"

# Results & reports
ALLURE_RESULTS_DIR = os.path.join(PROJECT_ROOT, "allure-results")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
ALLURE_REPORT_DIR = os.path.join(REPORTS_DIR, "allure-report")

# Optional metadata folder
SUPPORTS_DIR = os.path.join(PROJECT_ROOT, "supports")

# Python version / venv (used for checking benchmark install inside the container)
PYTHON_BIN = "/usr/bin/python3.10"

# FIX: Windows Process Group Creation Flag for robust Ctrl+C handling
# This flag isolates the child process, allowing the parent (this script)
# to catch Ctrl+C and then explicitly terminate the child process group.
CREATE_NEW_PROCESS_GROUP = 0x00000200 if platform.system() == "Windows" else 0

# --- OpenCL Import ---
try:
    import pyopencl as cl # For host-side Intel GPU detection
except ImportError:
    cl = None

# Placeholder for trend functions (assuming they exist in supports/performance_trend.py)
try:
    from supports.performance_trend import update_performance_trend, plot_trend_chart
except ImportError:
    def update_performance_trend(*args, **kwargs): return None
    def plot_trend_chart(*args, **kwargs): return None


# --------------------------
# Subprocess Execution (FIXED for Ctrl+C)
# --------------------------
def run_cmd(cmd, check=True, shell=False, suppress_output=False, timeout=None):
    """
    Run a command on host using Popen/wait for explicit KeyboardInterrupt handling.
    This provides the most reliable Ctrl+C behavior on Windows for blocking commands
    like 'docker build', ensuring the process tree is killed upon interrupt.
    """
    stdout_target = subprocess.PIPE if suppress_output else sys.stdout
    stderr_target = subprocess.PIPE if suppress_output else sys.stderr
    
    # Use Popen instead of run to gain control over the wait/interrupt sequence
    process = subprocess.Popen(
        cmd,
        text=True,
        stdout=stdout_target,
        stderr=stderr_target,
        shell=shell,
        creationflags=CREATE_NEW_PROCESS_GROUP
    )

    try:
        # Wait for the process to complete or be interrupted
        process.wait(timeout=timeout)
        
        # If output was captured (suppress_output=True), read it now
        stdout, stderr = (process.communicate() if suppress_output else (None, None))

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
                # Use os.killpg for POSIX process group handling
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
            except OSError:
                process.terminate()

        # Re-raise the interrupt to stop the main script execution
        raise 
        
    except subprocess.TimeoutExpired as e:
        process.kill()
        if check:
            raise
        return 1, None, f"Command timed out after {e.timeout} seconds."

    # Handle errors after the process has finished naturally
    if check and process.returncode != 0:
        if suppress_output:
            # Re-read the output for the error message
            if not stdout or not stderr:
                 stdout, stderr = process.communicate()
            if stdout: print(stdout, end='')
            if stderr: print(stderr, end='', file=sys.stderr)
            
        raise subprocess.CalledProcessError(process.returncode, cmd, stdout, stderr)

    return process.returncode, stdout, stderr

# --------------------------
# Docker Management
# --------------------------
def check_docker_running():
    """Check Docker daemon is running."""
    rc, _, _ = run_cmd(["docker", "info"], check=False, suppress_output=True)
    if rc == 0:
        print("✅ Docker is running and responsive.")
    else:
        print("❌ Docker is not running. Start Docker and retry.")
        sys.exit(1)


def ensure_image_built(image_name: str, context_dir: str = PROJECT_ROOT, dockerfile: str = "Dockerfile"):
    """Ensure Docker image exists, build if missing."""
    # Check if image exists
    result = subprocess.run(["docker", "images", "-q", image_name], capture_output=True, text=True)
    if result.stdout.strip():
        print(f"✅ Found image {image_name}. Skipping build.")
        return
    print(f"⚠️ Image '{image_name}' not found. Building...")
    
    # Use run_cmd for robust Ctrl+C handling during the potentially long build process
    run_cmd(
        ["docker", "build", "-t", image_name, "-f", dockerfile, context_dir], 
        check=True
    )
    print(f"✅ Successfully built image {image_name}")


def container_has_pytest_benchmark():
    """Check if pytest-benchmark exists inside Docker."""
    rc, stdout, _ = run_cmd(
        ["docker", "run", "--rm", IMAGE_NAME, PYTHON_BIN, "-m", "pip", "show", "pytest-benchmark"],
        check=False, suppress_output=True
    )
    return rc == 0 and "Name: pytest-benchmark" in (stdout or "")

# --------------------------
# Hardware Info Detection
# --------------------------
def detect_cpu_info():
    """Detect CPU name on Windows/Linux."""
    try:
        if sys.platform.startswith("win"):
            _, stdout, _ = run_cmd(["wmic", "cpu", "get", "name", "/value"], check=False, suppress_output=True)
            if stdout:
                for line in stdout.splitlines():
                    if line.startswith("Name="):
                        return line.split('=')[-1].strip()
                return "Unknown CPU (WMIC failure)"
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
    """Detect GPU info using nvidia-smi and pyopencl (for Intel/AMD)."""
    vendor, name = "None", "None"
    
    # 1. NVIDIA check
    try:
        _, stdout, _ = run_cmd(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False, suppress_output=True, timeout=2
        )
        if stdout and stdout.strip():
            return "NVIDIA", stdout.strip().split("\n")[0]
    except Exception:
        pass

    # 2. OpenCL check (Intel/AMD)
    if cl:
        try:
            for p in cl.get_platforms():
                for d in p.get_devices():
                    if d.type & cl.device_type.GPU:
                        vendor = d.vendor.strip()
                        name = d.name.strip()
                        if "Intel" in vendor or "INTEL" in vendor:
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
    """Writes system hardware info to Allure's environment.properties file."""
    dest_dir = os.path.join(ALLURE_RESULTS_DIR, str(build_number))
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


def prepare_allure_metadata(run_results_dir, build_number):
    """Copy environment and categories metadata + executor.json"""
    print("--- Step 3: Preparing Allure Metadata ---")
    
    # 1. Seed history
    prev_history = os.path.join(ALLURE_RESULTS_DIR, "latest", "history")
    dest_history = os.path.join(run_results_dir, "history")
    if os.path.exists(prev_history):
        shutil.copytree(prev_history, dest_history, dirs_exist_ok=True)
        print("Seeded previous history for trend.")
    else:
        os.makedirs(dest_history, exist_ok=True)
        print("No previous history found; trend will start fresh.")

    # 2. Write system info
    write_environment_properties(build_number)

    # 3. Copy categories
    src_path = os.path.join(SUPPORTS_DIR, "categories.json")
    dest_path = os.path.join(run_results_dir, "categories.json")
    if os.path.exists(src_path):
        shutil.copy2(src_path, dest_path)
        
    # 4. executor.json 
    try:
        git_branch = subprocess.getoutput("git rev-parse --abbrev-ref HEAD")
        git_commit = subprocess.getoutput("git rev-parse --short HEAD")
        executor = {
            "name": "GPU Benchmark Runner",
            "type": "Local",
            "buildOrder": build_number,
            "buildName": f"GPU Benchmark Build #{build_number} - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "data": {
                "Git Branch": git_branch.strip(),
                "Git Commit": git_commit.strip(),
                "OS": platform.system(),
                "Python": platform.python_version(),
                "Docker Image": IMAGE_NAME,
            },
        }
        with open(os.path.join(run_results_dir, "executor.json"), "w", encoding="utf-8") as f:
            json.dump(executor, f, indent=2)
        print("Wrote executor.json")
    except Exception as e:
        print(f"Warning: Failed to write executor.json: {e}", file=sys.stderr)


def generate_allure_report(results_dir):
    """Generate Allure report."""
    os.makedirs(ALLURE_REPORT_DIR, exist_ok=True)
    allure_bin = shutil.which("allure") or shutil.which("allure.cmd")
    if not allure_bin:
        print("[CRITICAL] Allure CLI not found. Install it via Scoop, npm, or download manually.", file=sys.stderr)
        return False
        
    # Use run_cmd for consistent flag application
    rc, _, _ = run_cmd(
        [allure_bin, "generate", results_dir, "-o", ALLURE_REPORT_DIR, "--clean"], 
        check=False
    )
    if rc == 0:
        print(f"✅ Report generated to {ALLURE_REPORT_DIR}")
        return True
    return False


def open_allure_report():
    allure_bin = shutil.which("allure") or shutil.which("allure.cmd")
    if allure_bin:
        try:
            # Use Popen to run in background, applying process group flag
            subprocess.Popen([allure_bin, "open", ALLURE_REPORT_DIR], creationflags=CREATE_NEW_PROCESS_GROUP) 
            print(f"🚀 Attempting to open report in default browser: {ALLURE_REPORT_DIR}/index.html")
        except Exception as e:
            print(f"[WARN] Failed to open Allure report: {e}", file=sys.stderr)
    else:
        print("[WARN] Cannot open report: Allure CLI is not available.", file=sys.stderr)


def save_latest(build_number):
    """Save history and latest results for Allure trend."""
    print("\n--- Saving Latest History for Trend Analysis ---")
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
        print("[WARN] Generated report history not found; trend may be incomplete.")


def run_docker_tests(suite: str, run_results_dir: str):
    """Run Docker tests with robust Ctrl+C handling."""
    has_benchmark = container_has_pytest_benchmark()
    results_sub_dir = os.path.basename(run_results_dir)

    docker_test_command = [
        "docker", "run", "--rm",
        # Use absolute path for mounting on Windows
        "-v", f"{os.path.abspath(ALLURE_RESULTS_DIR)}:/app/allure-results",
        IMAGE_NAME,
        "pytest",
        f"--alluredir=/app/allure-results/{results_sub_dir}",
        "-m", suite
    ]

    if has_benchmark:
        docker_test_command += [
            f"--benchmark-json=/app/allure-results/{results_sub_dir}/benchmark.json",
            "--benchmark-save=latest",
            "--benchmark-autosave",
            "--benchmark-min-rounds=5",
        ]

    print(f"--- Running Docker tests for suite '{suite}' ---")
    
    # Popen here is for the same reason as run_cmd: explicit Ctrl+C handling
    process = subprocess.Popen(docker_test_command, creationflags=CREATE_NEW_PROCESS_GROUP)
    
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n[WARN] Ctrl+C detected — terminating Docker container/process group...", file=sys.stderr)
        if platform.system() == "Windows":
             # Terminate the entire process group reliably on Windows
             subprocess.call(
                 ['taskkill', '/F', '/T', '/PID', str(process.pid)],
                 creationflags=CREATE_NEW_PROCESS_GROUP,
                 stdout=subprocess.DEVNULL,
                 stderr=subprocess.DEVNULL 
             )
        else:
            # Standard Linux/Unix signal handling
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
            except OSError:
                process.terminate()

        process.wait() # Wait for termination
        return 130 

    return process.returncode


if __name__ == "__main__":
    try:
        os.chdir(PROJECT_ROOT)

        if len(sys.argv) < 2:
            print("Usage: python run_docker.py <BUILD_NUMBER> [suite]")
            sys.exit(1)

        build_number = sys.argv[1].strip()
        suite = sys.argv[2].strip() if len(sys.argv) > 2 else "preprocess"
        rc = 0 

        print(f"=== GPU Benchmark Build #{build_number}, Suite: {suite} ===")

        # Step 0: Docker check
        check_docker_running()

        # Step 1: Workspace
        shutil.rmtree(os.path.join(ALLURE_RESULTS_DIR, build_number), ignore_errors=True)
        run_results_dir = os.path.join(ALLURE_RESULTS_DIR, build_number)
        os.makedirs(run_results_dir, exist_ok=True)

        # Step 2: Docker image (Uses the fixed run_cmd)
        ensure_image_built(IMAGE_NAME)

        # Step 3: Metadata
        prepare_allure_metadata(run_results_dir, build_number)

        # Step 4: Docker tests
        rc = run_docker_tests(suite, run_results_dir)

        # --- Step 5: Report Generation ---
        report_success = generate_allure_report(run_results_dir)

        if report_success:
            # Performance Trend 
            trend = update_performance_trend(
                build_number,
                results_dir=ALLURE_RESULTS_DIR,
                history_dir=os.path.join(ALLURE_REPORT_DIR, "history")
            )
            if trend:
                chart = plot_trend_chart(trend, os.path.join(ALLURE_RESULTS_DIR, str(build_number)), build_number)
                if chart and os.path.exists(chart):
                    print(f"[INFO] Trend chart generated: {chart}")
            
            # Save latest
            save_latest(build_number)
            
            # Open report
            open_allure_report()

        print(f"\n=== GPU Benchmark Build #{build_number} ({suite}) Complete (RC: {rc}) ===")
        sys.exit(rc)
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user (Ctrl+C). Exiting.", file=sys.stderr)
        sys.exit(130) # Standard exit code for SIGINT
    except Exception as e:
        print(f"\nAn unhandled error occurred: {e}", file=sys.stderr)
        sys.exit(1)