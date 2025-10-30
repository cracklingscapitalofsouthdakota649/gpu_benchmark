#!/usr/bin/env python3
# =============================================================================
# 📁 Filename: run_kubernestes.py
# -----------------------------------------------------------------------------
# 🎯 Purpose:
# End-to-End Kubernetes Orchestration Pipeline for Dockerized Test Execution &
# Allure Report Generation. Automates the full build → test → report → publish
# workflow for robotics-bdd or gpu-benchmark frameworks.
#
# 🧭 Goals:
# 1️⃣ Consistent Execution - identical test runs across environments.
# 2️⃣ Robust Reporting - detailed Allure HTML report with build & system metadata.
# 3️⃣ Automation - streamline Docker build, test execution, and report packaging.
#
# 🧑‍💻 Author: Bang Thien Nguyen
# 📧 Email: ontario1998@gmail.com
#
# -----------------------------------------------------------------------------
# 💡 Usage:
#   python run_kubernestes.py -h
#   python run_kubernestes.py <build_number> <framework> [test_arg] [dockerfile]
#
#   Required:
#     <build_number>   Unique integer build number (e.g., 101)
#     <framework>      robotics-bdd | robotics-tdd | gpu-benchmark
#
#   Optional:
#     [test_arg]       Target test/suite to run:
#                      robotics-bdd  → pytest marker (walking, pick, "navigation or pic", etc.)
#                      robotics-tdd  → tests/test_*.py | pytest marker (walking, pick, "navigation or pick", etc.)
#                      gpu-benchmark → test file (tests/test_*.py) | marker (gpu/cpu)
#                      Default: tests/test_data_preprocessing.py
#     [dockerfile]     Dockerfile to use (default: Dockerfile.mini)
#
#   Flags:
#     -h, --help       Show this help message and exit
# =============================================================================

import sys
import argparse
import textwrap

# Early help handler (before main imports)
if "-h" in sys.argv or "--help" in sys.argv:
    print(textwrap.dedent("""
    ======================================================================================================
    End-to-End Kubernetes Orchestration Pipeline for Dockerized Test Execution & Allure Report Generation.
    Author: Bang Thien Nguyen - ontario1998@gmail.com
    ======================================================================================================
    
    Usage:
      python run_kubernestes.py -h
      python run_kubernestes.py <build_number> <framework> [test_arg] [dockerfile]

    Arguments:
      Required <build_number>:  Unique integer build number for tagging (e.g., '101').
      Required <framework>:     robotics-bdd | robotics-tdd | gpu-benchmark
      Optional [test_arg]:      Target test/suite to run.
                                (1) robotics-bdd  → marker (walking, pick, 'navigation or pick', etc.)
                                    - Default: navigation
                                (2) robotics-tdd  → tests/test_*.py | marker (walking, pick, 'navigation or pick', etc.)
                                    - Default: navigation
                                (3) gpu-benchmark → tests/test_*.py | marker (gpu, cpu, nvidia, benchmark, etc.)
                                    - Default: tests/test_data_preprocessing.py
      Optional [dockerfile]:    Dockerfile to use (default: Dockerfile.mini)
      -h, --help:               Show this help message and exit.

    Example:
      python run_kubernestes.py 101 robotics-bdd walking
      python run_kubernestes.py 104 robotics-tdd tests/test_real_actions.py
      python run_kubernestes.py 202 gpu-benchmark tests/test_cpu_benchmark.py
    """))
    sys.exit(0)

# Continue normal imports
import subprocess, os, platform, shutil, json, time, webbrowser, re, psutil
from typing import Optional, Tuple

try:
    import pyopencl as cl
except ImportError:
    cl = None

# -----------------------------------------------------------------------------
# 🧩 Core constants and initial validation
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DOCKER_USER = os.getenv('DOCKER_USER')
if not DOCKER_USER:
    print("\n❌ CRITICAL: DOCKER_USER is not set. Please set environment variables before running:")
    print("   set DOCKER_USER=mydocker_username && set DOCKER_PASS=mydocker_password")
    sys.exit(1)

LOCAL_IMAGE_TAG = None
REPORT_IMAGE_TAG = None
IMAGE_ID_FILE = "python_image_id.tmp"

ALLURE_RESULTS_DIR = os.path.join(PROJECT_ROOT, "allure-results")
ALLURE_REPORT_DIR = os.path.join(PROJECT_ROOT, "allure-report")
SUPPORTS_DIR = os.path.join(PROJECT_ROOT, "supports")

# Regex for Docker build/push progress parsing
STEP_PROGRESS_RE = re.compile(r'\[(\d+)/(\d+)]')
STEP_DESC_RE = re.compile(r'-> BUILD INFO: #\d+ \[.*] (.*)')
DOCKER_PUSH_PROGRESS_RE = re.compile(
    r'([\da-f]+): (Waiting|Downloading|Extracting|Pushing|Pushed|Mounted|Layer already exists)\s+(?:\[.*]\s*(\d+)%)?'
)

CREATE_NEW_PROCESS_GROUP = 0x00000200 if sys.platform.startswith("win") else 0

# -----------------------------------------------------------------------------
# 🛠 Utility & Hardware Detection Functions
# -----------------------------------------------------------------------------
def get_command_output(command: list) -> Tuple[int, str, str]:
    """Run a command quietly and return (exit_code, stdout, stderr)."""
    try:
        result = subprocess.run(
            command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, creationflags=CREATE_NEW_PROCESS_GROUP if sys.platform.startswith("win") else 0
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)

def detect_cpu_info() -> str:
    """Detect and return CPU model name."""
    try:
        if sys.platform.startswith("win"):
            res = subprocess.run(["wmic", "cpu", "get", "name", "/value"], capture_output=True, text=True)
            for line in res.stdout.splitlines():
                if "Name=" in line:
                    return line.split("=")[1].strip()
        elif sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        return platform.processor() or "Unknown CPU"
    except Exception:
        return "Unknown CPU"

def detect_memory_info() -> str:
    """Return total system memory in GB."""
    try:
        return f"{round(psutil.virtual_memory().total / (1024**3), 2)} GB"
    except Exception:
        return "Unknown Memory"

def detect_gpu_info() -> Tuple[str, str]:
    """Detect and return (vendor, name) of the GPU if available."""
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=2)
        if res.returncode == 0 and res.stdout.strip():
            return "NVIDIA", res.stdout.strip().split("\n")[0]
    except Exception:
        pass
    if cl:
        try:
            for p in cl.get_platforms():
                for d in p.get_devices():
                    if d.type & cl.device_type.GPU:
                        vendor = d.vendor.strip()
                        name = d.name.strip()
                        return vendor.split()[0], name
        except Exception:
            pass
    return "None", "None"

# --- END HARDWARE DETECTION FUNCTIONS ---

def set_global_tags(framework_name: str):
    """Sets the dynamic Docker tags based on the DOCKER_USER and framework."""
    global LOCAL_IMAGE_TAG, REPORT_IMAGE_TAG
    # Use lowercase and replace underscores with hyphens for clean Docker tagging
    framework_norm = framework_name.lower().replace('_', '-')
    LOCAL_IMAGE_TAG = f"{DOCKER_USER}/{framework_norm}-local:latest"
    REPORT_IMAGE_TAG = f"{DOCKER_USER}/{framework_norm}-report"
    print(f"Set LOCAL_IMAGE_TAG: {LOCAL_IMAGE_TAG}")
    print(f"Set REPORT_IMAGE_TAG: {REPORT_IMAGE_TAG}")
    
    if not LOCAL_IMAGE_TAG or not REPORT_IMAGE_TAG:
        print("\nERROR: Failed to set global image tags. Exiting.")
        sys.exit(1)

def execute_command(command, error_message, check_output=False, exit_on_error=True, docker_build_status=False, docker_push_status=False):
    """
    Executes a shell command and handles errors, with streaming status for Docker operations.
    Implements PASS/UNSTABLE/FAIL policy for test runs (when exit_on_error=False).
    """
    if docker_build_status or docker_push_status:
        # --- Streaming Logic for Docker Build/Push ---
        if docker_build_status:
            try:
                # The original code used a complex way to get the image name from a string command
                # We'll use a simpler approach based on the known tag structure
                image_name = REPORT_IMAGE_TAG.split('/')[1].split(':')[0]
            except IndexError:
                image_name = "Docker Image"
            print(f"Starting Docker Build with Live Status: {image_name}")
            
        p = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            universal_newlines=True,
            bufsize=1 
        )
        
        current_step = 0
        total_steps = 0
        step_description = "Initializing..."
        layer_statuses = {} 
        return_code = None
        
        for line in iter(p.stdout.readline, ''):
            
            if docker_build_status:
                match_progress = STEP_PROGRESS_RE.search(line)
                match_desc = STEP_DESC_RE.search(line)

                if match_progress:
                    current_step = int(match_progress.group(1))
                    total_steps = int(match_progress.group(2))
                    
                    if match_desc:
                        step_description = match_desc.group(1).split('\n')[0].strip()
                        if step_description.startswith('FROM'):
                             step_description = f"FROM {step_description.split(':')[1].strip()}"
                        elif len(step_description) > 50:
                             step_description = step_description[:50] + "..."

                if total_steps > 0:
                    progress_percent = int((current_step / total_steps) * 100)
                    status_line = (
                        f"  [Docker Build Status] Step {current_step}/{total_steps} ({progress_percent}%) | "
                        f"Task: {step_description:<50} | "
                        f"{time.strftime('%H:%M:%S')} \r"
                    )
                    sys.stdout.write(status_line)
                    sys.stdout.flush()

            elif docker_push_status:
                match_push_progress = DOCKER_PUSH_PROGRESS_RE.search(line)
                
                if match_push_progress:
                    layer_id = match_push_progress.group(1)
                    status = match_push_progress.group(2)
                    percent_str = match_push_progress.group(3)
                    percent = int(percent_str) if percent_str else (100 if status in ('Pushed', 'Layer already exists', 'Mounted') else 0)
                    
                    layer_statuses[layer_id] = percent
                    
                    total_layers = len(layer_statuses)
                    if total_layers > 0:
                        active_layers = [p for p in layer_statuses.values() if p < 100]
                        total_units_possible = total_layers * 100
                        total_units_achieved = sum(layer_statuses.values())
                        overall_percent = int((total_units_achieved / total_units_possible) * 100)
                        
                        status_line = (
                            f"  [Docker Push Status] Total Progress: {overall_percent}% "
                            f"| Layers: {len(active_layers)} active / {total_layers} total | "
                            f"{time.strftime('%H:%M:%S')} \r"
                        )
                        sys.stdout.write(status_line)
                        sys.stdout.flush()

            if "ERROR" in line.upper() or "FATAL" in line.upper() or "STEP COMPLETE:" in line or "Login Succeeded" in line:
                 sys.stdout.write(" " * 120 + "\r")
                 print(line.strip())

        p.stdout.close()
        return_code = p.wait()

        sys.stdout.write(" " * 120 + "\r")
        sys.stdout.flush()

        if return_code != 0:
            print("\n==========================================================")
            print(f"ERROR UNHANDLED ERROR during Docker process: {error_message}")
            print(f"Command failed: {command}")
            print("==========================================================")
            if exit_on_error:
                sys.exit(return_code)
            return return_code
        
        if docker_build_status:
            target_tag = REPORT_IMAGE_TAG if "Dockerfile.report" in command else LOCAL_IMAGE_TAG
            print(f"✅ Docker build completed successfully: {target_tag}")
            
        return 0
    else:
        # --- Standard subprocess.run for non-streaming commands (like pytest) ---
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                universal_newlines=True,
            )
            output = result.stdout.strip()
            
            print(output) 
            return 0
        except subprocess.CalledProcessError as e:
            
            # --- PASS/UNSTABLE/FAIL Policy ---
            if not exit_on_error:
                if e.returncode == 1:
                    print(f"\n⚠️  UNSTABLE: Tests failed (exit code 1). Proceeding to report generation.")
                    print("----------------------------------------------------------")
                    print("----------------------------------------------------------")
                    return e.returncode
                else:
                    print("\n==========================================================")
                    print(f"❌ FAIL: Test execution failed with setup/environment error (exit code {e.returncode}).")
                    print(f"Command: {command}")
                    print("----------------------------------------------------------")
                    print(f"Output:\n{e.stdout}")
                    print("\nHalting pipeline. No report will be generated.")
                    print("==========================================================")
                    sys.exit(e.returncode)
            
            # Standard Error Block
            print("\n==========================================================")
            print(f"ERROR: {error_message}")
            print(f"Command failed: {command}")
            print("----------------------------------------------------------")
            print(f"Output:\n{e.stdout}")
            print("==========================================================")
            sys.exit(1)

def docker_image_exists(image_tag):
    """Checks if a Docker image with the given tag exists locally."""
    print(f"Checking for local image: {image_tag}")
    try:
        subprocess.run(
            f"docker image inspect {image_tag}",
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError:
        return False

def check_dependencies():
    """Verifies that essential command-line tools are installed."""
    print("--- Step 1: Checking Dependencies ---\n")
    dependencies = ["docker", "pytest", "allure"]
    missing = []
    
    for dep in dependencies:
        if shutil.which(dep) is None:
            missing.append(dep)
            
    if missing:
        print("ERROR: The following dependencies are missing:")
        for dep in missing:
            print(f"- {dep}")
        print("\nPlease install the missing dependencies (e.g., Docker, pytest, 'allure-commandline').")
        sys.exit(1)
        
    print("✅ All dependencies found (docker, pytest, allure).")
    return 0

def validate_and_get_test_args(framework_name, suite_marker, testfile):
    """Validates suite/testfile based on framework and returns the pytest command part."""
    
    # Normalize to use hyphens for validation logic
    framework_name = framework_name.lower().replace('_', '-')
    
    if framework_name == "robotics-bdd":
        CORRECT_SUITES = ["navigation", "reverse", "pick", "walking", "safety", "stand", "ground", "all"]
        # Validation
        # The strict validation check is commented out to allow complex pytest marker expressions (e.g., "navigation or pick").
        # if suite_marker not in CORRECT_SUITES:
        #     print(f"ERROR: Invalid suite marker for robotics-bdd: '{suite_marker}'")
        #     print(f"  Valid suites: {', '.join(CORRECT_SUITES)}")
        #     sys.exit(1)
        # Command Construction
        # FIX: The suite_marker argument is now enclosed in **double quotes** to ensure proper
        # cross-platform shell parsing of boolean expressions like "navigation or pick or walking"
        return f"pytest -m \"{suite_marker}\" --ignore=features/manual_tests --alluredir={{CONTAINER_ALLURE_RESULTS_DIR}}"
        
    elif framework_name == "gpu-benchmark":
        if testfile:
            # Check 1: Validate file name pattern
            if testfile.startswith("tests/test_") and testfile.endswith(".py"):
                 # Check 2: Validate file existence
                 full_path = os.path.join(PROJECT_ROOT, testfile)
                 if not os.path.exists(full_path):
                     print("\n==========================================================")
                     print(f"❌ ERROR: Test file not found for gpu-benchmark: '{testfile}'")
                     print(f"   Expected absolute path: {full_path}")
                     print("   Please ensure the file exists relative to the script's root directory.")
                     print("==========================================================")
                     sys.exit(1)
                 # Passed all checks
                 pass 
            else:
                 print(f"ERROR: Invalid test file format for gpu-benchmark: '{testfile}'")
                 print("  Valid test files must match the pattern: tests/test_*.py")
                 sys.exit(1)
            # Command Construction
            return f"pytest {testfile} --alluredir={{CONTAINER_ALLURE_RESULTS_DIR}}"
        
        elif suite_marker:
            CORRECT_SUITES = ["gpu", "cpu", "benchmark"]
            if suite_marker not in CORRECT_SUITES:
                print(f"ERROR: Invalid suite marker for gpu-benchmark: '{suite_marker}'")
                print(f"  Valid suites: {', '.join(CORRECT_SUITES)}")
                sys.exit(1)
            # Command Construction
            # FIX: Use double quotes for the suite marker here too.
            return f"pytest -m \"{suite_marker}\" --alluredir={{CONTAINER_ALLURE_RESULTS_DIR}}"
        
        else:
            print("ERROR: No test suite or test file specified for gpu-benchmark.")
            sys.exit(1)
            
    else:
        print(f"ERROR: Unsupported framework name: '{framework_name}'")
        sys.exit(1)


def run_tests(framework_name, suite_marker, testfile, dockerfile):
    """Runs the Tests inside the Docker container."""
    
    # Get the validated pytest command string
    pytest_cmd_suffix = validate_and_get_test_args(framework_name, suite_marker, testfile)
    
    print(f"\n--- Step 4: Running Tests (Framework: {framework_name}) ---")
    
    if os.path.exists(ALLURE_RESULTS_DIR):
        shutil.rmtree(ALLURE_RESULTS_DIR)
        
    os.makedirs(ALLURE_RESULTS_DIR)
    
    CONTAINER_ALLURE_RESULTS_DIR = "/app/allure-results" 
    
    # Replace the placeholder in the command
    final_pytest_cmd = pytest_cmd_suffix.replace("{CONTAINER_ALLURE_RESULTS_DIR}", CONTAINER_ALLURE_RESULTS_DIR)
    
    # Check for conftest bypass condition
    framework_norm = framework_name.lower().replace('_', '-')
    
    if framework_norm == "gpu-benchmark" and dockerfile == "Dockerfile.mini":
        print("INFO: Detected gpu-benchmark with Dockerfile.mini. Applying conftest bypass logic.")
        
        # Ensures correct shell quoting for Windows compatibility
        container_execution_command = (
            f'sh -c "if [ -f /app/tests/conftest.py ]; then mv /app/tests/conftest.py /app/tests/conftest.bak; fi; '
            f'{final_pytest_cmd} ; '
            f'test_exit_code=$?; ' 
            f'if [ -f /app/tests/conftest.bak ]; then mv /app/tests/conftest.bak /app/tests/conftest.py; fi; '
            f'exit $test_exit_code"'
        )
    else:
        # Note: final_pytest_cmd already contains the necessary double-quotes around the marker.
        container_execution_command = final_pytest_cmd

    docker_run_command = (
        f"docker run --rm "
        f"-v \"{ALLURE_RESULTS_DIR}\":{CONTAINER_ALLURE_RESULTS_DIR} "
        f"-v \"{SUPPORTS_DIR}\":/app/supports "
        f"{LOCAL_IMAGE_TAG} " 
        f"{container_execution_command}"
    )
    
    print(f"Executing: {docker_run_command}")
    
    test_exit_code = execute_command(
        docker_run_command, 
        "Test execution failed.",
        exit_on_error=False
    )
    
    if test_exit_code == 0:
        print("✅ PASS: All tests passed.")
        
    print("✅ Test run finished. Results saved to allure-results.")


def get_docker_hub_url(tag):
    """Generates the Docker Hub URL for an image tag."""
    parts = tag.split('/')
    if len(parts) < 2:
        return None 
    
    repo = parts[-1].split(':')[0]
    user = parts[-2]
    
    return f"https://hub.docker.com/r/{user}/{repo}/tags"

def publish_image_tags(image_tag_list, error_artifact_name):
    """Handles Docker login and pushes a list of image tags to Docker Hub."""
    print(f"\n--- Publishing {error_artifact_name} to Docker Hub (if credentials exist) ---")
    
    docker_user = os.getenv("DOCKER_USER")
    docker_pass = os.getenv("DOCKER_PASS")
    
    if not (docker_user and docker_pass):
        print("⚠️ Docker credentials not found.")
        print("   Please set environment variables DOCKER_USER and DOCKER_PASS if publishing is required.")
        print("   Skipping Docker Hub push.")
        return

    print("Logging in to Docker Hub...")
    login_result = execute_command(
        f"echo {docker_pass} | docker login -u {docker_user} --password-stdin",
        "Docker login failed.",
        exit_on_error=False
    )
    if login_result != 0:
        return 
    
    all_successful = True
    for tag in image_tag_list:
        repo_url = get_docker_hub_url(tag)
        
        print(f"--- Pushing tag: {tag} to {repo_url} ---")
        
        push_command = f"docker push {tag}"
        push_result = execute_command(
            push_command, 
            f"Failed to push {tag}. Check connection and image existence.",
            docker_push_status=True,
            exit_on_error=False 
        )
        if push_result != 0:
            all_successful = False
        else:
            print(f"✅ Push of {tag} completed.") 

    if all_successful:
        print(f"✅ All tags for {error_artifact_name} published successfully.")
    else:
        print(f"⚠️ Warning: One or more tags for {error_artifact_name} failed to publish.")


def generate_report(build_number, suite_marker, cpu_info: str, gpu_vendor: str, gpu_name: str, memory_info: str):
    """Generates the Allure HTML report, adds metadata, and packages it into a Docker image."""
    print("\n--- Step 5: Generating Allure Report and Packaging ---")
    
    DOCKER_HUB_USER_FOR_LINKS = f"{DOCKER_USER}"
    base_repo_name = REPORT_IMAGE_TAG.split('/')[1].split(':')[0] 
    base_framework_name = base_repo_name.split('-')[0].upper() 
    REPORT_REPO_BASE_URL = f"https://hub.docker.com/r/{DOCKER_HUB_USER_FOR_LINKS}/{base_repo_name}"
    
    # 5.1. Creating Allure executor.json for build metadata
    print("  5.1. Creating Allure executor.json for build metadata...")
    try:
        executor_data = {
            "name": f"{base_framework_name} Pipeline Runner", 
            "type": "Local_Execution",
            "url": f"{REPORT_REPO_BASE_URL}/tags",
            "reportUrl": f"{REPORT_REPO_BASE_URL}/tags?build={build_number}",
            "buildName": f"Build #{build_number} ({suite_marker.upper()} suite)",
            "buildUrl": f"{REPORT_REPO_BASE_URL}/tags?build={build_number}",
            "buildOrder": int(build_number)  
        }
        with open(os.path.join(ALLURE_RESULTS_DIR, "executor.json"), "w") as f:
            json.dump(executor_data, f, indent=4)
        print(f"  ✅ executor.json created for Build #{build_number}.")
    except ValueError:
        print("⚠️ WARNING: Could not set buildOrder. Ensure build_number is a numeric string.")
    except Exception as e:
        print(f"⚠️ WARNING: Failed to create executor.json: {e}")

    # 5.2. Create environment.properties for report title and environment section
    print("  5.2. Creating Allure environment.properties for report details...")
    try:
        environment_data = [
            f"Report Title={base_framework_name}: {suite_marker.upper()} Suite Run #{build_number}", 
            f"Platform={platform.system()} {platform.release()}",
            f"Test Suite Marker={suite_marker}",
            # --- START: Requested order change (GPU on top of CPU) ---
            f"GPU_Model={gpu_name}", 
            f"CPU_Model={cpu_info}",
            # --- END: Requested order change ---
            f"System_Memory={memory_info}",
            f"Docker User={DOCKER_USER}" 
        ]
        with open(os.path.join(ALLURE_RESULTS_DIR, "environment.properties"), "w") as f:
            f.write('\n'.join(environment_data) + '\n')
        print("  ✅ environment.properties created.")
    except Exception as e:
        print(f"⚠️ WARNING: Failed to create environment.properties: {e}")

    # 5.3. History setup
    history_source = os.path.join(ALLURE_REPORT_DIR, "history")
    history_destination = os.path.join(ALLURE_RESULTS_DIR, "history")
    if os.path.exists(history_source):
        try:
            shutil.copytree(history_source, history_destination)
            print("  ✅ Copied previous report history.")
        except Exception as e:
            print(f"⚠️ WARNING: Could not copy history files: {e}")
    else:
        print("  ℹ️ Previous report history not found. Starting new history.")

    # --- START: Copy categories.json for Allure categorization ---
    print("  5.3. Copying custom categorization file...")
    categories_source_path = os.path.join(SUPPORTS_DIR, "categories.json")
    categories_dest_path = os.path.join(ALLURE_RESULTS_DIR, "categories.json")
    
    if os.path.exists(categories_source_path):
        try:
            shutil.copy(categories_source_path, categories_dest_path)
            print("  ✅ Copied categories.json to allure-results for report generation.")
        except Exception as e:
            print(f"⚠️ WARNING: Failed to copy categories.json: {e}")
    else:
        print("  ℹ️ 'supports/categories.json' not found. Skipping custom categorization.")
    # --- END: Copy categories.json ---

    if os.path.exists(ALLURE_REPORT_DIR):
        shutil.rmtree(ALLURE_REPORT_DIR)
    
    # 5.4 Allure Report Generation
    print("  5.4. Generate Allure Report...")
    allure_generate_command = f"allure generate {ALLURE_RESULTS_DIR} --clean -o {ALLURE_REPORT_DIR}"
    execute_command(
        allure_generate_command, 
        "Allure report generation failed."
    )
    print("✅ Allure HTML report generated.")

    if not os.path.isdir(os.path.join(ALLURE_REPORT_DIR, "data")):
        print("\n==========================================================")
        print("ERROR CONTENT ERROR: New Allure report failed to generate content.")
        print(f"The directory '{ALLURE_REPORT_DIR}' is missing the 'data' folder.")
        print("Old report data likely persisted or generation failed. ABORTING PUSH.")
        print("==========================================================")
        sys.exit(1)

    # --- 5.5. Package Report into Docker Image ---
    print("  5.5. Packaging Allure Report into a Deployable Docker Image")
    
    REPORT_TAG = f"{REPORT_IMAGE_TAG}:{build_number}"
    REPORT_LATEST_TAG = f"{REPORT_IMAGE_TAG}:latest"
    
    report_dockerfile_content = f"""
FROM nginx:alpine
COPY {os.path.basename(ALLURE_REPORT_DIR)} /usr/share/nginx/html
EXPOSE 8081
CMD ["nginx", "-g", "daemon off;"]
"""
    dockerfile_path = os.path.join(PROJECT_ROOT, "Dockerfile.report")
    with open(dockerfile_path, "w") as f:
        f.write(report_dockerfile_content)
        
    print(f"  Dockerfile.report created for tag {REPORT_TAG}.")
    
    docker_build_report_command = f"docker build -t {REPORT_TAG} -f {dockerfile_path} ."
    execute_command(
        docker_build_report_command, 
        f"Failed to build report Docker image {REPORT_TAG}",
        docker_build_status=True
    )
    
    docker_tag_command = f"docker tag {REPORT_TAG} {REPORT_LATEST_TAG}"
    execute_command(
        docker_tag_command,
        f"Failed to tag image {REPORT_TAG} as {REPORT_LATEST_TAG}"
    )
    
    print(f"  ✅ Report image tagged as {REPORT_TAG} and {REPORT_LATEST_TAG}.")
    
    return REPORT_TAG, REPORT_LATEST_TAG


def open_report():
    """Opens the Allure report index.html in the default web browser."""
    print("\n--- Step 7: Opening Allure Report Locally ---")
    index_file = os.path.join(ALLURE_REPORT_DIR, "index.html")
    allure_bin = shutil.which("allure") or shutil.which("allure.cmd")
    if allure_bin:
        try:
            subprocess.Popen([allure_bin, "open", ALLURE_REPORT_DIR])
            print(f"🚀 Opening via Allure CLI at: {index_file}")
        except Exception as e:
            print(f"⚠️ CLI open failed: {e}")
            webbrowser.open_new_tab(index_file)
            print(f"🚀 Opening directly in browser at: {index_file}")
    else:
        webbrowser.open_new_tab(index_file)
        print(f"🚀 Opening directly in browser at: {index_file}")
        
def full_pipeline(build_number, framework_name, suite_marker, testfile, dockerfile, cpu_info: str, gpu_vendor: str, gpu_name: str, memory_info: str):
    """Runs the full pipeline."""
    # 1. Set global tags based on framework
    set_global_tags(framework_name)
    
    # This step will exit if dependencies are missing
    check_dependencies()
    
    # --- Step 2: Build Main Docker Image ---
    print("\n--- Step 2: Building Main Docker Image ---\n")
    if docker_image_exists(LOCAL_IMAGE_TAG):
        print(f"Image {LOCAL_IMAGE_TAG} already exists locally. Skipping build.")
        docker_tag_command = f"docker tag {LOCAL_IMAGE_TAG} {LOCAL_IMAGE_TAG}"
        execute_command(docker_tag_command, "Failed to re-tag existing image.")
    else:
        DOCKER_BUILD_COMMAND = f"docker build -t {LOCAL_IMAGE_TAG} -f {dockerfile} ."
        print(f"Local image not found. Starting build using {dockerfile}...")
        execute_command(
            DOCKER_BUILD_COMMAND, 
            f"Failed to build Docker image {LOCAL_IMAGE_TAG}",
            docker_build_status=True
        )

    # --- Step 3: Publish Main Image ---
    publish_image_tags([LOCAL_IMAGE_TAG], "Main Image")

    # --- Step 4: Run Tests ---
    run_tests(framework_name, suite_marker, testfile, dockerfile) 

    # --- Step 5: Generate and Package Report ---
    # This step is only reached if tests are PASS or UNSTABLE
    report_suite_marker = suite_marker or (testfile.split('/')[-1] if testfile else "default")
    
    REPORT_VERSION_TAG, REPORT_LATEST_TAG = generate_report(
        build_number, report_suite_marker, cpu_info, gpu_vendor, gpu_name, memory_info
    )

    # --- Step 6: Publish Report Image ---
    publish_image_tags([REPORT_VERSION_TAG, REPORT_LATEST_TAG], "Allure Report Image")

    # --- Step 7: Open Report ---
    open_report()

# -----------------------------------------------------------------------------
# 📦 Main Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Handle incorrect usage
    if len(sys.argv) < 3:
        print("Usage: python run_kubernestes.py <build_number> <framework> [test_arg] [dockerfile]")
        print("Run 'python run_kubernestes.py -h' for detailed help.")
        sys.exit(1)

    build_number = sys.argv[1]
    framework_name = sys.argv[2]
    
    optional_test_arg = sys.argv[3] if len(sys.argv) > 3 else None
    dockerfile = sys.argv[4] if len(sys.argv) > 4 else "Dockerfile.mini"

    # --- NEW VALIDATION: Check for valid FRAMEWORK_NAME ---
    SUPPORTED_FRAMEWORKS = ["robotics-bdd", "gpu-benchmark"]
    framework_norm = framework_name.lower().replace('_', '-')

    if framework_norm not in [f.lower().replace('_', '-') for f in SUPPORTED_FRAMEWORKS]:
        print("\n==========================================================")
        print(f"❌ ERROR: Invalid or unsupported FRAMEWORK_NAME: '{framework_name}'")
        print(f"   The second argument must be one of: {', '.join(SUPPORTED_FRAMEWORKS)}")
        print("   It appears you provided a test file path in the FRAMEWORK_NAME position.")
        print("==========================================================")
        sys.exit(1)
    # --- END NEW VALIDATION ---


    # --- Set defaults for Test Arg before printing ---
    suite_marker = None
    testfile = None
    
    # framework_norm is now guaranteed to be valid
    test_arg_display = "None"
    
    if framework_norm == 'robotics-bdd':
        # Set default 'navigation' if no 3rd arg is provided
        suite_marker = optional_test_arg or "navigation"
        test_arg_display = suite_marker
        if optional_test_arg and (optional_test_arg.startswith("tests/") or optional_test_arg.endswith(".py")):
             print(f"WARNING: Test file '{optional_test_arg}' is not a valid argument for 'robotics-bdd'. Treating as suite marker.")
        
    elif framework_norm == 'gpu-benchmark':
        if optional_test_arg:
            # Test files must be of the form tests/test_*.py
            if optional_test_arg.startswith("tests/test_") and optional_test_arg.endswith(".py"):
                 testfile = optional_test_arg
            else:
                 suite_marker = optional_test_arg
        else:
            # Set default test file if no 3rd arg is provided
            testfile = "tests/test_data_preprocessing.py"
        test_arg_display = suite_marker or testfile

    if not build_number.isdigit():
        print("\n==========================================================")
        print(f"ERROR: The <BUILD_NUMBER> argument must be an integer.")
        print(f"Received: '{build_number}'")
        print("==========================================================")
        sys.exit(1)
        
    # --- Call detection functions here to display results ---
    cpu_info = detect_cpu_info()
    memory_info = detect_memory_info()
    gpu_vendor, gpu_name = detect_gpu_info()
    # --- END NEW ---

    print(f"=======================================================")
    print(f"STARTING ORCHESTRATION PIPELINE")
    print(f"Build Number: {build_number}")
    print(f"Framework:    {framework_name}")
    
    # --- UPDATED print statement to include hardware info ---
    print(f"CPU:          {cpu_info}")
    print(f"GPU:          {gpu_name} ({gpu_vendor})")
    print(f"Memory:       {memory_info}")
    # --- END UPDATED print statement ---
    
    print(f"Test Arg:     {test_arg_display}") 
    print(f"Dockerfile:   {dockerfile}")
    print(f"=======================================================")
    
    try:
        # --- UPDATED: Pass hardware info to full_pipeline ---
        full_pipeline(build_number, framework_name, suite_marker, testfile, dockerfile, cpu_info, gpu_vendor, gpu_name, memory_info)
    except KeyboardInterrupt:
        print("\n\n============================================")
        print(" 🛑 PIPELINE MANUALLY TERMINATED (Ctrl+C).")
        print("============================================")