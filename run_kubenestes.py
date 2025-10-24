import sys
import subprocess
import os
import argparse
import platform
import shutil
import json
import time
import webbrowser

# Constants
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_IMAGE_TAG = "luckyjoy/gpu-benchmark-local:latest"
IMAGE_ID_FILE = "python_image_id.tmp"

ALLURE_RESULTS_DIR = os.path.join(PROJECT_ROOT, "allure-results")
ALLURE_REPORT_DIR = os.path.join(PROJECT_ROOT, "allure-report")
SUPPORTS_DIR = os.path.join(PROJECT_ROOT, "supports")


def execute_command(command, error_message, check_output=False, exit_on_error=True):
    """Executes a shell command and handles errors."""
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
        if check_output:
            return result.stdout.strip()
        print(result.stdout.strip())
        return 0
    except subprocess.CalledProcessError as e:
        print("\n==========================================================")
        print(f"FATAL UNHANDLED ERROR during command execution: {error_message}")
        print(f"Command failed: {command}")
        print(f"Return Code: {e.returncode}")
        print(f"Output: {e.output.strip()}")
        print("==========================================================")
        if exit_on_error:
            sys.exit(e.returncode)
        else:
            return e.returncode
    except FileNotFoundError:
        print("\n==========================================================")
        print(f"FATAL ERROR: Command not found. Is '{command.split()[0]}' installed and in PATH?")
        print("==========================================================")
        sys.exit(1)


def get_local_image_id():
    """Return image ID if it exists locally, else None (no fatal exit)."""
    try:
        result = subprocess.run(
            f"docker inspect --format='{{{{.Id}}}}' {LOCAL_IMAGE_TAG}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            universal_newlines=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None
    except Exception:
        return None


def build_image_if_missing():
    """Checks if the local Docker image exists and builds it if necessary."""
    print("\n--- Step 2: Checking Local Docker Image ---")
    local_id = get_local_image_id()

    if local_id:
        print(f"✅ Found existing Docker image: {LOCAL_IMAGE_TAG}")
        print(f"   Image ID: {local_id[:12]} (no rebuild needed)")
        return local_id

    print(f"⚠️ Docker image '{LOCAL_IMAGE_TAG}' not found. Starting build...")
    execute_command(f"docker build -t {LOCAL_IMAGE_TAG} .", "Docker build failed.")

    new_local_id = get_local_image_id()
    if new_local_id:
        print(f"✅ Successfully built Docker image: {LOCAL_IMAGE_TAG}")
        print(f"   New Image ID: {new_local_id[:12]}")
        return new_local_id
    else:
        print("❌ Build completed, but image ID could not be retrieved.")
        sys.exit(1)


def docker_path(path):
    """Convert host path to Docker-friendly format for both Windows and Unix."""
    path = os.path.abspath(path)
    if platform.system() == "Windows":
        drive, tail = os.path.splitdrive(path)
        path = f"/{drive[0].lower()}{tail.replace(os.sep, '/')}"
    else:
        path = path.replace(os.sep, '/')
    return path


def main():
    if platform.system() == "Windows":
        try:
            subprocess.run("chcp 65001", shell=True, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Run GPU Benchmarks via Docker.")
    parser.add_argument("build_number", type=int, help="Build identifier.")
    parser.add_argument("suite", type=str, nargs="?", default="all",
                        help="The pytest marker for the test suite.")
    args = parser.parse_args()

    print("==========================================================")
    print(f"Running GPU Benchmark Test Workflow for Build #{args.build_number} | Suite: {args.suite}")
    print("==========================================================")

    # --- Docker Daemon Check ---
    print("--- Checking Docker Daemon Status ---")
    execute_command("docker info", "Docker daemon is not running.", check_output=True)
    print("✅ Docker is running and responsive.")

    # --- Step 1: Prepare Workspace and History ---
    print("\n--- Step 1: Prepare Workspace and History ---")
    latest_history = os.path.join(ALLURE_REPORT_DIR, "history")
    history_dest = os.path.join(ALLURE_RESULTS_DIR, "history")

    shutil.rmtree(ALLURE_RESULTS_DIR, ignore_errors=True)
    shutil.rmtree(os.path.join(PROJECT_ROOT, "__pycache__"), ignore_errors=True)
    shutil.rmtree(os.path.join(PROJECT_ROOT, ".pytest_cache"), ignore_errors=True)
    os.makedirs(ALLURE_RESULTS_DIR, exist_ok=True)

    if os.path.exists(latest_history):
        print("Copying previous Allure history for trend data...")
        try:
            shutil.copytree(latest_history, history_dest, dirs_exist_ok=True)
        except Exception as e:
            print(f"⚠️ Failed to copy history folder: {e}")
    else:
        print("No previous history found; trends will start fresh.")

    # --- Step 2: Build Image Conditionally ---
    local_image_id = build_image_if_missing()
    with open(IMAGE_ID_FILE, "w", encoding="utf-8") as f:
        f.write(local_image_id.strip())

    # --- Step 3: Preparing Allure Metadata ---
    print("\n--- Step 3: Preparing Allure Metadata ---")

    system_os = platform.system()
    env_property_file = "windows.properties" if system_os == 'Windows' else "ubuntu.properties"
    print(f"Detected OS: {system_os}. Using {env_property_file} for Allure metadata.")

    # Copy environment and categories files
    metadata_files = [
        (env_property_file, "environment.properties"),
        ("categories.json", "categories.json"),
    ]

    for src_name, dest_name in metadata_files:
        src_path = os.path.join(SUPPORTS_DIR, src_name)
        dest_path = os.path.join(ALLURE_RESULTS_DIR, dest_name)
        try:
            shutil.copy2(src_path, dest_path)
            print(f"  Copied: {src_name} → {dest_name}")
        except FileNotFoundError:
            print(f"  ⚠️ Warning: Allure metadata file not found: {src_name}. Skipping.")

    # Generate dynamic executor.json
    print("  Generating dynamic executor.json...")
    try:
        executor_data = {
            "name": "Local GPU Benchmark Runner",
            "type": "Local_Execution",
            "buildOrder": args.build_number,
            "buildName": f"Local Run #{args.build_number}",
            "data": {
                "Test Framework": "Gherkin (Behave) / Pytest",
                "OS": platform.system(),
                "Python": platform.python_version(),
                "Docker Image": LOCAL_IMAGE_TAG
            }
        }

        dest_executor_path = os.path.join(ALLURE_RESULTS_DIR, "executor.json")
        with open(dest_executor_path, "w", encoding="utf-8") as f:
            json.dump(executor_data, f, indent=2)
        print(f"  ✅ Created executor.json at {os.path.basename(ALLURE_RESULTS_DIR)}/executor.json")
    except Exception as e:
        print(f"  ⚠️ Failed to generate executor.json: {e}")

    # --- Step 4: Running Docker Tests ---
    print("\n--- Step 4: Running Docker Tests ---")
    allure_results_path = docker_path(ALLURE_RESULTS_DIR)
    test_command = (
        f"docker run --rm "
        f"-v {allure_results_path}:/app/allure-results "
        f"{LOCAL_IMAGE_TAG} "
        f"pytest --alluredir=allure-results -m {args.suite} --ignore=features/manual_tests"
    )
    print(f"Executing: {test_command}")
    test_exit = execute_command(test_command, "Docker Test Run", exit_on_error=False)
    if test_exit not in (0, 1):
        print("❌ FAIL POLICY: Test execution failed. Stopping.")
        sys.exit(1)
    elif test_exit == 1:
        print("⚠️ Some tests failed; continuing to generate report.")
    else:
        print("✅ All tests passed; continuing to report generation.")

    # --- Step 5: Allure Report Generation (via Docker) ---
    print("\n--- Step 5: Allure Report Generation (via Docker) ---")
    allure_report_path = docker_path(ALLURE_REPORT_DIR)
    os.makedirs(ALLURE_REPORT_DIR, exist_ok=True)
    report_cmd = (
        f"docker run --rm "
        f"-v {allure_results_path}:/app/allure-results "
        f"-v {allure_report_path}:/app/allure-report "
        f"{LOCAL_IMAGE_TAG} "
        f"allure generate allure-results -o allure-report --clean"
    )
    print(f"Executing: {report_cmd}")
    execute_command(report_cmd, "Allure report generation failed.")

    # --- Step 6: Publish Docker Image of Report ---
    print("\n--- Step 6: Publishing Allure Report to Docker Hub ---")

    dockerfile_path = os.path.join(ALLURE_REPORT_DIR, "Dockerfile")
    dockerfile_content = (
        "FROM nginx:stable-alpine\n"
        "LABEL maintainer=\"CI Pipeline\"\n"
        f"ENV REPORT_BUILD_NUMBER={args.build_number}\n"
        "COPY . /usr/share/nginx/html\n"
        "EXPOSE 80\n"
        "CMD [\"nginx\", \"-g\", \"daemon off;\"]\n"
    )
    with open(dockerfile_path, "w", encoding="utf-8") as f:
        f.write(dockerfile_content)
    print(f"✅ Created Dockerfile inside: {dockerfile_path}")

    dockerhub_repo = "luckyjoy/gpu-benchmark-report"
    version_tag = f"{dockerhub_repo}:{args.build_number}"
    latest_tag = f"{dockerhub_repo}:latest"

    execute_command(f"docker build -t {version_tag} -t {latest_tag} {ALLURE_REPORT_DIR}",
                    "Failed to build Docker image for Allure report.")

    docker_user = os.getenv("DOCKER_USER")
    docker_pass = os.getenv("DOCKER_PASS")
    if docker_user and docker_pass:
        execute_command(
            f"echo {docker_pass} | docker login -u {docker_user} --password-stdin",
            "Docker login failed."
        )
        execute_command(f"docker push {version_tag}", f"Failed to push {version_tag}")
        execute_command(f"docker push {latest_tag}", f"Failed to push {latest_tag}")
        print(f"✅ Published to Docker Hub: {latest_tag}")
    else:
        print("⚠️ Docker credentials not found.")
        print("   Please set environment variables DOCKER_USER and DOCKER_PASS if publishing is required.")
        print("   Skipping Docker Hub push for this local run.")

    # --- Step 7: Open Allure Report Locally ---
    print("\n--- Step 7: Opening Allure Report Locally ---")
    index_file = os.path.join(ALLURE_REPORT_DIR, "index.html")
    allure_bin = shutil.which("allure") or shutil.which("allure.cmd")
    if allure_bin:
        try:
            subprocess.Popen([allure_bin, "open", ALLURE_REPORT_DIR])
            print(f"🚀 Opening via Allure CLI at: {index_file}")
        except Exception as e:
            print(f"⚠️ CLI open failed: {e}")
            webbrowser.open(f"file://{index_file}")
    else:
        webbrowser.open(f"file://{index_file}")
        print(f"Opened report in default browser: {index_file}")

    print("\n--- Workflow Complete ---")


if __name__ == "__main__":
    main()
