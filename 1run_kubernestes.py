# run_kubernestes.py
import sys
import subprocess
import os
import platform
import shutil
import json
import time
import webbrowser
import re
import signal

# ==============================
# Configuration & Constants
# ==============================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Required for tagging and pushing
DOCKER_USER = os.getenv('DOCKER_USER')

# --- Start Explicit Error Handling for DOCKER_USER ---
if not DOCKER_USER:
    print("\n==========================================================")
    print("CRITICAL ENVIRONMENT ERROR: DOCKER_USER is not set.")
    print("The pipeline requires the 'DOCKER_USER' environment variable")
    print("to be configured (e.g., in your env or launcher script) to tag/push images.")
    print("==========================================================")
    sys.exit(1)
# --- End Explicit Error Handling for DOCKER_USER ---

# Repo base name aligned with this framework (override via env if desired)
REPO_BASENAME = os.getenv("DOCKER_REPO_BASENAME", "gpu-benchmark")

# Defaults requested
DEFAULT_DOCKERFILE = "Dockerfile.mini"
DEFAULT_TEST_FILE = "tests/test_data_preprocessing.py"

LOCAL_IMAGE_TAG = f"{DOCKER_USER}/{REPO_BASENAME}-local:latest"
REPORT_IMAGE_TAG_BASE = f"{DOCKER_USER}/{REPO_BASENAME}-report"

ALLURE_RESULTS_DIR = os.path.join(PROJECT_ROOT, "allure-results")
ALLURE_REPORT_DIR = os.path.join(PROJECT_ROOT, "allure-report")
SUPPORTS_DIR = os.path.join(PROJECT_ROOT, "supports")

# Persistent history cache to make Allure "Trend" survive clean workspaces
HISTORY_CACHE_DIR = os.path.join(PROJECT_ROOT, ".allure-history")

# Regex to capture the step progress: [CurrentStep/TotalSteps]
STEP_PROGRESS_RE = re.compile(r'\[(\d+)/(\d+)\]')
# Regex to capture a short description line
STEP_DESC_RE = re.compile(r'-> BUILD INFO: #\d+ \[(.*)\]')
# Regex for docker push/pull progress (best-effort parse)
DOCKER_PUSH_PROGRESS_RE = re.compile(
    r'([0-9a-f]+):\s+(Waiting|Downloading|Extracting|Pushing|Pushed|Mounted|Layer already exists)\s*(?:\[(\d+)%\])?'
)

# ------------------------------
# Status/Helper Utilities
# ------------------------------
_LAST_STATUS_LEN = 0
def _status_print(msg: str):
    """Overwrite status line dynamically (works on Windows CMD)."""
    global _LAST_STATUS_LEN
    msg = msg.replace("\r", " ").replace("\n", " ")
    padded = msg.ljust(max(_LAST_STATUS_LEN, len(msg)))
    sys.stdout.write(f"\r{padded}")
    sys.stdout.flush()
    _LAST_STATUS_LEN = len(msg)

def _status_clear():
    """Clears the dynamically printed status line."""
    global _LAST_STATUS_LEN
    if _LAST_STATUS_LEN:
        sys.stdout.write("\r" + " " * _LAST_STATUS_LEN + "\r")
        sys.stdout.flush()
        _LAST_STATUS_LEN = 0


# ------------------------------
# Small helpers
# ------------------------------
def q(path: str) -> str:
    """Shell-quote a path (good enough for our use here)."""
    return f"\"{path}\""

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _copy_tree(src: str, dst: str):
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def _is_test_path_or_file(selector: str) -> bool:
    """Returns True if selector looks like a test path/file instead of a pytest marker."""
    if not selector:
        return True
    s = selector.strip().replace("\\", "/")
    return s.endswith(".py") or s.startswith("tests/") or s.startswith("./tests/")


# ------------------------------
# CTRL-C HANDLER
# ------------------------------
_current_process = None
def _sigint_handler(sig, frame):
    global _current_process
    _status_clear()
    print("\n🛑 Ctrl-C detected — stopping pipeline...")
    if _current_process and _current_process.poll() is None:
        try:
            if os.name == "nt":
                subprocess.call(f"taskkill /F /T /PID {_current_process.pid}", shell=True)
            else:
                _current_process.terminate()
        except Exception:
            pass
    print("✅ Graceful shutdown complete.")
    sys.exit(130)
signal.signal(signal.SIGINT, _sigint_handler)


# ------------------------------
# Command execution
# ------------------------------
def execute_command(
    command: str,
    error_message: str,
    check_output: bool = False,
    exit_on_error: bool = True,
    docker_build_status: bool = False,
    docker_push_status: bool = False,
):
    """Executes a shell command and handles errors, with streaming status for Docker operations."""
    if docker_build_status or docker_push_status:
        if docker_build_status:
            print(f"Starting Docker Build with Live Status...")
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

        for line in iter(p.stdout.readline, ''):
            if docker_build_status:
                match_progress = STEP_PROGRESS_RE.search(line)
                match_desc = STEP_DESC_RE.search(line)
                if match_progress:
                    current_step = int(match_progress.group(1))
                    total_steps = int(match_progress.group(2))
                if match_desc:
                    step_description = match_desc.group(1).split('\n')[0].strip()
                    if len(step_description) > 60:
                        step_description = step_description[:60] + "..."
                if total_steps > 0:
                    progress_percent = int((current_step / total_steps) * 100)
                    status_line = (
                        f" [Docker Build Status] Step {current_step}/{total_steps} ({progress_percent}%) "
                        f"Task: {step_description:<55} "
                        f"{time.strftime('%H:%M:%S')}"
                    )
                    _status_print(status_line)
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
                        total_units_possible = total_layers * 100
                        total_units_achieved = sum(layer_statuses.values())
                        overall_percent = int((total_units_achieved / total_units_possible) * 100)
                        status_line = (
                            f" [Docker Push Status] Total Progress: {overall_percent}% "
                            f"Layers: {sum(1 for pcent in layer_statuses.values() if pcent < 100)} active / {total_layers} total "
                            f"{time.strftime('%H:%M:%S')}"
                        )
                        _status_print(status_line)
            if "ERROR" in line.upper() or "FATAL" in line.upper() or "Login Succeeded" in line:
                _status_clear()
                print(line.strip())

        p.stdout.close()
        return_code = p.wait()
        _status_clear()

        if return_code != 0:
            print("\n==========================================================")
            print(f"FATAL UNHANDLED ERROR during Docker process: {error_message}")
            print(f"Command failed: {command}")
            print("==========================================================")
            if exit_on_error:
                sys.exit(return_code)
            return return_code

        if docker_build_status:
            print("✅ Docker build completed successfully.")
        return 0

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
        if check_output:
            return output
        if output:
            print(output)
        return 0
    except subprocess.CalledProcessError as e:
        print("\n==========================================================")
        print(f"FATAL UNHANDLED ERROR during command execution: {error_message}")
        print(f"Command failed: {command}")
        print("----------------------------------------------------------")
        print(f"Output:\n{e.stdout}")
        print("==========================================================")
        if exit_on_error:
            sys.exit(1)
        return 1
    except KeyboardInterrupt:
        _sigint_handler(None, None)


# ------------------------------
# Test execution
# ------------------------------
def run_tests(suite_marker: str, dockerfile_path: str):
    print(f"\n--- Step 4: Running Tests (Suite: {suite_marker}) ---")
    if os.path.exists(ALLURE_RESULTS_DIR):
        shutil.rmtree(ALLURE_RESULTS_DIR)
    os.makedirs(ALLURE_RESULTS_DIR, exist_ok=True)

    CONTAINER_ALLURE_RESULTS_DIR = "/app/allure-results"
    volume_mounts = f"-v {q(ALLURE_RESULTS_DIR)}:{CONTAINER_ALLURE_RESULTS_DIR} -v {q(SUPPORTS_DIR)}:/app/supports "
    print("  ℹ️ Mounting 'supports' directory.")

    if dockerfile_path == DEFAULT_DOCKERFILE:
        target_test = suite_marker if suite_marker.endswith(".py") else DEFAULT_TEST_FILE
        print(f"  ⚠️ Using Dockerfile.mini → running isolated test: {target_test}")
        docker_run_command = (
            f"docker run --rm {volume_mounts}"
            f"{LOCAL_IMAGE_TAG} "
            f'bash -c "if [ -f /app/tests/conftest.py ]; then mv /app/tests/conftest.py /app/tests/conftest.bak; fi && '
            f'pytest /app/{target_test} --ignore=supports/*.py '
            f'--alluredir={CONTAINER_ALLURE_RESULTS_DIR} && '
            f'if [ -f /app/tests/conftest.bak ]; then mv /app/tests/conftest.bak /app/tests/conftest.py; fi"'
        )
    else:
        print(f"  ℹ️ Using custom Dockerfile: {dockerfile_path} (standard pytest discovery)")
        docker_run_command = (
            f"docker run --rm {volume_mounts}"
            f"{LOCAL_IMAGE_TAG} "
            f"pytest /app/tests -m {suite_marker} --ignore=features/manual_tests "
            f"--alluredir={CONTAINER_ALLURE_RESULTS_DIR}"
        )

    print(f"Executing: {docker_run_command}")
    execute_command(
        docker_run_command,
        "Test execution failed. Check test logs above.",
        exit_on_error=False  # Allow allure generation even if tests fail
    )
    print("✅ Tests completed and results saved to allure-results.")


# ------------------------------
# Report generation & packaging
# ------------------------------
def generate_report(build_number: str, selector: str):
    print("\n--- Step 5: Generating Allure Report and Packaging ---")
    DOCKER_HUB_USER_FOR_LINKS = f"{DOCKER_USER}"
    REPORT_REPO_BASE_URL = f"https://hub.docker.com/r/{DOCKER_HUB_USER_FOR_LINKS}/{REPO_BASENAME}-report"
    try:
        os.makedirs(ALLURE_RESULTS_DIR, exist_ok=True)
        executor_data = {
            "name": f"{REPO_BASENAME.title()} Pipeline Runner",
            "type": "Local_Execution",
            "url": f"{REPORT_REPO_BASE_URL}/tags",
            "reportUrl": f"{REPORT_REPO_BASE_URL}/tags?build={build_number}",
            "buildName": f"Build #{build_number} ({selector})",
            "buildUrl": f"{REPORT_REPO_BASE_URL}/tags?build={build_number}",
            "buildOrder": int(build_number)
        }
        with open(os.path.join(ALLURE_RESULTS_DIR, "executor.json"), "w") as f:
            json.dump(executor_data, f, indent=4)
    except Exception:
        pass

    try:
        environment_data = [
            f"Report Title={REPO_BASENAME.title()}: {selector} Run #{build_number}",
            f"Docker User={DOCKER_USER}",
            f"Platform={platform.system()} {platform.release()}",
            f"Test Selector={selector}"
        ]
        with open(os.path.join(ALLURE_RESULTS_DIR, "environment.properties"), "w") as f:
            f.write('\n'.join(environment_data) + '\n')
    except Exception:
        pass

    results_history = os.path.join(ALLURE_RESULTS_DIR, "history")
    cache_history = HISTORY_CACHE_DIR
    report_history = os.path.join(ALLURE_REPORT_DIR, "history")
    history_src = None
    if os.path.isdir(cache_history) and os.listdir(cache_history):
        history_src = cache_history
    elif os.path.isdir(report_history) and os.listdir(report_history):
        history_src = report_history
    if history_src:
        try:
            _copy_tree(history_src, results_history)
        except Exception:
            pass

    if os.path.exists(ALLURE_REPORT_DIR):
        shutil.rmtree(ALLURE_REPORT_DIR)
    allure_generate_command = f"allure generate {q(ALLURE_RESULTS_DIR)} --clean -o {q(ALLURE_REPORT_DIR)}"
    execute_command(allure_generate_command, "Allure report generation failed.")
    try:
        new_history = os.path.join(ALLURE_REPORT_DIR, "history")
        if os.path.isdir(new_history) and os.listdir(new_history):
            _ensure_dir(HISTORY_CACHE_DIR)
            _copy_tree(new_history, HISTORY_CACHE_DIR)
    except Exception:
        pass

    dockerfile_content = """\
FROM nginx:alpine
COPY ./ /usr/share/nginx/html/
EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]    
"""
    dockerfile_path = os.path.join(PROJECT_ROOT, "Dockerfile.report")
    with open(dockerfile_path, "w") as f:
        f.write(dockerfile_content)
    report_tag_version = f"{REPORT_IMAGE_TAG_BASE}:{build_number}"
    report_tag_latest = f"{REPORT_IMAGE_TAG_BASE}:latest"
    docker_build_report_command = f"docker build -t {report_tag_version} -f {q(dockerfile_path)} {q(ALLURE_REPORT_DIR)}"
    execute_command(docker_build_report_command, f"Failed to build report Docker image {report_tag_version}", docker_build_status=True)
    docker_tag_command = f"docker tag {report_tag_version} {report_tag_latest}"
    execute_command(docker_tag_command, f"Failed to tag image {report_tag_version} as {report_tag_latest}")

    # --- 5.5. Trim Allure Trend History to Last 6 Builds ---
    try:
        history_file = os.path.join(HISTORY_CACHE_DIR, "history-trend.json")
        if os.path.exists(history_file):
            with open(history_file, "r+", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 6:
                    trimmed = data[-6:]
                    f.seek(0)
                    json.dump(trimmed, f, indent=4)
                    f.truncate()
                    print(f" ✅ Trimmed Allure trend to last 6 builds ({len(trimmed)} retained).")
    except Exception as e:
        print(f"⚠️ WARNING: Failed to trim Allure trend history: {e}")

    print(f" ✅ Report image tagged as {report_tag_version} and {report_tag_latest}.")
    return report_tag_version, report_tag_latest


# ------------------------------
# Docker Hub publishing
# ------------------------------
def get_docker_hub_url(tag: str):
    """Generates the Docker Hub URL for an image tag."""
    parts = tag.split('/')
    if len(parts) < 2:
        return None  # Not a standard user/repo format
    repo = parts[-1].split(':')[0]
    user = parts[-2]
    return f"https://hub.docker.com/r/{user}/{repo}/tags"


def publish_image_tags(image_tag_list, artifact_name: str):
    """
    Handles Docker login and pushes a list of image tags to Docker Hub.
    """
    print(f"\n--- Publishing {artifact_name} to Docker Hub (if credentials exist) ---")
    docker_user = os.getenv("DOCKER_USER")
    docker_pass = os.getenv("DOCKER_PASS")

    if not (docker_user and docker_pass):
        print("⚠️ Docker credentials not found.")
        print(" Please set environment variables DOCKER_USER and DOCKER_PASS if publishing is required.")
        print(" Skipping Docker Hub push.")
        return

    print("Logging in to Docker Hub...")
    # Note: if your password has special characters, consider using a file or env-safe quoting.
    login_result = execute_command(
        f"echo {docker_pass} | docker login -u {docker_user} --password-stdin",
        "Docker login failed.",
        exit_on_error=False
    )
    if login_result != 0:
        return  # Stop if login failed

    all_successful = True
    for tag in image_tag_list:
        repo_url = get_docker_hub_url(tag)
        print(f"--- Pushing tag: {tag} to {repo_url} ---")
        push_command = f"docker push {tag}"
        push_result = execute_command(
            push_command,
            f"Failed to push {tag}. Check connection and image existence.",
            docker_push_status=True,
            exit_on_error=False  # Continue even if one push fails
        )
        if push_result != 0:
            all_successful = False
        else:
            print(f"✅ Push of {tag} completed.")
    if all_successful:
        print(f"✅ All tags for {artifact_name} published successfully.")
    else:
        print(f"⚠️ Warning: One or more tags for {artifact_name} failed to publish.")


# ------------------------------
# Clean Up Artifacts
# ------------------------------
def cleanup_docker_artifacts():
    """
    Cleans up dangling images, build cache, and stopped containers using 'docker system prune -a',
    but preserves the gpu-benchmark-local:latest image.
    """
    print("\n--- Step 8: Final Cleanup (Pruning Docker Artifacts) ---")
    
    # Identify image to preserve
    preserve_image = f"{LOCAL_IMAGE_TAG}"
    print(f"Preserving essential image: {preserve_image}")

    # Get all image IDs except the preserved one
    try:
        result = subprocess.run(
            f"docker images -q | sort | uniq",
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            universal_newlines=True,
        )
        all_images = result.stdout.strip().splitlines()
        preserved_id = subprocess.run(
            f"docker images -q {preserve_image}",
            shell=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
            universal_newlines=True,
        ).stdout.strip()
        removable_images = [i for i in all_images if i and i != preserved_id]

        # Remove containers and unused resources
        subprocess.run("docker container prune -f", shell=True)
        subprocess.run("docker volume prune -f", shell=True)
        subprocess.run("docker builder prune -f", shell=True)

        # Remove all other images explicitly
        for img in removable_images:
            subprocess.run(f"docker rmi -f {img}", shell=True)

        print(f"✅ Docker cleanup successful. Preserved: {preserve_image}")
    except Exception as e:
        print(f"⚠️ WARNING: Docker cleanup failed or partial. Error: {e}")


# ------------------------------
# Open report locally
# ------------------------------
def open_report():
    """Opens the Allure report index.html in the default web browser."""
    print("\n--- Step 7: Opening Allure Report Locally ---")
    index_file = os.path.join(ALLURE_REPORT_DIR, "index.html")
    allure_bin = shutil.which("allure") or shutil.which("allure.cmd")
    if allure_bin:
        try:
            subprocess.Popen([allure_bin, "open", ALLURE_REPORT_DIR])
            print(f"🚀 Opening via Allure CLI at: {index_file}")
            return
        except Exception as e:
            print(f"⚠️ CLI open failed: {e}")
    webbrowser.open_new_tab(index_file)
    print(f"🚀 Opening directly in browser at: {index_file}")


# ------------------------------
# Full pipeline
# ------------------------------
def full_pipeline(build_number: str, suite_marker: str, dockerfile_path: str, dockerfile_was_explicit: bool):
    """Runs the full pipeline."""
    # Ensure the script runs cleanup even if the main pipeline fails
    try:
        check_dependencies()

        # --- Step 2: Build Main Docker Image ---
        print("\n--- Step 2: Building Main Docker Image ---\n")
        print(f"Using Dockerfile: {dockerfile_path}")
        # If Dockerfile is explicitly passed, ALWAYS rebuild.
        must_rebuild = dockerfile_was_explicit or (not docker_image_exists(LOCAL_IMAGE_TAG))
        dockerfile_abs = os.path.join(PROJECT_ROOT, dockerfile_path)
        if must_rebuild:
            build_cmd = f"docker build -t {LOCAL_IMAGE_TAG} -f {q(dockerfile_abs)} {q(PROJECT_ROOT)}"
            print("Building local image...")
            execute_command(
                build_cmd,
                f"Failed to build Docker image {LOCAL_IMAGE_TAG}",
                docker_build_status=True
            )
        else:
            print(f"Image {LOCAL_IMAGE_TAG} already exists locally. Skipping build (no explicit Dockerfile override).")
            # Re-tag (no-op) to ensure 'latest' is correct
            docker_tag_command = f"docker tag {LOCAL_IMAGE_TAG} {LOCAL_IMAGE_TAG}"
            execute_command(docker_tag_command, "Failed to re-tag existing image.")

        # --- Step 3: Publish Main Image ---
        publish_image_tags([LOCAL_IMAGE_TAG], "Main Image")

        # --- Step 4: Run Tests ---
        run_tests(suite_marker, dockerfile_path)

        # --- Step 5: Generate and Package Report ---
        report_version_tag, report_latest_tag = generate_report(build_number, suite_marker)

        # --- Step 6: Publish Report Image ---
        publish_image_tags([report_version_tag, report_latest_tag], "Allure Report Image")

        # --- Step 7: Open Report ---
        open_report()

    finally:
        # --- Step 8: Final Cleanup (Always runs) ---
        cleanup_docker_artifacts()


# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    def print_usage():
        print("Usage: python run_kubernestes.py <BUILD_NUMBER> [SUITE_MARKER or tests] [Dockerfile]")
        print("Example: python run_kubernestes.py 1 tests/test_data_preprocessing.py Dockerfile.custom")
        print("Defaults: Dockerfile=Dockerfile.mini, Test=tests/test_data_preprocessing.py")

    # Validate arguments
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    build_number_arg = sys.argv[1]
    if not build_number_arg.isdigit():
        print("\n==========================================================")
        print("FATAL ERROR: The <BUILD_NUMBER> argument must be an integer.")
        print(f"Received: '{build_number_arg}'")
        print_usage()
        print("==========================================================")
        sys.exit(1)

    # Defaults
    suite_marker_arg = DEFAULT_TEST_FILE
    dockerfile_arg = DEFAULT_DOCKERFILE
    dockerfile_explicit = False

    # Override defaults if provided
    if len(sys.argv) >= 3 and sys.argv[2].strip():
        suite_marker_arg = sys.argv[2].strip()
    if len(sys.argv) >= 4 and sys.argv[3].strip():
        dockerfile_arg = sys.argv[3].strip()
        dockerfile_explicit = True

    print("=======================================================")
    print("    STARTING GPU BENCHMARK ORCHESTRATION PIPELINE      ")
    print("    Author: Bang Thien Nguyen - ontario1998@gmail.com   ") 
    print(f"    Build Number : {build_number_arg}")
    print(f"    Target Suite : {suite_marker_arg}")
    print(f"    Dockerfile   : {dockerfile_arg}")
    print("=======================================================")

    full_pipeline(build_number_arg, suite_marker_arg, dockerfile_arg, dockerfile_explicit)