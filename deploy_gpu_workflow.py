# ==============================================================================
# FILENAME: deploy_gpu_workflow.py
#
# PURPOSE:
# This script orchestrates the deployment of a Dockerized application (specifically,
# an Allure test report server) onto a Kubernetes cluster. It handles resource
# cleanup, dynamic GPU detection for scheduling, service creation, and local
# port-forwarding for accessing the deployed application.
#
# GOALS:
# - Automate deployment to Kubernetes from a CI/CD pipeline or local environment.
# - Intelligently schedule workloads onto GPU-enabled nodes if available.
# - Provide a reliable fallback to CPU-only deployment.
# - Ensure a clean cluster state by removing resources before and after execution.
# - Offer easy local access to the deployed Allure report via port-forwarding.
#
# FEATURES:
# - Dynamic Framework Support: Adapts deployment name, image tag, and local port
#   based on the framework name provided (e.g., 'robotics-bdd', 'gpu-benchmark').
# - GPU Auto-Detection: Scans Kubernetes nodes for specified GPU resources
#   (Intel i915, NVIDIA) using the Kubernetes API.
# - CPU Fallback: If no suitable GPU is found, deploys using standard CPU/Memory limits.
# - Robust Cleanup: Deletes all associated Kubernetes resources (Deployments, Services, Pods)
#   before starting and after completion to prevent conflicts and resource leakage.
# - Port Forwarding: Automatically starts `kubectl port-forward` to map a local port
#   to the deployed service, opening the report in the default web browser.
# - Safe Docker Pruning: Includes an optional, non-aggressive Docker cleanup step
#   to remove build cache and stopped containers while preserving tagged local images.
# - Graceful Termination: Handles Ctrl+C (KeyboardInterrupt) during port-forwarding
#   and cleanup steps to exit cleanly.
# ==============================================================================
import os
import time
import subprocess
from typing import Optional, Dict, List
import sys
import webbrowser
import threading
import signal
from datetime import datetime, timezone

import kubernetes.client as client
from kubernetes.client.rest import ApiException
from kubernetes import config

# ------------------ CONFIGURATION & CONSTANTS ------------------
DOCKER_USER = os.getenv('DOCKER_USER')
if not DOCKER_USER:
    print("\n============================================================================")
    print("❌ CRITICAL: DOCKER_USER is not set.")
    print("Please set it before running this script:")
    print("Example: set DOCKER_USER=mydocker_username, set DOCKER_PASS=mydcoker_password")
    print("=============================================================================")
    sys.exit(1)

# Environment flags controlling GPU scheduling behavior
REQUIRE_GPU = os.getenv("REQUIRE_GPU", "false").lower() == "true"
STRICT_GPU = os.getenv("STRICT_GPU", "false").lower() == "true"
# List of GPU resource keys to look for on Kubernetes nodes
GPU_RESOURCE_KEYS = ["gpu.intel.com/i915", "nvidia.com/gpu"]

# K8S Client Instances (Initialized globally in load_kube_config)
apps_v1: Optional[client.AppsV1Api] = None
core_v1: Optional[client.CoreV1Api] = None

# Static K8s Configuration
NAMESPACE = "default"
CONTAINER_APP_PORT = 80
K8S_SERVICE_PORT = 80

# Console color codes
COLOR_GREEN = '\033[92m'
COLOR_RESET = '\033[0m'
COLOR_YELLOW = '\033[93m'
COLOR_BLUE = '\033[94m'

# Global state for port-forwarding process
PORT_FORWARD_PROCESS: Optional[subprocess.Popen] = None

# Dynamic variables determined in __main__
SERVICE_NAME: Optional[str] = None
DEPLOYMENT_NAME: Optional[str] = None
LOCAL_PORT: Optional[int] = None
IMAGE_NAME: Optional[str] = None

# Resource cleaning configuration
CLEANUP_RESOURCES = [
    ("Deployments", lambda apps, selector: apps.delete_collection_namespaced_deployment(
        namespace=NAMESPACE, label_selector=selector)),
    ("Services", lambda core, selector: core.delete_collection_namespaced_service(
        namespace=NAMESPACE, label_selector=selector)),
    ("Pods", lambda core, selector: core.delete_collection_namespaced_pod(
        namespace=NAMESPACE, label_selector=selector))
]

# ------------------ UTILITY FUNCTIONS ------------------

def _format_time_age(creation_timestamp: datetime) -> str:
    """
    Helper function to calculate and format the age of a Kubernetes resource.

    Feature: Provides human-readable age for pods/resources in the console output.
    """
    now_utc = datetime.now(timezone.utc)
    # Ensure creation_timestamp is timezone-aware for correct comparison
    if creation_timestamp.tzinfo is None:
        creation_aware = creation_timestamp.replace(tzinfo=timezone.utc)
    else:
        creation_aware = creation_timestamp

    age_delta = now_utc - creation_aware
    age_sec = age_delta.total_seconds()

    if age_sec < 60:
        return f"{int(age_sec)}s"
    if age_sec < 3600:
        return f"{int(age_sec/60)}m"
    if age_sec < 86400:
        return f"{int(age_sec/3600)}h"
    return f"{int(age_sec/86400)}d"


def load_kube_config():
    """
    Loads Kubernetes configuration from default locations (kubeconfig file or in-cluster).
    Initializes global Kubernetes API client instances (core_v1, apps_v1).

    Feature: Establishes connection to the target Kubernetes cluster.
    """
    global apps_v1, core_v1
    print("--- 1. Loading Kubernetes Configuration ---")
    try:
        config.load_kube_config()
        apps_v1 = client.AppsV1Api()
        core_v1 = client.CoreV1Api()
        print("✅ Kubernetes config loaded (local) and clients initialized.")
    except config.ConfigException as local_e:
        print("   -> Local kubeconfig not found. Trying in-cluster config...")
        try:
            config.load_incluster_config()
            apps_v1 = client.AppsV1Api()
            core_v1 = client.CoreV1Api()
            print("✅ Kubernetes config loaded (in-cluster) and clients initialized.")
        except config.ConfigException as cluster_e:
            print(f"❌ CRITICAL: Failed to load Kubernetes config: Local Error ({local_e}), Cluster Error ({cluster_e})")
            sys.exit(1)
    except Exception as e:
        print(f"❌ CRITICAL: Failed to load Kubernetes config: {e}")
        sys.exit(1)


def print_available_pods(core_v1_api: client.CoreV1Api, header: str, highlight_pod_name: Optional[str] = None):
    """
    Prints a formatted list of pods currently running in the target namespace.
    Optionally highlights a specific pod name.

    Feature: Provides visibility into the cluster state before and after deployment.
    """
    print(f"\n--- {header} Pods in Namespace '{NAMESPACE}' ---")
    try:
        pods = core_v1_api.list_namespaced_pod(namespace=NAMESPACE).items
        if not pods:
            print("  (No pods found.)")
            return
        
        print(f"  {'NAME':<50} {'STATUS':<15} {'AGE':<10}")
        print("  " + "="*75)
        
        for pod in pods:
            name = pod.metadata.name
            status = pod.status.phase or "Unknown"
            age_str = "N/A"
            
            if pod.metadata.creation_timestamp:
                age_str = _format_time_age(pod.metadata.creation_timestamp)

            output = f"  {name:<50} {status:<15} {age_str:<10}"
            
            if highlight_pod_name and name == highlight_pod_name:
                print(f"{COLOR_YELLOW}{output}{COLOR_RESET}")
            else:
                print(output)
                
    except ApiException as e:
        print(f"❌ Failed to list pods: {e.status} {e.reason}")
    except Exception as e:
        print(f"❌ Unexpected error listing pods: {e}")

def _free_local_port(port: int):
    """
    Detects and terminates any process currently using the specified local TCP port.
    Works on both Windows and Unix-like systems.

    Feature: Prevents 'Only one usage of each socket address' errors
             by ensuring the port is free before starting port-forwarding.
    """
    print(f"\n--- Checking if port {port} is already in use ---")
    try:
        if os.name == "nt":
            # Windows: use netstat + taskkill
            cmd_find = f'netstat -ano | findstr :{port}'
            output = subprocess.check_output(cmd_find, shell=True).decode(errors="ignore")
            pids = set()
            for line in output.splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[-1].isdigit():
                    pids.add(parts[-1])
            if pids:
                print(f"  ⚠️ Port {port} is in use by PIDs: {', '.join(pids)}. Attempting to terminate...")
                for pid in pids:
                    subprocess.run(f"taskkill /PID {pid} /F /T", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"  ✅ Freed port {port}.")
            else:
                print(f"  ✅ Port {port} is free.")
        else:
            # Linux/macOS
            cmd_find = ["lsof", "-t", f"-i:{port}"]
            output = subprocess.check_output(cmd_find).decode().strip()
            if output:
                pids = output.splitlines()
                print(f"  ⚠️ Port {port} is in use by PIDs: {', '.join(pids)}. Attempting to terminate...")
                for pid in pids:
                    subprocess.run(["kill", "-9", pid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"  ✅ Freed port {port}.")
            else:
                print(f"  ✅ Port {port} is free.")
    except subprocess.CalledProcessError:
        print(f"  ✅ Port {port} appears free.")
    except Exception as e:
        print(f"  ⚠️ Could not verify or free port {port}: {e}")


def clean_up_deployments(core_v1_api: client.CoreV1Api, apps_v1_api: client.AppsV1Api):
    """
    Deletes all existing Deployments, Services, and Pods in the target namespace
    that match the dynamically set DEPLOYMENT_NAME label. This ensures a clean
    slate for each run, crucial for CI environments.

    Feature: Robust, CI-friendly cleanup using label selectors. Prevents resource
             conflicts between concurrent runs.
    """
    if not core_v1_api or not apps_v1_api:
         print("❌ CRITICAL: Kubernetes clients not provided for cleanup.")
         sys.exit(1)

    print(f"\n--- 2. Cleaning up old Deployments, Services & Pods in '{NAMESPACE}' ---")

    # Define the label selector based on the dynamic deployment name
    label_selector = f"app={DEPLOYMENT_NAME}"
    cleanup_successful = True

    for resource_type, delete_func in CLEANUP_RESOURCES:
        try:
            print(f"  -> Deleting {resource_type} matching label '{label_selector}'...")
            # delete_func receives core_v1 or apps_v1 based on resource type
            api = apps_v1_api if resource_type == "Deployments" else core_v1_api
            delete_func(api, label_selector)
        except ApiException as e:
            # Ignore 404 Not Found errors, as resources might not exist
            if e.status != 404:
                print(f"  ⚠️ Warning during {resource_type} cleanup: {e.status} {e.reason}")
                cleanup_successful = False
        except Exception as e:
            print(f"  ❌ Unexpected error during {resource_type} cleanup: {e}")
            cleanup_successful = False

    if cleanup_successful:
        print("  -> Waiting briefly for resources to terminate...")
        time.sleep(5)
        print("✅ Clean-up completed.")
    else:
        print("⚠️ Clean-up finished with warnings.")


def find_available_gpu_resource_key(core_v1_api: client.CoreV1Api) -> Optional[str]:
    """
    Scans Kubernetes nodes for advertised GPU resources matching GPU_RESOURCE_KEYS.
    Returns the first matching key found, or None if no suitable GPU is available.
    Respects REQUIRE_GPU and STRICT_GPU environment variables.

    Feature: Automated GPU detection for intelligent workload scheduling.
    """
    print("\n--- 3. Detecting Available GPU Resources on Nodes ---")
    try:
        if not REQUIRE_GPU:
            print("  -> REQUIRE_GPU is false. Skipping GPU detection, defaulting to CPU.")
            return None

        nodes = core_v1_api.list_node().items
        for key in GPU_RESOURCE_KEYS:
            print(f"  -> Checking for GPU key: {key}")
            for node in nodes:
                # Check if the node has the capacity and allocatable amount for the GPU key
                if (key in node.status.capacity and int(node.status.capacity[key]) > 0 and
                    key in node.status.allocatable and int(node.status.allocatable[key]) > 0):
                    print(f"  ✅ Found available GPU '{key}' on node '{node.metadata.name}'")
                    return key

        # If loop completes without finding a GPU
        print("  ❌ No specified GPU resources found or available on any node.")
        if STRICT_GPU:
            print("  -> STRICT_GPU is true. Failing deployment.")
            sys.exit(1)
        else:
            print("  -> STRICT_GPU is false. Falling back to CPU.")
            return None

    except ApiException as e:
        print(f"❌ Error communicating with Kubernetes API during GPU detection: {e.status} {e.reason}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error during GPU detection: {e}")
        sys.exit(1)


def create_gpu_deployment(apps_v1_api: client.AppsV1Api, image_name: str, gpu_key: Optional[str]):
    """
    Creates the Kubernetes Deployment object. If a gpu_key is provided, it adds
    the corresponding resource limit to the container spec. Otherwise, it uses
    default CPU/Memory limits. Applies standard labels for cleanup.

    Feature: Creates the primary application workload (Allure report server)
             with appropriate resource requests (GPU or CPU).
    """
    print("\n--- 4. Creating Deployment ---")

    limits: Dict[str, str] = {}
    mode = "CPU"
    if gpu_key:
        limits = {gpu_key: "1"}
        mode = "GPU"
        print(f"  -> Configuring deployment for GPU mode with limit: {limits}")
    elif STRICT_GPU:
        print("❌ CRITICAL: STRICT_GPU=true but no GPU key provided. Exiting.")
        sys.exit(1)
    else:
        limits = {"cpu": "500m", "memory": "512Mi"}
        mode = "CPU"
        print(f"  -> Configuring deployment for CPU mode with limits: {limits}")

    # Define specs
    container = client.V1Container(
        name=f"{DEPLOYMENT_NAME}-container",
        image=image_name,
        ports=[client.V1ContainerPort(container_port=CONTAINER_APP_PORT)],
        resources=client.V1ResourceRequirements(limits=limits, requests=limits),
        image_pull_policy="IfNotPresent"
    )
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": DEPLOYMENT_NAME}),
        spec=client.V1PodSpec(containers=[container])
    )
    selector = client.V1LabelSelector(
        match_labels={"app": DEPLOYMENT_NAME}
    )
    spec = client.V1DeploymentSpec(
        replicas=1,
        template=template,
        selector=selector
    )
    deployment = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(
            name=DEPLOYMENT_NAME,
            labels={"app": DEPLOYMENT_NAME}
        ),
        spec=spec
    )

    # Create or replace
    try:
        apps_v1_api.create_namespaced_deployment(namespace=NAMESPACE, body=deployment)
        print(f"✅ Deployment '{DEPLOYMENT_NAME}' created successfully. Mode: {mode}")
    except ApiException as e:
        if e.status == 409: # Resource already exists
            print(f"  ℹ️ Deployment '{DEPLOYMENT_NAME}' already exists. Attempting to replace...")
            try:
                apps_v1_api.replace_namespaced_deployment(name=DEPLOYMENT_NAME, namespace=NAMESPACE, body=deployment)
                print(f"  ✅ Deployment '{DEPLOYMENT_NAME}' replaced successfully. Mode: {mode}")
            except ApiException as replace_e:
                print(f"  ❌ Failed to replace existing Deployment: {replace_e.status} {replace_e.reason}")
                sys.exit(1)
        else:
            print(f"❌ Failed to create Deployment: {e.status} {e.reason}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error creating Deployment: {e}")
        sys.exit(1)


def create_cluster_ip_service(core_v1_api: client.CoreV1Api):
    """
    Creates a Kubernetes ClusterIP Service to expose the Deployment's pods internally
    within the cluster on K8S_SERVICE_PORT, targeting the CONTAINER_APP_PORT.
    Applies standard labels for cleanup.

    Feature: Provides a stable internal network endpoint for the application pods,
             necessary for port-forwarding.
    """
    print("\n--- 5. Creating ClusterIP Service ---")

    service = client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(
            name=SERVICE_NAME,
            labels={"app": DEPLOYMENT_NAME}
        ),
        spec=client.V1ServiceSpec(
            selector={"app": DEPLOYMENT_NAME},
            ports=[client.V1ServicePort(
                protocol="TCP",
                port=K8S_SERVICE_PORT,
                target_port=CONTAINER_APP_PORT
            )],
            type="ClusterIP"
        )
    )

    try:
        core_v1_api.create_namespaced_service(namespace=NAMESPACE, body=service)
        print(f"✅ Service '{SERVICE_NAME}' created successfully.")
    except ApiException as e:
        if e.status == 409:
            print(f"  ℹ️ Service '{SERVICE_NAME}' already exists. Attempting to replace...")
            try:
                core_v1_api.replace_namespaced_service(name=SERVICE_NAME, namespace=NAMESPACE, body=service)
                print(f"  ✅ Service '{SERVICE_NAME}' replaced successfully.")
            except ApiException as replace_e:
                print(f"  ❌ Failed to replace existing Service: {replace_e.status} {replace_e.reason}")
                sys.exit(1)
        else:
            print(f"❌ Failed to create Service: {e.status} {e.reason}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error creating Service: {e}")
        sys.exit(1)


def wait_for_pod_running(core_v1_api: client.CoreV1Api, timeout_sec: int = 180) -> Optional[str]:
    """
    Waits for a pod managed by the Deployment (matching DEPLOYMENT_NAME label)
    to enter the 'Running' and 'Ready' state. Returns the name of the running
    pod or None if timeout occurs or pod fails.

    Feature: Ensures the application container is up and running before proceeding
             (e.g., before starting port-forward). Detects immediate pod failures.
    """
    print(f"\n--- 6. Waiting up to {timeout_sec}s for Pod '{DEPLOYMENT_NAME}' to be Running & Ready ---")
    start = time.time()
    last_status_msg = ""

    while time.time() - start < timeout_sec:
        try:
            pods = core_v1_api.list_namespaced_pod(
                namespace=NAMESPACE, label_selector=f"app={DEPLOYMENT_NAME}"
            ).items

            live_pod = next((p for p in pods if p.status.phase not in ["Succeeded", "Failed", "Unknown"]), None)

            if live_pod:
                pod_name = live_pod.metadata.name
                phase = live_pod.status.phase

                # Check container statuses for readiness and image pull errors
                is_ready = False
                if live_pod.status.container_statuses:
                    is_ready = all(c.ready for c in live_pod.status.container_statuses)
                    
                    for cs in live_pod.status.container_statuses:
                         if cs.state and cs.state.waiting and "ImagePullBackOff" in cs.state.waiting.reason:
                             print(f"❌ Pod '{pod_name}' is stuck in ImagePullBackOff. Check image tag/registry access.")
                             return None # Pod failed due to image issue

                # Success condition
                if phase == "Running" and is_ready:
                    print(f"✅ Pod '{pod_name}' is Running & Ready.")
                    return pod_name
                
                current_status_msg = f"Pod '{pod_name}' is {phase}. Ready={is_ready}."
                if current_status_msg != last_status_msg:
                    print(f"  -> {current_status_msg} Waiting...")
                    last_status_msg = current_status_msg

            elif pods:
                # If pods exist but all are in terminal states (Failed/Succeeded)
                terminal_pod = pods[0]
                print(f"❌ Pod '{terminal_pod.metadata.name}' exited or failed immediately (status={terminal_pod.status.phase}). Cannot proceed.")
                return None

        except ApiException as e:
            print(f"  -> Warning: K8s API error while checking pod status: {e.status}. Retrying...")
        except Exception as e:
            print(f"  -> Warning: Unexpected error checking pod status: {e}. Retrying...")

        time.sleep(5)

    print(f"❌ Timeout waiting for pod '{DEPLOYMENT_NAME}' to become Running & Ready.")
    return None


def _open_browser_nonblocking(url: str):
    """
    Helper function to open a URL in a new browser tab in a separate thread
    after a short delay, preventing it from blocking the main script execution.

    Feature: Improves user experience by automatically opening the report URL.
    """
    try:
        time.sleep(2)
        print(f"   -> Opening browser to: {url}")
        webbrowser.open_new_tab(url)
    except Exception as e:
        print(f"   -> Warning: Could not automatically open browser: {e}")


def start_port_forward(local_port: int):
    """
    Starts the `kubectl port-forward` command in a separate background process.
    Manages the process reference globally for later termination. Opens the browser.

    Feature: Provides local access to the service running inside Kubernetes via localhost.
             Handles process management for reliable start/stop.
    """
    global PORT_FORWARD_PROCESS
    url = f"http://127.0.0.1:{local_port}"
    print(f"\n--- 7. Starting Port Forwarding to {url} ---")

    # 🔧 Ensure the port is free before attempting port-forward
    _free_local_port(local_port)
    
    cmd = ["kubectl", "port-forward", f"service/{SERVICE_NAME}", f"{local_port}:{K8S_SERVICE_PORT}", "-n", NAMESPACE]

    try:
        print(f"  -> Executing: {' '.join(cmd)}")
        
        # Use common process creation flags for better cross-platform termination
        process_flags = {}
        if os.name == 'posix':
             process_flags['preexec_fn'] = os.setsid
        else:
             process_flags['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP

        PORT_FORWARD_PROCESS = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, **process_flags)

        time.sleep(2)

        if PORT_FORWARD_PROCESS.poll() is not None:
             stderr_output = PORT_FORWARD_PROCESS.stderr.read().decode()
             print(f"❌ Port-forward failed to start. Error: {stderr_output.strip()}")
             PORT_FORWARD_PROCESS = None
             return

        print(f"  -> Port-forward process started (PID: {PORT_FORWARD_PROCESS.pid}).")

        thread = threading.Thread(target=_open_browser_nonblocking, args=(url,))
        thread.daemon = True
        thread.start()

    except FileNotFoundError:
        print("❌ 'kubectl' command not found. Cannot start port-forwarding.")
        PORT_FORWARD_PROCESS = None
    except Exception as e:
        print(f"❌ Failed to start port-forward process: {e}")
        PORT_FORWARD_PROCESS = None

def stop_port_forward():
    """
    Gracefully terminates the background `kubectl port-forward` process if it's running.

    Feature: Ensures network resources are released cleanly upon script exit or interruption.
    """
    global PORT_FORWARD_PROCESS
    if PORT_FORWARD_PROCESS and PORT_FORWARD_PROCESS.poll() is None:
        print("\n--- Stopping Port Forwarding ---")
        try:
            pid = PORT_FORWARD_PROCESS.pid
            print(f"  -> Terminating port-forward process (PID: {pid})...")
            
            if os.name == 'posix':
                # Kill the entire process group
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            else:
                # Terminate on Windows (taskkill /T kills process tree)
                subprocess.run(f"taskkill /F /PID {pid} /T", check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            PORT_FORWARD_PROCESS.wait(timeout=5)
            print("  ✅ Port-forward process terminated.")

        except ProcessLookupError:
            print("  -> Port-forward process already terminated.")
        except subprocess.TimeoutExpired:
             print("  ⚠️ Warning: Timeout waiting for port-forward process to terminate.")
        except Exception as e:
            print(f"  ⚠️ Error stopping port-forward process: {e}")
        finally:
            PORT_FORWARD_PROCESS = None


def docker_cleanup():
    """
    Performs a safe Docker cleanup: removes stopped containers, unused networks,
    build cache, but preserves tagged local development images (like *-local).

    Feature: Helps manage disk space on the execution host (local or CI runner)
             without accidentally removing essential base or development images.
    """
    print("\n--- 8. Running Docker Cleanup (Safe Prune) ---")
    
    try:
        # 1. System Prune (Containers, Networks, Dangling Images/Volumes)
        print("  -> Running safe system prune...")
        subprocess.run(["docker", "system", "prune", "--force", "--volumes"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 2. Prune Docker Build Cache
        print("  -> Clearing Docker build cache.")
        # Suppress output to keep console clean unless error occurs
        try:
            subprocess.run(["docker", "builder", "prune", "--all", "--force"],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
             if "docker builder" not in e.stderr: # Ignore if builder command is missing (old docker)
                 print(f"  -> Warning during build cache prune: {e.stderr.strip()}")
        
        print("✅ Docker cleanup finished. Tagged local images should be preserved.")

    except FileNotFoundError:
        print("❌ Docker command not found. Skipping cleanup.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during Docker prune: {e.stderr.strip()}")
    except Exception as e:
        print(f"❌ An unexpected error occurred during Docker cleanup: {e}")


# ------------------ MAIN WORKFLOW & ENTRY POINT ------------------

def main_workflow():
    """
    Handles argument parsing, dynamic configuration, and orchestrates the
    entire deployment lifecycle.
    """
    global DEPLOYMENT_NAME, SERVICE_NAME, LOCAL_PORT, IMAGE_NAME

    # Argument parsing (simplified)
    if len(sys.argv) < 2 or (len(sys.argv) < 3 and sys.argv[1] != "cleanup-only"):
        print("\nUsage:")
        print("  python deploy_gpu_workflow.py <BUILD_NUMBER> <FRAMEWORK_NAME>")
        print("  python deploy_gpu_workflow.py cleanup-only")
        print("\nExample:")
        print("  python deploy_gpu_workflow.py 1 gpu-benchmark")
        print("  python deploy_gpu_workflow.py 1 robotics_bdd")        
        sys.exit(1)

    cleanup_mode = (len(sys.argv) == 2 and sys.argv[1] == "cleanup-only")

    if not cleanup_mode:
        BUILD_NUMBER = sys.argv[1]
        FRAMEWORK_ARG = sys.argv[2].lower()

        if not BUILD_NUMBER.isdigit():
            print("❌ BUILD_NUMBER must be an integer.")
            sys.exit(1)

        # Standardize framework name
        FRAMEWORK_NAME_NORM = FRAMEWORK_ARG.replace("_", "-")

        # Dynamic Port Assignment
        port_map = {"robotics-bdd": 8081, "gpu-benchmark": 8082}
        LOCAL_PORT = port_map.get(FRAMEWORK_NAME_NORM, 8080)
        if LOCAL_PORT == 8080:
             print(f"⚠️ Warning: Unknown framework '{FRAMEWORK_ARG}'. Defaulting local port to 8080.")

        # Set Global Dynamic Names
        DEPLOYMENT_NAME = f"{FRAMEWORK_NAME_NORM}-deployment"
        SERVICE_NAME = f"{FRAMEWORK_NAME_NORM}-service"
        IMAGE_NAME = f"{DOCKER_USER}/{FRAMEWORK_NAME_NORM}-report:{BUILD_NUMBER}"

        print("\n====================================================================")
        print("🚀 STARTING KUBERNETES DEPLOYMENT WORKFLOW")
        print("====================================================================")
        print(f"🎯 Target Image: {IMAGE_NAME}")
        print(f"🛠️ Deployment/Service Base: {FRAMEWORK_NAME_NORM}")
        print(f"🌐 Local Port: {LOCAL_PORT}")
    else:
        DEPLOYMENT_NAME = "cleanup"
        SERVICE_NAME = "cleanup"
        print("\n====================================================================")
        print("🚀 RUNNING KUBERNETES CLEANUP ONLY")
        print("====================================================================")


    # --- Execute Workflow ---
    try:
        load_kube_config()

        if cleanup_mode:
            clean_up_deployments(core_v1, apps_v1)
        else:
            # Full Deployment Workflow
            clean_up_deployments(core_v1, apps_v1)
            print_available_pods(core_v1, "Pods Before Deployment")

            gpu_key = find_available_gpu_resource_key(core_v1)
            create_gpu_deployment(apps_v1, IMAGE_NAME, gpu_key)
            create_cluster_ip_service(core_v1)

            pod_name = wait_for_pod_running(core_v1, 180)

            if pod_name:
                print_available_pods(core_v1, "Pods After Deployment", pod_name)
                start_port_forward(LOCAL_PORT)

                # Keep script alive while port-forward runs (for local execution)
                if os.getenv("CI", "false").lower() != "true":
                    print("\n" + "="*60)
                    print(f"   Deployment active. Access report at http://localhost:{LOCAL_PORT}")
                    print("   Press Ctrl+C to stop port-forwarding and clean up resources.")
                    print("="*60)
                    while PORT_FORWARD_PROCESS and PORT_FORWARD_PROCESS.poll() is None:
                        time.sleep(1)
                else:
                    print("\n   CI environment detected. Port-forward running in background.")
            else:
                print("❌ Pod failed to start or become ready. Skipping port-forwarding.")
                sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print(f"{COLOR_YELLOW} 🛑 WORKFLOW MANUALLY INTERRUPTED (Ctrl+C). Cleaning up...{COLOR_RESET}")
        print("="*60)
        # Final cleanup in 'finally' block will handle the rest.

    except ApiException as e:
         print(f"\n❌ FATAL KUBERNETES API ERROR: {e.status} {e.reason}")
         if e.body:
             print(f"   Body: {e.body}")

    except Exception as e:
        print(f"\n❌ UNEXPECTED CRITICAL ERROR during workflow: {e}")

    finally:
        # --- GUARANTEED CLEANUP ---
        print("\n--- Final Cleanup Phase ---")
        stop_port_forward()
        if core_v1 and apps_v1 and DEPLOYMENT_NAME:
             # Only cleanup K8s resources if clients and dynamic names are set
             clean_up_deployments(core_v1, apps_v1)
        docker_cleanup()

        print("\n====================================================================")
        print("WORKFLOW COMPLETED")
        print("====================================================================")

if __name__ == "__main__":
    main_workflow()
