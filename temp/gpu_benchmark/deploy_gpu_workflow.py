import os
import time
import subprocess
from typing import Optional, Tuple, Dict, List

import kubernetes.client as client
from kubernetes.client.rest import ApiException
from kubernetes import config

# ==============================================================================
# SCRIPT PURPOSE & WORKFLOW (Auto GPU Detect + CPU Fallback)
# ==============================================================================
# This script deploys a benchmark workload to Kubernetes and automatically:
#   * Detects available GPU extended resource keys on nodes (Intel/NVIDIA/AMD)
#   * Uses a suitable key if present (prefers Intel: gpu.intel.com/xe -> i915)
#   * Falls back to CPU-only if no GPU resource is advertised
#   * Optionally respects env var REQUIRE_GPU=true to try GPU first, but still
#     falls back to CPU-only if unavailable (set STRICT_GPU=true to hard-fail)
#   * Prints scheduling/event diagnostics while waiting for Pod readiness
#   * Starts `kubectl port-forward` once the Pod is Running
#
# Env vars (optional):
#   DOCKER_USER            : Docker user for image (default: luckyjoy)
#   IMAGE_NAME             : Full image name; if set, overrides DOCKER_USER
#   REQUIRE_GPU            : 'true'/'false' (default: false)
#   STRICT_GPU             : 'true' to fail if GPU not available (default: false)
#   GPU_PREFERRED_KEYS     : Comma list to override key preference order
#                             e.g. "gpu.intel.com/xe,nvidia.com/gpu"
#
# Static Configuration:
#   NAMESPACE: The Kubernetes namespace to deploy into.
#   SERVICE_NAME: The name of the ClusterIP service created.
#   DEPLOYMENT_NAME: The name of the Deployment object.
#
# ==============================================================================

# Static Configuration
NAMESPACE = "default"
SERVICE_NAME = "allure-report-service"
DEPLOYMENT_NAME = "gpu-benchmark-deployment"
LOCAL_PORT = 8080 # The port to use for kubectl port-forward

# Environment variables
DOCKER_USER = os.getenv("DOCKER_USER", "luckyjoy")
# The image built by the companion script is used. The image tag must be correct.
IMAGE_NAME = os.getenv("IMAGE_NAME", f"{DOCKER_USER}/gpu-benchmark-report:latest")
REQUIRE_GPU = os.getenv("REQUIRE_GPU", "false").lower() == "true"
STRICT_GPU = os.getenv("STRICT_GPU", "false").lower() == "true"
# Default preference: Intel i915, then NVIDIA
GPU_PREFERRED_KEYS_DEFAULT = "gpu.intel.com/i915,nvidia.com/gpu"
GPU_PREFERRED_KEYS = os.getenv(
    "GPU_PREFERRED_KEYS", GPU_PREFERRED_KEYS_DEFAULT
).split(",")


def load_kube_config():
    """
    Loads Kubernetes configuration from the default location (e.g., ~/.kube/config)
    or from the service account if running inside a cluster.
    """
    print("--- 1. Loading Kubernetes Configuration ---")
    try:
        config.load_kube_config()
        print("✅ Kubernetes config loaded successfully from local environment.")
    except config.ConfigException:
        print("ℹ️ Attempting to load in-cluster configuration...")
        try:
            config.load_incluster_config()
            print("✅ Kubernetes config loaded successfully from in-cluster service account.")
        except config.ConfigException:
            print("❌ Failed to load any Kubernetes configuration. Is kubectl configured?")
            raise


def find_available_gpu_resource_key(core_v1: client.CoreV1Api) -> Optional[str]:
    """
    Scans all cluster nodes for extended GPU resource keys, preferring the order
    defined in GPU_PREFERRED_KEYS.

    Args:
        core_v1: The Kubernetes CoreV1Api client.

    Returns:
        The first found preferred GPU resource key string (e.g., 'nvidia.com/gpu') or None.
    """
    print("\n--- 2. Detecting Available GPU Resources on Cluster Nodes ---")
    try:
        # Get all nodes in the cluster
        nodes = core_v1.list_node().items
        if not nodes:
            print("⚠️ No nodes found in the cluster.")
            return None

        # Check nodes against preferred keys
        for key in GPU_PREFERRED_KEYS:
            print(f"  -> Checking for resource key: {key}")
            for node in nodes:
                # extended resources are usually found in node.status.capacity
                if key in node.status.capacity:
                    print(f"  ✅ Detected preferred GPU resource key '{key}' on node '{node.metadata.name}'.")
                    return key

        print("  ❌ No preferred GPU resource key was detected on any node.")
        return None

    except ApiException as e:
        print(f"❌ Error communicating with Kubernetes API during node scan: {e}")
        return None


def create_gpu_deployment(
    apps_v1: client.AppsV1Api, core_v1: client.CoreV1Api, image_name: str
) -> Tuple[str, Optional[str]]:
    """
    Creates a Kubernetes Deployment tailored for either GPU or CPU, based on
    cluster detection and environment variables.

    Args:
        apps_v1: The Kubernetes AppsV1Api client.
        core_v1: The Kubernetes CoreV1Api client.
        image_name: The Docker image tag to deploy.

    Returns:
        A tuple (deployment_type_status, used_resource_key)
    """
    print("\n--- 3. Configuring and Creating Deployment ---")
    used_resource_key = find_available_gpu_resource_key(core_v1)
    
    # --- Determine the deployment type ---
    if used_resource_key:
        deployment_type = "GPU"
        print(f"  -> Deploying with GPU resource request: {used_resource_key}")
        # The request/limit value for most extended resources is 1
        resource_limits = {used_resource_key: "1"}
        status_message = "Deploying GPU workload"
    elif REQUIRE_GPU and STRICT_GPU:
        print("  ❌ FATAL: REQUIRE_GPU and STRICT_GPU are true, but no GPU found.")
        raise SystemExit("Deployment failed: Required GPU resource not available.")
    else:
        deployment_type = "CPU"
        print("  -> Deploying with CPU-only (no GPU resource detected).")
        # Standard CPU/memory limits for CPU-only fallback
        resource_limits = {"cpu": "1", "memory": "1Gi"} 
        used_resource_key = None
        status_message = "Deploying CPU-only workload"

    # --- Deployment Definition ---
    container = client.V1Container(
        name="allure-report-server",
        image=image_name,
        # The report image is expected to serve on port 80 (nginx default)
        ports=[client.V1ContainerPort(container_port=80)],
        resources=client.V1ResourceRequirements(
            limits=resource_limits
        ),
        # Assuming the image's ENTRYPOINT/CMD runs the nginx server
    )

    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": DEPLOYMENT_NAME}),
        spec=client.V1PodSpec(containers=[container]),
    )

    spec = client.V1DeploymentSpec(
        replicas=1,
        selector=client.V1LabelSelector(match_labels={"app": DEPLOYMENT_NAME}),
        template=template,
    )

    deployment = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name=DEPLOYMENT_NAME),
        spec=spec,
    )

    # --- Create/Update Deployment ---
    try:
        # Check if deployment exists and update it, or create it
        apps_v1.read_namespaced_deployment(name=DEPLOYMENT_NAME, namespace=NAMESPACE)
        print(f"  ℹ️ Deployment '{DEPLOYMENT_NAME}' already exists. Updating...")
        apps_v1.replace_namespaced_deployment(
            name=DEPLOYMENT_NAME, namespace=NAMESPACE, body=deployment
        )
        print(f"  ✅ Deployment '{DEPLOYMENT_NAME}' updated successfully.")
    except ApiException as e:
        if e.status == 404:
            print(f"  ℹ️ Deployment '{DEPLOYMENT_NAME}' not found. Creating new...")
            apps_v1.create_namespaced_deployment(namespace=NAMESPACE, body=deployment)
            print(f"  ✅ Deployment '{DEPLOYMENT_NAME}' created successfully.")
        else:
            print(f"❌ Error creating/updating deployment: {e}")
            raise

    return status_message, used_resource_key


def create_cluster_ip_service(core_v1: client.CoreV1Api):
    """
    Creates a Kubernetes ClusterIP Service to expose the Deployment.

    Args:
        core_v1: The Kubernetes CoreV1Api client.
    """
    print("\n--- 4. Creating ClusterIP Service ---")
    service_body = client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(name=SERVICE_NAME),
        spec=client.V1ServiceSpec(
            selector={"app": DEPLOYMENT_NAME},
            ports=[
                client.V1ServicePort(
                    protocol="TCP",
                    port=LOCAL_PORT,  # Service port is 8080
                    target_port=80,  # Target container port is 80 (nginx default)
                )
            ],
            type="ClusterIP",
        ),
    )

    try:
        # Check if service exists and update it, or create it
        core_v1.read_namespaced_service(name=SERVICE_NAME, namespace=NAMESPACE)
        print(f"  ℹ️ Service '{SERVICE_NAME}' already exists. Updating...")
        core_v1.replace_namespaced_service(
            name=SERVICE_NAME, namespace=NAMESPACE, body=service_body
        )
        print(f"  ✅ Service '{SERVICE_NAME}' updated successfully.")
    except ApiException as e:
        if e.status == 404:
            print(f"  ℹ️ Service '{SERVICE_NAME}' not found. Creating new...")
            core_v1.create_namespaced_service(namespace=NAMESPACE, body=service_body)
            print(f"  ✅ Service '{SERVICE_NAME}' created successfully.")
        else:
            print(f"❌ Error creating/updating service: {e}")
            raise


def wait_for_pod_running(core_v1: client.CoreV1Api, timeout_sec: int) -> Optional[str]:
    """
    Waits for the Pod created by the Deployment to enter the Running phase.
    Provides detailed status and events during the wait.

    Args:
        core_v1: The Kubernetes CoreV1Api client.
        timeout_sec: The maximum time to wait in seconds.

    Returns:
        The name of the running Pod, or None if timeout/failure occurs.
    """
    print(f"\n--- 5. Waiting for Pod Readiness (Timeout: {timeout_sec}s) ---")
    start_time = time.time()
    last_phase = ""
    last_event_time = 0
    pod_name = None
    
    # Function to print new events since last check
    def print_new_events(pod_labels):
        nonlocal last_event_time
        try:
            # Query events for the matching Pod label
            events = core_v1.list_namespaced_event(
                namespace=NAMESPACE, 
                field_selector=f'involvedObject.name={pod_name}',
                limit=10 # Only grab the last 10 for efficiency
            ).items
            
            new_events = sorted([
                e for e in events if e.last_timestamp and e.last_timestamp.timestamp() > last_event_time
            ], key=lambda x: x.last_timestamp)
            
            for event in new_events:
                ts = event.last_timestamp.strftime("%H:%M:%S") if event.last_timestamp else "N/A"
                print(f"  [{ts} | EVENT] {event.type} ({event.reason}): {event.message}")
                last_event_time = event.last_timestamp.timestamp() if event.last_timestamp else last_event_time

        except ApiException as e:
             # Just log API errors and continue waiting
            if e.status != 404:
                print(f"  ⚠️ Warning: Failed to retrieve events: {e}")
        except Exception:
            pass # Ignore timestamp conversion errors etc.


    while time.time() - start_time < timeout_sec:
        try:
            # Find the Pod associated with the Deployment
            pods = core_v1.list_namespaced_pod(
                namespace=NAMESPACE, label_selector=f"app={DEPLOYMENT_NAME}"
            ).items

            if not pods:
                print("  ℹ️ Waiting for Pod to be created by Deployment...", end='\r')
                time.sleep(2)
                continue

            # The deployment should only manage one replica, so we take the first
            pod = pods[0]
            pod_name = pod.metadata.name
            current_phase = pod.status.phase

            if current_phase != last_phase:
                print(f"\n  [STATUS] Pod '{pod_name}' Phase changed to: {current_phase}")
                last_phase = current_phase
                time.sleep(1) # Give it a moment after a phase change

            if current_phase == "Running":
                # Check for container readiness
                if pod.status.container_statuses and all(c.ready for c in pod.status.container_statuses):
                    print(f"\n  ✅ Pod '{pod_name}' is Running and Ready!")
                    return pod_name
                else:
                    print(f"  [STATUS] Pod '{pod_name}' Running, but containers not yet Ready. Status: {pod.status.container_statuses[0].state.running.started_at.strftime('%H:%M:%S') if pod.status.container_statuses and pod.status.container_statuses[0].state.running else 'Starting...'}", end='\r')

            elif current_phase in ["Pending", "ContainerCreating"]:
                # Print ongoing status and events
                print(f"  [STATUS] Pod '{pod_name}' in {current_phase}. Waiting for scheduler/image pull... ({int(time.time() - start_time)}s)", end='\r')
                print_new_events({"app": DEPLOYMENT_NAME})

            elif current_phase in ["Failed", "Error", "Unknown"]:
                print(f"\n  ❌ Pod '{pod_name}' failed with status: {current_phase}")
                print(f"  Last Pod Events:")
                print_new_events({"app": DEPLOYMENT_NAME})
                return None # Exit on failure

        except ApiException as e:
            print(f"\n❌ Error during pod status check: {e}")
            return None

        time.sleep(5)  # Poll every 5 seconds

    print(f"\n❌ Timeout reached ({timeout_sec}s). Pod did not reach 'Running' status.")
    return None


def start_port_forward(pod_name: str, local_port: int):
    """
    Starts a blocking kubectl port-forward command to access the report service.

    Args:
        pod_name: The name of the Pod to forward from.
        local_port: The local port to listen on.
    """
    print("\n--- 6. Starting Kubectl Port-Forward ---")
    
    # Command: kubectl port-forward <pod-name> <local-port>:<container-port>
    # Since the Nginx inside the report image listens on 80, we use 80 here.
    cmd: List[str] = [
        "kubectl", 
        "port-forward", 
        pod_name, 
        f"{local_port}:80",
        "-n", NAMESPACE,
    ]
    print(f"Executing command: {' '.join(cmd)}")
    print(f"Access the Allure report at: \033[1;34mhttp://127.0.0.1:{local_port}\033[0m")
    print("\n*** This command is BLOCKING. Press \033[1;31mCtrl+C\033[0m to stop forwarding and exit. ***")
    try:
        # The Popen command starts a child process, which inherits the parent's
        # signal handling, allowing Ctrl+C to terminate it.
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("\nERROR: 'kubectl' not found in PATH.")
    except subprocess.CalledProcessError as e:
        print(f"\nERROR during port-forwarding: {e}")
    except KeyboardInterrupt:
        print("\nPort forwarding stopped by user (\033[1;31mCtrl+C\033[0m). Exiting.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == '__main__':
    print("==============================================================================")
    print("      STARTING KUBERNETES GPU BENCHMARK DEPLOYMENT WORKFLOW                   ")
    print("==============================================================================")
    print(f"Using Docker image: {IMAGE_NAME}")
    print(f"Namespace: {NAMESPACE} | Require GPU: {REQUIRE_GPU} | Strict GPU: {STRICT_GPU}")
    print(f"Access Port: {LOCAL_PORT}")

    try:
        load_kube_config()
        # Initialize Kubernetes API clients
        apps_v1 = client.AppsV1Api()
        core_v1 = client.CoreV1Api()

        # 1. Create/Update Deployment
        status, used_key = create_gpu_deployment(apps_v1, core_v1, IMAGE_NAME)
        print(f"\n[STATUS SUMMARY] Deployment created/updated. Mode: {status}. Resource Key: {used_key or 'N/A'}")

        # 2. Create/Update Service
        create_cluster_ip_service(core_v1)
        print("\n[STATUS SUMMARY] Service is ready for port-forwarding.")

        # 3. Wait for Pod to be Running
        # Start a 3-minute timer for the Pod to be scheduled and run its container
        pod_name = wait_for_pod_running(core_v1, timeout_sec=180) 
        
        # 4. Start Port-Forwarding
        if pod_name:
            start_port_forward(pod_name, LOCAL_PORT)
        else:
            print("\nWorkflow completed, but the Pod failed to start. Cannot start port-forwarding.")

    except Exception as e:
        print(f"\n--- FATAL WORKFLOW ERROR ---")
        print(f"An error prevented the deployment workflow from completing: {e}")
        sys.exit(1)
    finally:
        print("\n==============================================================================")
        print("WORKFLOW COMPLETED. (The port-forwarding window has closed.)")
        print("==============================================================================")
