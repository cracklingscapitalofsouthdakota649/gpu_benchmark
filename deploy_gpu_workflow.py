# ==============================================================================
# SCRIPT PURPOSE & WORKFLOW (Auto GPU Detect + CPU Fallback)
# Filename: deploy_gpu_workflow.py
# ==============================================================================
# This script deploys a benchmark workload to Kubernetes and automatically:
#   1. Deletes all existing deployments for a clean slate.
#   2. Detects available GPU extended resource keys on cluster nodes.
#   3. Configures the Deployment to request that GPU resource or fall back to CPU.
#   4. Creates the necessary Deployment and Service.
#   5. Monitors Pod creation, reporting scheduling events.
#   6. Starts a blocking `kubectl port-forward` to access the application.

import os
import time
import subprocess
from typing import Optional, Tuple, Dict, List
import sys

# Import the necessary Kubernetes client libraries
import kubernetes.client as client
from kubernetes.client.rest import ApiException
from kubernetes import config

# Static Configuration:
NAMESPACE = "default"
SERVICE_NAME = "allure-report-service"
DEPLOYMENT_NAME = "gpu-benchmark-deployment"
LOCAL_PORT = 8080 # The local port to use for kubectl port-forward
# ==============================================================================

# --- ANSI Color Codes for Output ---
COLOR_GREEN = '\033[92m'
COLOR_RESET = '\033[0m'
COLOR_YELLOW = '\033[93m'
COLOR_BLUE = '\033[94m'

# --- Environment Configuration ---
DOCKER_USER = os.getenv("DOCKER_USER", "luckyjoy")
IMAGE_NAME = os.getenv("IMAGE_NAME", f"{DOCKER_USER}/gpu-benchmark-report:latest")
REQUIRE_GPU = os.getenv("REQUIRE_GPU", "false").lower() == "true"
STRICT_GPU = os.getenv("STRICT_GPU", "false").lower() == "true"

# Resource keys are vendor-specific (e.g., Intel, NVIDIA).
GPU_PREFERRED_KEYS_DEFAULT = "gpu.intel.com/i915,nvidia.com/gpu"
GPU_PREFERRED_KEYS = os.getenv(
    "GPU_PREFERRED_KEYS", GPU_PREFERRED_KEYS_DEFAULT
).split(",")


def print_available_pods(core_v1: client.CoreV1Api, title: str, highlight_pod_name: Optional[str] = None):
    """
    Prints a list of existing pods in the target namespace, highlighting the
    newly created pod in green if its name is provided.
    """
    print(f"\n--- {title} Pods in Namespace '{NAMESPACE}' ---")
    try:
        pods = core_v1.list_namespaced_pod(namespace=NAMESPACE).items
        if not pods:
            print("  (No pods found in this namespace.)")
            return
        
        # Format and print pod details header
        print(f"  {'NAME':<35} {'STATUS':<15} {'AGE':<10}")
        print("  " + "="*60)
        
        for pod in pods:
            name = pod.metadata.name
            phase = pod.status.phase if pod.status else "Unknown"
            
            # Calculate age for better logging
            creation_time = pod.metadata.creation_timestamp
            age = "N/A"
            if creation_time:
                delta = time.time() - creation_time.timestamp()
                if delta < 60:
                    age = f"{int(delta)}s"
                elif delta < 3600:
                    age = f"{int(delta / 60)}m"
                else:
                    age = f"{int(delta / 3600)}h"

            # Apply color if this is the highlighted new pod
            color_prefix = COLOR_GREEN if name == highlight_pod_name else ""
            color_suffix = COLOR_RESET if name == highlight_pod_name else ""

            print(f"  {color_prefix}{name:<35} {phase:<15} {age:<10}{color_suffix}")

    except ApiException as e:
        print(f"❌ Warning: Could not list pods. API Error: {e.status}")
    except Exception as e:
        print(f"❌ Warning: An unexpected error occurred while listing pods: {e}")


def load_kube_config():
    """
    --- 1. Loading Kubernetes Configuration ---
    Loads Kubernetes configuration from the default location or in-cluster.
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

def clean_up_deployments():
    """
    --- 1.1. Clean Up Existing Development Deployments ---
    Deletes ALL Deployments in the target namespace using kubectl.
    """
    print(f"\n---- 1.1. Clean Up Existing Development Deployments (Namespace: {NAMESPACE}) ---")
    
    cmd: List[str] = [
        "kubectl", 
        "delete", 
        "deployment", 
        "--all", 
        "-n", NAMESPACE,
    ]
    
    try:
        # Run the delete command. capture_output=True to hide stdout/stderr unless there's an error.
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        
        if "No resources found" in result.stderr or "No resources found" in result.stdout:
            print("  ✅ No existing Deployments found. Clean slate.")
        elif result.returncode == 0:
            print(f"  ✅ Successfully deleted existing Deployments in namespace '{NAMESPACE}'.")
        else:
            # Handle cases where deletion failed, but print the error
            print(f"  ⚠️ Warning: Deletion command returned non-zero exit code ({result.returncode}). Output:")
            print(result.stderr.strip())
            
    except FileNotFoundError:
        print("\nERROR: 'kubectl' not found in PATH. Skipping clean up.")
    except Exception as e:
        print(f"\nERROR during clean up: {e}")
        
    # Wait a moment for resources to terminate
    time.sleep(3)


def find_available_gpu_resource_key(core_v1: client.CoreV1Api) -> Optional[str]:
    """
    --- 2. Detecting Available GPU Resources on Cluster Nodes ---
    Scans all cluster nodes for extended GPU resource keys.
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
                # Extended resources are found in node.status.capacity
                if key in node.status.capacity:
                    print(f"  ✅ Detected preferred GPU resource key '{key}' on node '{node.metadata.name}'.")
                    return key

        print("  ❌ No preferred GPU resource key was detected on any node.")
        return None

    except ApiException as e:
        print(f"❌ Error communicating with Kubernetes API during node scan: {e}")
        return None


def create_gpu_deployment(
    apps_v1: client.AppsV1Api, image_name: str, used_resource_key: Optional[str]
) -> Tuple[str, str]:
    """
    --- 3. Configuring and Creating Deployment ---
    Creates a Kubernetes Deployment tailored for either GPU or CPU using the
    pre-detected resource key.
    """
    print("\n--- 3. Configuring and Creating Deployment ---")
    
    # --- Determine the deployment type and resource limits ---
    if used_resource_key:
        deployment_type = "GPU"
        # The request/limit value for most extended resources is 1
        resource_limits = {used_resource_key: "1"}
        status_message = "Deploying GPU workload"
        key_status = f"Using extended resource key: {used_resource_key}"
        
        print(f"  -> {COLOR_YELLOW}MODE: GPU Acceleration Enabled{COLOR_RESET}")
        print(f"  -> Resource Request/Limits (GPU): {resource_limits}")
        
    elif REQUIRE_GPU and STRICT_GPU:
        print("  ❌ FATAL: REQUIRE_GPU and STRICT_GPU are true, but no GPU found.")
        raise SystemExit("Deployment failed: Required GPU resource not available.")
    else:
        deployment_type = "CPU"
        # Standard CPU/memory limits for CPU-only fallback
        resource_limits = {"cpu": "1", "memory": "1Gi"} 
        status_message = "Deploying CPU-only workload"
        key_status = "No GPU key found. Defaulting to CPU-only limits (1 core / 1Gi memory)."
        
        print(f"  -> {COLOR_YELLOW}MODE: CPU Fallback Activated{COLOR_RESET}")
        print(f"  -> Resource Request/Limits (CPU): {resource_limits}")

    # Explicitly print the image and deployment name
    print(f"  -> Selected Docker Image: {image_name}")
    print(f"  -> Target Deployment Name: {DEPLOYMENT_NAME}")

    # --- Deployment Definition ---
    container = client.V1Container(
        name="allure-report-server",
        image=image_name,
        ports=[client.V1ContainerPort(container_port=80)],
        resources=client.V1ResourceRequirements(
            limits=resource_limits
        ),
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
    # Since we added cleanup, we assume the deployment does not exist, but we keep 
    # the read/replace logic for robustness against non-cleanup runs.
    try:
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

    return status_message, key_status


def create_cluster_ip_service(core_v1: client.CoreV1Api):
    """
    --- 4. Creating ClusterIP Service ---
    Creates a Kubernetes ClusterIP Service to expose the Deployment.
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
                    port=LOCAL_PORT,
                    target_port=80,
                )
            ],
            type="ClusterIP",
        ),
    )

    try:
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
    --- 5. Waiting for Pod Readiness ---
    Waits for the Pod created by the Deployment to enter the Running phase.
    """
    print(f"\n--- 5. Waiting for Pod Readiness (Timeout: {timeout_sec}s) ---")
    start_time = time.time()
    last_phase = ""
    last_event_time = 0
    pod_name = None
    
    # Helper function to print new events (e.g., 'Scheduled', 'Pulling image')
    def print_new_events():
        nonlocal last_event_time
        try:
            if not pod_name:
                return
                
            events = core_v1.list_namespaced_event(
                namespace=NAMESPACE, 
                field_selector=f'involvedObject.name={pod_name}',
                limit=10 
            ).items
            
            new_events = sorted([
                e for e in events if e.last_timestamp and e.last_timestamp.timestamp() > last_event_time
            ], key=lambda x: x.last_timestamp)
            
            for event in new_events:
                ts = event.last_timestamp.strftime("%H:%M:%S") if event.last_timestamp else "N/A"
                print(f"  [{ts} | EVENT] {event.type} ({event.reason}): {event.message}")
                last_event_time = event.last_timestamp.timestamp() if event.last_timestamp else last_event_time

        except ApiException as e:
            if e.status != 404:
                print(f"  ⚠️ Warning: Failed to retrieve events: {e}")
        except Exception:
            pass


    while time.time() - start_time < timeout_sec:
        try:
            # Find the Pod associated with the Deployment label selector
            pods = core_v1.list_namespaced_pod(
                namespace=NAMESPACE, label_selector=f"app={DEPLOYMENT_NAME}"
            ).items

            if not pods:
                print("  ℹ️ Waiting for Pod to be created by Deployment...", end='\r')
                time.sleep(2)
                continue

            pod = pods[0]
            
            # Explicitly report the selected pod name when found
            if pod_name is None:
                pod_name = pod.metadata.name
                print(f"\n  ℹ️ Deployment Pod created: {pod_name}")

            current_phase = pod.status.phase

            if current_phase != last_phase:
                print(f"\n  [STATUS] Pod '{pod_name}' Phase changed to: {current_phase}")
                last_phase = current_phase
                time.sleep(1) 

            if current_phase == "Running":
                # Check for container readiness
                if pod.status.container_statuses and all(c.ready for c in pod.status.container_statuses):
                    print(f"\n  ✅ Pod '{pod_name}' is Running and Ready!")
                    return pod_name
                else:
                    print(f"  [STATUS] Pod '{pod_name}' Running, but containers not yet Ready. ({int(time.time() - start_time)}s)", end='\r')

            elif current_phase in ["Pending", "ContainerCreating"]:
                print(f"  [STATUS] Pod '{pod_name}' in {current_phase}. Waiting for scheduler/image pull... ({int(time.time() - start_time)}s)", end='\r')
                print_new_events()

            elif current_phase in ["Failed", "Error", "Unknown"]:
                print(f"\n  ❌ Pod '{pod_name}' failed with status: {current_phase}")
                print(f"  Last Pod Events:")
                print_new_events()
                return None 

        except ApiException as e:
            print(f"\n❌ Error during pod status check: {e}")
            return None

        time.sleep(5)

    print(f"\n❌ Timeout reached ({timeout_sec}s). Pod did not reach 'Running' status.")
    return None


def start_port_forward(pod_name: str, local_port: int):
    """
    --- 6. Starting Kubectl Port-Forward ---
    Starts a blocking kubectl port-forward command to access the report service.
    """
    print("\n--- 6. Starting Kubectl Port-Forward ---")
    
    # Command: kubectl port-forward <pod-name> <local-port>:<container-port>
    cmd: List[str] = [
        "kubectl", 
        "port-forward", 
        pod_name, 
        f"{local_port}:80",
        "-n", NAMESPACE,
    ]
    print(f"Executing command: {' '.join(cmd)}")
    print(f"Access the Allure report at: {COLOR_BLUE}http://127.0.0.1:{local_port}{COLOR_RESET}")
    print("\n*** This command is BLOCKING. Press \033[1;31mCtrl+C\033[0m to stop forwarding and exit. ***")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("\nERROR: 'kubectl' not found in PATH. Please ensure kubectl is installed.")
    except subprocess.CalledProcessError as e:
        print(f"\nERROR during port-forwarding: {e}")
    except KeyboardInterrupt:
        print(f"\nPort forwarding stopped by user (\033[1;31mCtrl+C{COLOR_RESET}). Exiting.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == '__main__':
    print("==============================================================================")
    print("      STARTING KUBERNETES GPU BENCHMARK DEPLOYMENT WORKFLOW                   ")
    print("        Author: Bang Thien Nguyen - ontario1998@gmail.com                     ")
    print("==============================================================================")
    
    pod_name_to_highlight = None
    
    try:
        load_kube_config()
        # Initialize Kubernetes API clients
        apps_v1 = client.AppsV1Api()
        core_v1 = client.CoreV1Api()
        
        # New Step: Clean up existing deployments before starting
        clean_up_deployments()

        # Print current pod status before we begin deployment
        print_available_pods(core_v1, "Pods Before Deployment Workflow")

        # STEP 2: Detect available GPU resources
        used_resource_key = find_available_gpu_resource_key(core_v1)
        
        # STEP 3: Create/Update Deployment
        status, key_status = create_gpu_deployment(apps_v1, IMAGE_NAME, used_resource_key)
        print(f"\n[STATUS SUMMARY] Deployment created/updated. Mode: {status}. Resource Key Status: {key_status}")

        # STEP 4: Create/Update Service
        create_cluster_ip_service(core_v1)
        print("\n[STATUS SUMMARY] Service is ready for port-forwarding.")

        # STEP 5: Wait for Pod to be Running
        pod_name_to_highlight = wait_for_pod_running(core_v1, timeout_sec=180) 
        
        # Print pod status after the deployment attempt has finished, highlighting the new pod
        print_available_pods(core_v1, "Pods After Deployment Workflow", highlight_pod_name=pod_name_to_highlight)

        # STEP 6: Start Port-Forwarding
        if pod_name_to_highlight:
            start_port_forward(pod_name_to_highlight, LOCAL_PORT)
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