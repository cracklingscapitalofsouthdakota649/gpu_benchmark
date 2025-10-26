"""
# supports/gpu_check.py
Cross-platform GPU capability detection for tests.

Returns a dict shaped like:
{
  "summary": {
    "cuda": bool,
    "rocm": bool,
    "directml": bool,
    "metal": bool,                 # True when PyTorch MPS is available (Apple Metal)
    "opencl_gpu_devices": int,     # Number of OpenCL GPU devices detected (best-effort)
    "backend": "cuda"|"rocm"|"mps"|"directml"|"opencl"|None,
    "error": str (optional)        # Aggregate, non-fatal info if something went wrong
  }
}
"""
import json
import subprocess

def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return str(e)

def detect_with_torch():
    """Detect CUDA/ROCm/DirectML via torch if available."""
    info = {"cuda": False, "rocm": False, "directml": False, "backend": None}
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda"] = True
            info["backend"] = "cuda"
        elif torch.backends.mps.is_available():  # macOS
            info["backend"] = "mps"
        else:
            # Check for DirectML
            try:
                import torch_directml
                device = torch_directml.device()
                if device:
                    info["directml"] = True
                    info["backend"] = "directml"
            except ImportError:
                pass
    except Exception as e:
        info["error"] = str(e)
    return info


def detect_with_opencl():
    """Fallback detection using OpenCL — useful for Intel GPUs."""
    info = {"opencl_gpu_devices": 0, "intel_gpu": False}
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        for platform in platforms:
            devices = platform.get_devices()
            gpu_devices = [d for d in devices if d.type == cl.device_type.GPU]
            if gpu_devices:
                info["opencl_gpu_devices"] += len(gpu_devices)
                if "Intel" in platform.name:
                    info["intel_gpu"] = True
        info["backend"] = "opencl"
    except Exception:
        info["backend"] = None
    return info


def get_gpu_info():
    """Return GPU info as a Python dictionary."""
    summary = {}
    try:
        torch_info = detect_with_torch()
        opencl_info = detect_with_opencl()
        summary.update(torch_info)
        summary.update(opencl_info)
    except Exception as e:
        summary["error"] = str(e)
    return summary


def main():
    """CLI mode – print JSON to stdout for subprocess detection."""
    summary = get_gpu_info()
    print(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    main()
