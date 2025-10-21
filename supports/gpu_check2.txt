# supports/gpu_check.py
"""
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

from __future__ import annotations

import json
import importlib
import platform
import subprocess
from typing import Dict, Any


def _detect_with_torch() -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "cuda": False,
        "rocm": False,
        "directml": False,
        "metal": False,
        "backend": None,
    }
    try:
        import torch  # noqa: F401

        # CUDA (NVIDIA)
        try:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                summary["cuda"] = True
                summary["backend"] = summary["backend"] or "cuda"
        except Exception:
            pass

        # ROCm (AMD)
        try:
            if getattr(torch.version, "hip", None):
                summary["rocm"] = True
                summary["backend"] = summary["backend"] or "rocm"
        except Exception:
            pass

        # Apple Metal / MPS
        try:
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                summary["metal"] = True
                summary["backend"] = summary["backend"] or "mps"
        except Exception:
            pass

        # DirectML (Windows; requires torch-directml present)
        try:
            if importlib.util.find_spec("torch_directml") is not None:
                summary["directml"] = True
                summary["backend"] = summary["backend"] or "directml"
        except Exception:
            pass

    except Exception as e:
        # Torch not installed or failed: record as non-fatal info
        summary["error"] = str(e)

        # Even without torch, we can still detect DirectML through onnxruntime-directml
        try:
            if importlib.util.find_spec("onnxruntime") is not None:
                # Some ORT builds include a directml provider
                import onnxruntime as ort  # type: ignore
                providers = set(ort.get_available_providers())
                if any(p.lower().startswith("dml") or "directml" in p.lower() for p in providers):
                    summary["directml"] = True
                    summary["backend"] = summary["backend"] or "directml"
        except Exception:
            pass

    return summary


def _detect_with_opencl_py() -> Dict[str, Any]:
    """Best-effort OpenCL detection via pyopencl (if installed)."""
    info: Dict[str, Any] = {"opencl_gpu_devices": 0}
    try:
        import pyopencl as cl  # noqa: F401

        count = 0
        for plat in cl.get_platforms():
            for dev in plat.get_devices():
                if cl.device_type.to_string(dev.type) == "GPU":
                    count += 1
        info["opencl_gpu_devices"] = count
    except Exception:
        pass
    return info


def _detect_with_clinfo() -> Dict[str, Any]:
    """
    Fallback OpenCL detection by calling `clinfo` if present in PATH.
    Parses output heuristically to count 'Device #' lines where Type = GPU.
    """
    info: Dict[str, Any] = {"opencl_gpu_devices": 0}
    try:
        proc = subprocess.run(
            ["clinfo"],
            capture_output=True,
            text=True,
            check=False,
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if out:
            # Heuristic: count occurrences of lines like 'Device Type ... GPU'
            gpu_count = 0
            current_device_is_gpu = False
            for line in out.splitlines():
                s = line.strip()
                if s.lower().startswith("device ") and ("#" in s or "name" in s.lower()):
                    # reset on each new device section
                    current_device_is_gpu = False
                if "device type" in s.lower() and "gpu" in s.lower():
                    current_device_is_gpu = True
                if s.lower().startswith("max compute units") and current_device_is_gpu:
                    gpu_count += 1
                    current_device_is_gpu = False
            if gpu_count == 0:
                # Simpler heuristic if the above misses: count literal 'Type: GPU'
                gpu_count = out.lower().count("type: gpu")
            info["opencl_gpu_devices"] = max(info["opencl_gpu_devices"], gpu_count)
    except Exception:
        pass
    return info


def _detect_intel_gpu_windows() -> Dict[str, Any]:
    """
    Windows-specific detection: uses WMI (Win32_VideoController) via PowerShell.
    Counts Intel GPUs even when no OpenCL stack or pyopencl is present.
    """
    info: Dict[str, Any] = {"opencl_gpu_devices": 0}
    if platform.system().lower() != "windows":
        return info
    try:
        ps_cmd = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            r"Get-CimInstance Win32_VideoController | "
            r"Select-Object Name,PNPDeviceID,AdapterRAM,DriverVersion,VideoProcessor | "
            r"ConvertTo-Json -Depth 3"
        ]
        proc = subprocess.run(ps_cmd, capture_output=True, text=True, check=False)
        data = proc.stdout.strip()
        if not data:
            return info
        try:
            js = json.loads(data)
        except Exception:
            # Sometimes PowerShell returns UTF-16; try decoding stderr/stdout differently
            js = []
        devices = js if isinstance(js, list) else [js]
        intel_count = 0
        for d in devices:
            name = (d.get("Name") or "").lower()
            vproc = (d.get("VideoProcessor") or "").lower()
            if ("intel" in name or "intel" in vproc) and "microsoft basic render" not in name:
                intel_count += 1
        if intel_count > 0:
            # We can't confirm OpenCL runtime presence, but for test routing purposes
            # signal there is at least one GPU the system could use via OpenCL/DirectX.
            info["opencl_gpu_devices"] = max(info["opencl_gpu_devices"], intel_count)
        return info
    except Exception:
        return info


def get_gpu_info() -> Dict[str, Any]:
    """Public API used by tests.device_utils: returns {"summary": {...}}."""
    summary: Dict[str, Any] = {}

    # Torch-backed detection (CUDA/ROCm/MPS/DirectML)
    t = _detect_with_torch()
    summary.update(t)

    # OpenCL by pyopencl (if present)
    ocl = _detect_with_opencl_py()
    for k, v in ocl.items():
        if k == "opencl_gpu_devices":
            # keep the max value across methods
            summary["opencl_gpu_devices"] = max(summary.get("opencl_gpu_devices", 0), v)

    # OpenCL by clinfo (if present)
    ocli = _detect_with_clinfo()
    if "opencl_gpu_devices" in ocli:
        summary["opencl_gpu_devices"] = max(summary.get("opencl_gpu_devices", 0), ocli["opencl_gpu_devices"])

    # Windows Intel GPU via WMI (no external deps)
    win_intel = _detect_intel_gpu_windows()
    if "opencl_gpu_devices" in win_intel:
        summary["opencl_gpu_devices"] = max(summary.get("opencl_gpu_devices", 0), win_intel["opencl_gpu_devices"])

    # If we detected *only* an OpenCL-capable GPU (e.g., Intel via WMI),
    # and no other backend was chosen, prefer "opencl" as a routing hint.
    if summary.get("backend") is None and summary.get("opencl_gpu_devices", 0) > 0:
        summary["backend"] = "opencl"

    return {"summary": summary}


def main() -> None:
    print(json.dumps(get_gpu_info(), indent=2))


if __name__ == "__main__":
    main()