# gpu_benchmark.py
import subprocess
import sys
import os
import json
import venv

PYTHON_EXE = r"C:\Users\ontar\AppData\Local\Programs\Python\Python310\python.exe"
VENV_DIR = "venv310"

COMMON_PACKAGES = [
    "pytest==8.4.2",
    "pytest-benchmark==5.1.0",
    "allure-pytest==2.15.0",
    "psutil==7.1.0",
    "matplotlib==3.10.7",
    "numpy==2.2.6",
]

TORCH_PACKAGES = {
    "cuda": ["torch==2.2.0", "torchvision==0.18.1", "torchaudio==2.2.0"],
    "rocm": ["torch==2.2.0", "torchvision==0.18.1", "torchaudio==2.2.0"],
    "intel": ["torch==2.2.0", "torchvision==0.18.1", "torchaudio==2.2.0"],
    "directml": ["torch==2.2.0", "torchvision==0.18.1", "torchaudio==2.2.0"],
    "cpu": ["torch==2.2.0", "torchvision==0.18.1", "torchaudio==2.2.0"],
}


def run_cmd(cmd):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def create_venv():
    if not os.path.exists(VENV_DIR):
        print(f"[INFO] Creating virtual environment: {VENV_DIR}")
        venv.create(VENV_DIR, with_pip=True)
    else:
        print(f"[INFO] Using existing virtual environment: {VENV_DIR}")


def detect_gpu_flavor():
    print("[INFO] Detecting GPU...")
    try:
        ret = subprocess.run(
            [PYTHON_EXE, "supports/gpu_check.py"],
            capture_output=True,
            text=True,
            check=False,
        )
        if ret.returncode != 0:
            print(f"[WARN] gpu_check.py error: {ret.stderr.strip()}")
            return "cpu"

        if not ret.stdout.strip():
            print("[WARN] gpu_check.py returned empty output")
            return "cpu"

        # Parse JSON safely
        try:
            info = json.loads(ret.stdout)
        except json.JSONDecodeError:
            print(f"[WARN] gpu_check.py returned invalid JSON: {ret.stdout.strip()}")
            return "cpu"

        summary = info.get("summary", {})
        if summary.get("cuda"):
            return "cuda"
        elif summary.get("rocm"):
            return "rocm"
        elif summary.get("directml"):
            return "directml"
        elif summary.get("opencl_gpu_devices", 0) > 0:
            return "intel"
        else:
            return "cpu"

    except FileNotFoundError:
        print(f"[WARN] gpu_check.py not found. Falling back to CPU")
        return "cpu"
    except Exception as e:
        print(f"[WARN] Unexpected error during GPU detection: {e}")
        return "cpu"


def install_packages(pip_exe, packages):
    for pkg in packages:
        run_cmd([pip_exe, "install", "--no-deps", pkg])


def main():
    create_venv()
    pip_exe = os.path.join(VENV_DIR, "Scripts", "pip.exe")
    flavor = detect_gpu_flavor()
    print(f"[INFO] Detected GPU flavor: {flavor}")

    # Install common packages
    install_packages(pip_exe, COMMON_PACKAGES)

    # Install PyTorch packages for detected flavor
    install_packages(pip_exe, TORCH_PACKAGES.get(flavor, TORCH_PACKAGES["cpu"]))

    print("[INFO] Installation complete. Running benchmark tests...")
    python_exe = os.path.join(VENV_DIR, "Scripts", "python.exe")
    run_cmd([python_exe, "-m", "pytest", "--alluredir=allure-results", "-v"])


if __name__ == "__main__":
    main()
