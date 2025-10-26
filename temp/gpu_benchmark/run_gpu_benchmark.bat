@echo off
REM Enable delayed expansion to handle variable changes inside code blocks (e.g., in loops)
setlocal enabledelayedexpansion

REM --- Setup ---
echo --- Checking Virtual Environment ---
IF NOT EXIST "venv310\Scripts\activate.bat" (
    echo [ERROR] Virtual environment 'venv310' not found.
    echo Please create it by running: python -m venv venv310
    goto :error
)
call venv310\Scripts\activate.bat
echo --- Environment Activated ---

REM ----------------------------------------------------------------------
REM CRITICAL FIX: Prevents 'amdsmi' TypeError on non-AMD systems
REM This must be set BEFORE any Python process runs.
set "TORCH_HIP_PROHIBIT_AMDSMI_INIT=1"
REM ----------------------------------------------------------------------

REM --- Clean Up ---
IF EXIST "allure-results" (
    echo --- Cleaning up old allure-results directory ---
    rd /s /q "allure-results"
)
md "allure-results"
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create allure-results directory. Check permissions.
    goto :error
)

REM --- Install shared tooling ---
echo --- Installing/Updating test tooling ---
python -m pip install --upgrade pip
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to upgrade pip.
    goto :pipfail
)

echo --- Installing requirements.txt dependencies ---
pip install -r requirements.txt --no-deps
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install requirements from requirements.txt.
    goto :pipfail
)

REM --- Detect GPU vendor ---
set "GPU_VENDOR=CPU"
call :DETECT_GPU
echo --- Detected GPU Vendor: %GPU_VENDOR%

REM --- Framework install per vendor ---
if /I "%GPU_VENDOR%"=="NVIDIA" goto :install_nvidia
if /I "%GPU_VENDOR%"=="INTEL" goto :install_intel
goto :install_cpu


:install_nvidia
    echo --- Checking PyTorch CUDA installation for NVIDIA ---
    python -c "import torch" >NUL 2>&1
    IF %ERRORLEVEL% EQU 0 (
        echo [INFO] PyTorch already installed. Skipping reinstallation.
        goto :pytorch_installed
    )
    echo --- Installing PyTorch CUDA (cu124) for NVIDIA ---
    pip uninstall -y torch torchvision torchaudio >NUL 2>&1
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to install PyTorch for NVIDIA/CUDA.
        goto :pipfail
    )
    goto :pytorch_installed

:install_intel
    echo --- Checking PyTorch-DirectML installation for Intel GPU ---
    python -c "import torch_directml" >NUL 2>&1
    IF %ERRORLEVEL% EQU 0 (
        echo [INFO] PyTorch-DirectML already installed. Skipping reinstallation.
        goto :pytorch_installed
    )
    echo --- Installing PyTorch-DirectML for Intel GPU on Windows ---
    pip uninstall -y torch torchvision torchaudio torch-directml >NUL 2>&1
    pip install torch-directml
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to install PyTorch-DirectML for Intel.
        goto :pipfail
    )
    goto :pytorch_installed

:install_cpu
    echo --- Checking CPU-only PyTorch installation ---
    python -c "import torch" >NUL 2>&1
    IF %ERRORLEVEL% EQU 0 (
        echo [INFO] CPU-only PyTorch already installed. Skipping reinstallation.
        goto :pytorch_installed
    )
    echo --- Installing CPU-only PyTorch wheels (Vendor: %GPU_VENDOR%) ---
    pip uninstall -y torch torchvision torchaudio >NUL 2>&1
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to install CPU-only PyTorch.
        goto :pipfail
    )

:pytorch_installed
REM --- PyTorch Verification ---
echo --- Verifying PyTorch and GPU backend installation ---
python -c "import sys; import torch; print('torch_version:', torch.__version__); print('cuda_available:', getattr(torch.cuda, 'is_available', lambda: False)()); has_xpu = hasattr(torch, 'xpu') and getattr(torch.xpu, 'is_available', lambda: False)(); print('xpu_available:', has_xpu); directml=False; \
import importlib.util; spec=importlib.util.find_spec('torch_directml'); \
if spec: print('directml_available:', True); else: print('directml_available:', False)"

IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] PyTorch verification failed. Check the output above.
    goto :pipfail
)

REM --- Run GPU tests ---
echo --- Running pytest benchmark (Marker: gpu) ---
python -m pytest -m gpu --alluredir=allure-results
IF %ERRORLEVEL% NEQ 0 (
    if %ERRORLEVEL% EQU 5 (
        echo [WARN] GPU tests returned status 5 (no tests found or collected). This is often safe to ignore.
    ) else (
        echo [WARN] GPU tests returned non-zero error code %ERRORLEVEL%. Continuing to CPU tests...
    )
)

REM --- Run CPU tests ---
echo --- Running CPU-marked tests ---
python -m pytest -m cpu --alluredir=allure-results
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] CPU-marked pytest execution failed with code %ERRORLEVEL%.
    goto :pytestfail
)

REM --- Update Allure History ---
echo --- Updating Allure history trend ---
python scripts\update_trend.py
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Allure trend update failed. Ensure scripts\update_trend.py exists and works.
    goto :trendfail
)

echo.
echo --- Benchmark run complete ---
echo Run 'allure serve allure-results' to view the report.
echo.
goto :end


REM ----------------------------------------------------------------------
REM SUBROUTINES AND ERROR HANDLERS
REM ----------------------------------------------------------------------

:DETECT_GPU
REM Detect GPU vendor on Windows (robust CMD version)
where nvidia-smi >nul 2>&1
if %ERRORLEVEL%==0 (
    set "GPU_VENDOR=NVIDIA"
    goto :EOF
)

wmic path win32_VideoController get Name >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [WARN] WMIC command failed or is unavailable. Assuming CPU-only mode.
    goto :EOF
)

for /f "usebackq delims=" %%G in (`wmic path win32_VideoController get Name ^| findstr /R /C:"."`) do (
    echo %%G | findstr /I "Intel"  >nul 2>&1 && set "GPU_VENDOR=INTEL"
    echo %%G | findstr /I "Arc"    >nul 2>&1 && set "GPU_VENDOR=INTEL"
    echo %%G | findstr /I "AMD"    >nul 2>&1 && set "GPU_VENDOR=AMD"
    echo %%G | findstr /I "Radeon" >nul 2>&1 && set "GPU_VENDOR=AMD"
)
goto :EOF


:pipfail
echo ------------------------------------------------------
echo [FAILURE] Benchmark run FAILED during dependency installation or verification.
echo ------------------------------------------------------
goto :error

:pytestfail
echo ------------------------------------------------------
echo [FAILURE] Benchmark run FAILED during pytest execution.
echo ------------------------------------------------------
goto :error

:trendfail
echo ------------------------------------------------------
echo [FAILURE] Benchmark run FAILED during Allure report generation.
echo ------------------------------------------------------
goto :error

:error
endlocal
exit /b 1

:end
endlocal
exit /b 0
