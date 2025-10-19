@echo off
REM =============================================================
REM Run GPU Benchmark Suite in Docker with Allure Reporting
REM =============================================================

REM -------- Configuration --------
SET IMAGE_NAME=gpu-benchmark-local:latest
SET CONTAINER_NAME=gpu_benchmark_container
SET HOST_DIR=%CD%
SET RESULTS_DIR=%HOST_DIR%\allure-results
SET REPORT_DIR=%HOST_DIR%\allure-report

REM -------- Clean Previous Results --------
echo [INFO] Cleaning old Allure results and reports...
IF EXIST "%RESULTS_DIR%" rmdir /s /q "%RESULTS_DIR%"
IF EXIST "%REPORT_DIR%" rmdir /s /q "%REPORT_DIR%"

REM -------- Check if Image Exists --------
docker image inspect %IMAGE_NAME% >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo [INFO] Docker image %IMAGE_NAME% already exists. Skipping build.
) ELSE (
    echo [INFO] Building Docker image: %IMAGE_NAME%...
    docker build -t %IMAGE_NAME% .
)

REM -------- Run Container --------
echo [INFO] Running benchmark tests inside Docker container...
docker run --rm -v "%HOST_DIR%":/app -w /app %IMAGE_NAME% python gpu_benchmark.py

REM -------- Generate Allure Report --------
echo [INFO] Generating Allure report...
allure generate "%RESULTS_DIR%" -o "%REPORT_DIR%" --clean

REM -------- Serve Allure Report --------
echo [INFO] Serving Allure report at http://localhost:8080 ...
allure serve "%RESULTS_DIR%"

pause
