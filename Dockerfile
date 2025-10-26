# =============================================================
# GPU Benchmark Suite Dockerfile (Intel oneAPI)
# =============================================================
# Author: Bang Thien Nguyen (Target for Intel GPU on Windows 11/oneAPI)
# Description: Container for running GPU benchmark tests with PyTorch
#              targeting Intel GPUs via oneAPI. Includes Allure CLI.
# =============================================================

# ---- Base Image (Intel oneAPI GPU Runtime) ----
# FIX: Switched from the old NVIDIA base to a stable Intel oneAPI runtime 
# image, which provides the necessary Level Zero/OpenCL drivers and SYCL environment.
FROM intel/oneapi-runtime:2024.1.0-devel-ubuntu22.04

# ---- Set Working Directory ----
WORKDIR /app

# ---- Environment Variables ----
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# --- CRUCIAL INTEL/SYCL ENV VARS ---
# Explicitly tell the SYCL runtime to use Intel Level Zero for GPU/dGPU.
ENV SYCL_DEVICE_FILTER=LEVEL_ZERO:GPU 
ENV PATH="${PATH}:/opt/allure-2.29.0/bin"

# ===================================================================
# System, Java, and Allure CLI Installation (CONSOLIDATED FOR SMALLER IMAGE)
# ===================================================================

ENV ALLURE_VERSION=2.29.0

# Consolidated RUN command: 
# 1. Updates packages.
# 2. Installs all system dependencies (Python, Java, build tools).
# 3. Downloads Allure.
# 4. Unzips Allure.
# 5. CLEANS UP temporary files and package lists (crucial for size).
RUN set -ex; \
    # 1. Update package lists with retry logic
    for i in 1 2 3 4 5; do apt-get update && break || sleep 5; done; \
    \
    # 2. Install necessary system dependencies
    apt-get install -y --no-install-recommends \
        openjdk-21-jre-headless \
        wget \
        unzip; \
    \
    # 3. Download and configure the Allure Command Line tool
    wget -qO /tmp/allure-commandline.zip https://repo.maven.apache.org/maven2/io/qameta/allure/allure-commandline/${ALLURE_VERSION}/allure-commandline-${ALLURE_VERSION}.zip; \
    unzip /tmp/allure-commandline.zip -d /opt; \
    \
    # 4. Clean up everything installed/downloaded in this layer
    rm /tmp/allure-commandline.zip; \
    rm -rf /var/lib/apt/lists/*

# ===================================================================
# Python Dependency Installation
# ===================================================================

# Copy requirements.txt, ensure it exists, and install dependencies
COPY requirements.txt .
RUN if [ ! -f requirements.txt ]; then echo "" > requirements.txt; fi

# Consolidated RUN command for pip operations
RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install --no-cache-dir -r requirements.txt || true

# Copy the rest of the application code into the container
COPY . /app
