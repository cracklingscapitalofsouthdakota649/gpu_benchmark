# =============================================================
# GPU Benchmark Suite Dockerfile (Intel oneAPI)
# =============================================================
# Author: Bang Thien Nguyen
# Description: Container for running GPU benchmark tests with PyTorch
#              targeting Intel GPUs via oneAPI. Includes Allure CLI.
# =============================================================

# ---- Base Image (Intel oneAPI GPU Runtime) ----
# This image includes the necessary Intel GPU drivers, Level Zero, and OpenCL runtimes.
FROM intel/oneapi-gpu-runtime:latest-ubuntu-22.04

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
# This helps the application find the correct compute device.
ENV SYCL_DEVICE_FILTER=LEVEL_ZERO:GPU 
ENV PATH="${PATH}:/opt/allure-2.29.0/bin"

# ===================================================================
# FIX: Install Java and Allure CLI for report generation
# ===================================================================

# 1. Update package lists with retry logic
RUN for i in 1 2 3 4 5; do apt-get update && break || sleep 5; done

# 2. Install necessary system dependencies (Java JRE, wget, unzip)
RUN apt-get install -y --no-install-recommends \
    python3.10 python3-pip \
    openjdk-21-jre-headless \
    wget \
    unzip && \
    rm -rf /var/lib/apt/lists/*

# 3. Download and configure the Allure Command Line tool
ENV ALLURE_VERSION=2.29.0
RUN wget -qO /tmp/allure-commandline.zip https://repo.maven.apache.org/maven2/io/qameta/allure/allure-commandline/${ALLURE_VERSION}/allure-commandline-${ALLURE_VERSION}.zip && \
    unzip /tmp/allure-commandline.zip -d /opt && \
    rm /tmp/allure-commandline.zip

# 4. Add the Allure executable to the system PATH (already set via ENV)

# ===================================================================
# Python Dependency Installation
# ===================================================================

# Check if requirements.txt exists in the build context and copy it.
# Note: The Intel base image already has Python, we use the installed version.
COPY requirements.txt .

# Ensure requirements.txt exists by creating an empty one if the previous COPY failed.
RUN if [ ! -f requirements.txt ]; then echo "" > requirements.txt; fi

# Install core project dependencies, including the necessary testing packages
RUN python3.10 -m pip install --upgrade pip
RUN python3.10 -m pip install --no-cache-dir \
    -r requirements.txt || true

# Copy the rest of the application code into the container
COPY . /app

CMD ["/usr/local/bin/python", "-m", "pytest" -m "preprocess"]