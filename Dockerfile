# =============================================================
# GPU Benchmark Suite Dockerfile
# =============================================================
# Author: Bang Thien Nguyen
# Description: Container for running GPU benchmark tests with PyTorch and Allure
# Base image includes CUDA support if available; fallback to CPU
# =============================================================

# ---- Base Image ----
FROM nvidia/cuda:12.2.0-cudnn11.8-runtime-ubuntu22.04

# ---- Set Working Directory ----
WORKDIR /app

# ---- Environment Variables ----
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PIP_NO_CACHE_DIR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV PIP_DEFAULT_TIMEOUT=100

# ---- System Dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    curl wget git unzip \
    build-essential cmake pkg-config \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Allure CLI ----
RUN wget -qO- https://github.com/allure-framework/allure2/releases/download/2.27.2/allure-2.27.2.tgz | tar -xz -C /opt/ && \
    ln -s /opt/allure-2.27.2/bin/allure /usr/bin/allure

# ---- Upgrade pip ----
RUN python3.10 -m pip install --upgrade pip

# ---- Install Python Packages ----
COPY requirements.txt /app/requirements.txt
RUN python3.10 -m pip install --upgrade setuptools wheel
RUN python3.10 -m pip install -r requirements.txt

# ---- Copy Benchmark Scripts ----
COPY . /app

# ---- Expose Allure default port ----
EXPOSE 8080

# ---- Default Command ----
CMD ["python3.10", "gpu_benchmark.py"]
