
# Use Intel oneAPI runtime base image (Ubuntu 22.04)
FROM intel/oneapi-runtime:2024.1.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies and python3-pip
RUN set -ex; \
    for i in 1 2 3 4 5; do apt-get update && break || sleep 5; done; \
    apt-get install -y python3-pip \
	openjdk-21-jre-headless \
    wget \
    unzip && \
    rm -rf /var/lib/apt/lists/*
	
# Copy requirements.txt
COPY requirements.txt .

# Ensure requirements.txt exists
RUN if [ ! -f requirements.txt ]; then echo "" > requirements.txt; fi

# Install Python dependencies
RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install --no-cache-dir -r requirements.txt

# Copy benchmark scripts
COPY . /app

# Default command to run pytest benchmarks
CMD ["pytest", "-m", "gpu"]
