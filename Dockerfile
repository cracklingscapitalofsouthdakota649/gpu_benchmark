FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Use Python 3.10
RUN apt-get update && apt-get install -y python3.10 python3.10-venv python3-pip git wget

# Create venv
RUN python3.10 -m venv /venv310
ENV PATH="/venv310/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install \
    torch \
    torchvision \
    torchaudio \
    pytest \
    pytest-benchmark \
    allure-pytest\
	matplotlib\
	psutil

# Set workdir
WORKDIR /app
COPY . /app

# Default command
CMD ["pytest", "--alluredir=allure-results", "-m", "preprocess"]
