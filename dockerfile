# Minimal-clean Dockerfile for quantized model evaluation

# 1. Base image: CUDA 12.1 + Ubuntu 22.04 (devel for optional compilation)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 2. Basic OS packages and Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        git \
        curl \
        wget \
        ca-certificates \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Make sure 'python' and 'pip' point to Python 3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 3. Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip

# 4. Install PyTorch 2.4.0 + torchvision + torchaudio (CUDA 12.1 wheels)
#    Note: version numbers here are the official matching set:
#          torch 2.4.0, torchvision 0.19.0, torchaudio 2.4.0
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 5. Core Python dependencies for the quant-eval infrastructure
RUN pip install --no-cache-dir \
    numpy \
    matplotlib \
    pandas \
    scikit-learn \
    pyyaml \
    tqdm \
    optuna \
    pytest

# 6. Working directory for your project
WORKDIR /app

# 7. Default command: interactive shell for development
CMD ["/bin/bash"]