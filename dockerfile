# QBench runtime image (CUDA-enabled PyTorch base)
FROM nvcr.io/nvidia/pytorch:25.09-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
  git \
  curl \
  wget \
  ca-certificates \
  build-essential \
  graphviz \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies used across runspace evaluation, experiments, and dashboard.
RUN python -m pip install --no-cache-dir --upgrade pip && \
  python -m pip uninstall -y pynvml || true && \
  python -m pip install --no-cache-dir \
    numpy \
    pandas \
    pyyaml \
    tqdm \
    matplotlib \
    scikit-learn \
    optuna \
    pytest \
    pillow \
    pydot \
    streamlit \
    timm \
    torchview \
    opencv-python-headless \
    nvidia-ml-py

WORKDIR /app

# Keep project-local caches writable when bind-mounting source into /app.
ENV PYTHONPATH=/app \
  TORCH_HOME=/app/.cache/torch \
  PYTORCH_KERNEL_CACHE_PATH=/app/.cache/torch/kernels

CMD ["/bin/bash"]