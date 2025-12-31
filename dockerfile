# DGX Spark-compatible PyTorch (CUDA 13.x, Blackwell-ready)
FROM nvcr.io/nvidia/pytorch:25.09-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
      numpy matplotlib pandas scikit-learn pyyaml tqdm optuna pytest

WORKDIR /app
CMD ["/bin/bash"]