# DGX Spark best-practice base (per NVIDIA Spark guide)
FROM nvcr.io/nvidia/pytorch:24.08-py3

ENV DEBIAN_FRONTEND=noninteractive

# Optional: same tools you had
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates build-essential \
  && rm -rf /var/lib/apt/lists/*

# Upgrade pip (still fine)
RUN python -m pip install --no-cache-dir --upgrade pip

# Your extra Python deps (same as before)
RUN python -m pip install --no-cache-dir \
    numpy matplotlib pandas scikit-learn pyyaml tqdm optuna pytest

WORKDIR /app
CMD ["/bin/bash"]