#!/bin/bash
# Wrapper script to run QBench interactions within the Apptainer sandbox.

# Path to the sandbox directory
SANDBOX_DIR="qbench_sandbox"

# Verify sandbox exists
if [ ! -d "$SANDBOX_DIR" ]; then
    echo "Error: Apptainer sandbox '$SANDBOX_DIR' not found."
    echo "Please build it first using instructions in readme-apptainer.md."
    exit 1
fi

# Run the interactive script with CUDA enabled and ImageNet volume bound.
# We map the ImageNet data path so the warning is resolved automatically.
apptainer exec --nv --bind /data/shared_data/imagenet:/data/imagenet "$SANDBOX_DIR" python "$@"
