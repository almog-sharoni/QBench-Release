#!/bin/bash
# Wrapper script to run QBench interactions within the Apptainer sandbox.

SANDBOX_DIR="qbench_sandbox"
LOG_FILE="exec_commands.log"

# Verify sandbox exists
if [ ! -d "$SANDBOX_DIR" ]; then
    echo "Error: Apptainer sandbox '$SANDBOX_DIR' not found."
    exit 1
fi

# Build the exact command string
CMD="apptainer exec --nv --env PYTHONNOUSERSITE=1 --env TORCH_CUDA_ARCH_LIST=9.0 --env TORCH_HOME=$HOME/.cache/torch --env PYTHONPATH=$PWD --bind /data/shared_data/imagenet:/data/imagenet \"$SANDBOX_DIR\" python $*"

# Append command to log
echo "$CMD" >> "$LOG_FILE"

# Run command
eval $CMD