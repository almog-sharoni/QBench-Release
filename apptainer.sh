#!/bin/bash
# Wrapper script to run QBench interactions within the Apptainer sandbox.

SANDBOX_DIR="qbench_sandbox"
LOG_FILE="exec_commands.log"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/runspace/.torch_extensions}"
# Project-local site for packages that aren't in the read-only sandbox image
# (e.g. transformers/datasets/tokenizers for SLM/LLM experiments).
PYLIBS_DIR="$PWD/runspace/.pylibs"
# Project-local HuggingFace cache so model/dataset downloads are reproducible
# and don't depend on a writable $HOME.
HF_HOME_DIR="${HF_HOME:-$PWD/.cache/huggingface}"

# Verify sandbox exists
if [ ! -d "$SANDBOX_DIR" ]; then
    echo "Error: Apptainer sandbox '$SANDBOX_DIR' not found."
    exit 1
fi

# Build the exact command string
CMD="apptainer exec --nv --env PYTHONNOUSERSITE=1 --env TORCH_CUDA_ARCH_LIST=9.0 --env TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR --env TORCH_HOME=$HOME/.cache/torch --env PYTHONPATH=$PWD:$PYLIBS_DIR --env HF_HOME=$HF_HOME_DIR --env UCX_HANDLE_ERRORS=none --env RDMAV_FORK_SAFE=1 --env IBV_FORK_SAFE=1 --bind /data/shared_data/imagenet:/data/imagenet --bind $PWD:/app --bind /data/shared_data/scannet:/data/scannet \"$SANDBOX_DIR\" python $*"

# Append command to log
echo "$CMD" >> "$LOG_FILE"

# Run command
eval $CMD
