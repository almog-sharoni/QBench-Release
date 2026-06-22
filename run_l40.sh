#!/bin/bash
# Wrapper script to dispatch QBench jobs to the L40 (amls) Ray node.

SANDBOX_DIR="qbench_sandbox"
RAY_DASHBOARD_URL="http://132.70.226.91:8265"

# Verify sandbox exists (needed for local Ray job CLI submission)
if [ ! -d "$SANDBOX_DIR" ]; then
    echo "Error: Apptainer sandbox '$SANDBOX_DIR' not found."
    exit 1
fi

# Translate absolute paths matching the current directory to relative paths
# so they resolve correctly inside the remote Ray job working directory.
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "$PWD"* ]]; then
        # Strip the PWD prefix (including trailing slash if present)
        rel_path="${arg#$PWD/}"
        ARGS+=("$rel_path")
    else
        ARGS+=("$arg")
    fi
done

# Run ray job submit inside the local container
# This packages the code in /app and submits it to the remote Ray cluster.
# The entrypoint is executed directly using the remote node's Docker python
# environment which already contains all CUDA and library dependencies.
apptainer exec --nv \
    --env PYTHONNOUSERSITE=1 \
    --env TORCH_CUDA_ARCH_LIST=9.0 \
    --env UCX_HANDLE_ERRORS=none \
    --env RDMAV_FORK_SAFE=1 \
    --env IBV_FORK_SAFE=1 \
    --bind "$PWD":/app \
    "$SANDBOX_DIR" \
    ray job submit --address "$RAY_DASHBOARD_URL" \
        --working-dir /app \
        --entrypoint-num-gpus 1 \
        --entrypoint-resources '{"l40": 1}' \
        -- env PYTHONUNBUFFERED=1 python -u "${ARGS[@]}"
