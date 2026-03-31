#!/bin/bash
# Wrapper script to run the QBench Experiment Dashboard within the Apptainer sandbox.

SANDBOX_DIR="qbench_sandbox"
DASHBOARD_PY="runspace/src/database/dashboard.py"

# Verify sandbox exists
if [ ! -d "$SANDBOX_DIR" ]; then
    echo "Error: Apptainer sandbox '$SANDBOX_DIR' not found."
    exit 1
fi

echo "Starting QBench Dashboard..."
echo "Access at: http://localhost:8501"

# Run streamlit inside the container
apptainer exec --nv --env PYTHONNOUSERSITE=1 --bind /data/shared_data/imagenet:/data/imagenet "$SANDBOX_DIR" python -m streamlit run "$DASHBOARD_PY" --server.port 8501
