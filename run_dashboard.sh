#!/bin/bash
# Wrapper script to run the QBench Experiment Dashboard within the Apptainer sandbox.

SANDBOX_DIR="qbench_sandbox"
STREAMLIT_BIN="/home/almog/.local/bin/streamlit"
DASHBOARD_PY="runspace/src/database/dashboard.py"

# Verify sandbox exists
if [ ! -d "$SANDBOX_DIR" ]; then
    echo "Error: Apptainer sandbox '$SANDBOX_DIR' not found."
    exit 1
fi

echo "Starting QBench Dashboard..."
echo "Access at: http://localhost:8501"

# Run streamlit
apptainer exec --nv --bind /data/shared_data/imagenet:/data/imagenet "$SANDBOX_DIR" "$STREAMLIT_BIN" run "$DASHBOARD_PY" --server.port 8501
