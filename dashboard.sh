#!/bin/bash
# Wrapper script to run the QBench Experiment Dashboard within the Apptainer sandbox.

SANDBOX_DIR="qbench_sandbox"
DASHBOARD_PY="runspace/src/database/dashboard.py"
PORT=8501

# Verify sandbox exists
if [ ! -d "$SANDBOX_DIR" ]; then
    echo "Error: Apptainer sandbox '$SANDBOX_DIR' not found."
    exit 1
fi

# Kill process using the port if it exists
PID=$(lsof -t -i:$PORT)
if [ ! -z "$PID" ]; then
    echo "Port $PORT is in use by PID $PID. Killing it..."
    kill -9 $PID
fi

echo "Starting QBench Dashboard..."
echo "Access at: http://localhost:$PORT"

# Run streamlit inside the container
# Create state directory for tailscale persistence if not exists
mkdir -p tailscale_state

# Run streamlit and tailscale inside the container via the wrapper script
apptainer exec --nv --env PYTHONNOUSERSITE=1 \
    --bind /data/shared_data/imagenet:/data/imagenet \
    --bind "$(pwd)/tailscale_state":/var/lib/tailscale \
    "$SANDBOX_DIR" /usr/local/bin/start_tailscale_app.sh "$DASHBOARD_PY" "$PORT"

