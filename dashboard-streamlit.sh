#!/usr/bin/env bash
set -euo pipefail

# Compatibility launcher for the original Streamlit dashboard.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANDBOX_DIR="$PROJECT_DIR/qbench_sandbox"
DASHBOARD_PY="runspace/src/database/dashboard.py"
PORT="${PORT:-8501}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PROJECT_DIR/runspace/.torch_extensions}"

if [[ ! -d "$SANDBOX_DIR" ]]; then
    echo "Error: Apptainer sandbox '$SANDBOX_DIR' not found." >&2
    exit 1
fi

if command -v lsof >/dev/null 2>&1; then
    # Match only a TCP listener. A broad `-i:$PORT` also matches outbound
    # Tailscale connections whose remote endpoint happens to use this port.
    EXISTING_PID="$(lsof -nP -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | head -n 1 || true)"
    if [[ -n "$EXISTING_PID" ]]; then
        echo "Error: port $PORT is already in use by PID $EXISTING_PID." >&2
        echo "Stop that process or choose another port with PORT=<port>." >&2
        exit 1
    fi
fi

echo "Starting the legacy QBench Streamlit Dashboard at http://localhost:$PORT"
mkdir -p "$PROJECT_DIR/tailscale_state"
cd "$PROJECT_DIR"
exec apptainer exec --nv --env PYTHONNOUSERSITE=1 \
    --env TORCH_CUDA_ARCH_LIST=9.0 \
    --env TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR" \
    --bind /data/shared_data/imagenet:/data/imagenet \
    --bind "$PROJECT_DIR/tailscale_state":/var/lib/tailscale \
    "$SANDBOX_DIR" /usr/local/bin/start_tailscale_app.sh "$DASHBOARD_PY" "$PORT"
