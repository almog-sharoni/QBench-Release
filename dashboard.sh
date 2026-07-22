#!/usr/bin/env bash
set -euo pipefail

# QBench's default dashboard is the original Streamlit application. The
# Dashboard 1 Svelte + SQLite rewrite remains available as an explicit mode.
#
# Usage:
#   ./dashboard.sh              original Streamlit dashboard
#   ./dashboard.sh new          rewritten production dashboard on port 8501
#   ./dashboard.sh dev          rewritten Vite UI (5173) + API (8501)
#   ./dashboard.sh streamlit    original Streamlit dashboard

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASHBOARD_DIR="$PROJECT_DIR/dashboard1"
MODE="${1:-streamlit}"
PORT="${PORT:-8501}"

case "$MODE" in
    legacy|streamlit)
        exec "$PROJECT_DIR/dashboard-streamlit.sh"
        ;;
    new|modern|prod)
        MODE="prod"
        ;;
    dev)
        ;;
    *)
        echo "Unknown dashboard mode: $MODE" >&2
        echo "Usage: ./dashboard.sh [streamlit|new|dev]" >&2
        exit 2
        ;;
esac

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    echo "Error: Node.js and npm are required for Dashboard 1." >&2
    echo "Use './dashboard.sh' to start the original Streamlit dashboard." >&2
    exit 1
fi

# Fail before building when an older dashboard (or any other service) still
# owns the requested port. Without this check, Node reports EADDRINUSE while a
# browser continues showing the already-running Streamlit page.
if command -v curl >/dev/null 2>&1; then
    PORT_PROBE="$(curl --silent --max-time 1 --write-out $'\n__QBENCH_HTTP__%{http_code}' "http://127.0.0.1:$PORT/api/health" 2>/dev/null || true)"
    PORT_STATUS="${PORT_PROBE##*__QBENCH_HTTP__}"
    PORT_BODY="${PORT_PROBE%$'\n__QBENCH_HTTP__'*}"
    if [[ "$PORT_STATUS" != "000" && "$PORT_STATUS" != "$PORT_PROBE" ]]; then
        if [[ "$PORT_BODY" == *'"status":"ok"'* && "$PORT_BODY" == *'"dbDirectory"'* ]]; then
            echo "Dashboard 1 is already running at http://localhost:$PORT." >&2
        elif [[ "$PORT_BODY" == *"Streamlit"* || "$PORT_BODY" == *"streamlit"* ]]; then
            echo "Error: the Streamlit dashboard is currently using port $PORT." >&2
            echo "Stop it, then run './dashboard.sh new' again." >&2
        else
            echo "Error: port $PORT is already used by another HTTP service." >&2
            echo "Stop that service or choose another port, for example: PORT=8502 ./dashboard.sh" >&2
        fi
        exit 1
    fi
fi

if [[ ! -d "$DASHBOARD_DIR/node_modules" ]]; then
    echo "Installing dashboard dependencies from package-lock.json..."
    (cd "$DASHBOARD_DIR" && npm ci)
fi

if [[ "$MODE" == "dev" ]]; then
    echo "Starting QBench Dashboard 1 in development mode."
    echo "UI:  http://localhost:5173"
    echo "API: http://localhost:$PORT"
    (cd "$DASHBOARD_DIR" && PORT="$PORT" npm run dev)
else
    echo "Building QBench Dashboard 1..."
    (cd "$DASHBOARD_DIR" && npm run build)
    echo "Starting QBench Dashboard 1 at http://localhost:$PORT"
    cd "$DASHBOARD_DIR"
    exec env PORT="$PORT" NODE_ENV=production node server/index.js
fi
