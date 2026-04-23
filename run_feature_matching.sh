#!/bin/bash
# Run QBench batch evaluation on feature-matching base configs only.
# Fully detached from the invoking terminal.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/runspace/outputs/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_HOST="$LOG_DIR/run_feature_matching_${TIMESTAMP}.log"
LOG_CONTAINER="runspace/outputs/logs/run_feature_matching_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

docker exec -d qbench bash -c "python runspace/run_all.py --task feature_matching --workers 16 --stop-on-error > ${LOG_CONTAINER} 2>&1"
echo "Started run_all.py (feature_matching) detached in container qbench."
echo "Log: ${LOG_HOST}"
echo "Follow with: tail -f ${LOG_HOST}"
