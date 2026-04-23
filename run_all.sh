#!/bin/bash
# Run QBench batch evaluation inside the Docker container.
# Fully detached from the invoking terminal: logging happens inside the
# container (to a volume-mounted path), so closing the terminal does NOT
# stop the run or truncate the log.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/runspace/outputs/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_HOST="$LOG_DIR/run_all_${TIMESTAMP}.log"
LOG_CONTAINER="runspace/outputs/logs/run_all_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

docker exec -d qbench bash -c "python runspace/run_all.py --batches 5 --stop-on-error > ${LOG_CONTAINER} 2>&1"
echo "Started run_all.py detached in container qbench."
echo "Log: ${LOG_HOST}"
echo "Follow with: tail -f ${LOG_HOST}"
