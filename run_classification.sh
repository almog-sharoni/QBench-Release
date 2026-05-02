#!/bin/bash
# Run QBench batch evaluation on classification (adapter.type=generic) base configs only.
# Fully detached from the invoking terminal.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/runspace/outputs/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_HOST="$LOG_DIR/run_classification_${TIMESTAMP}.log"
LOG_CONTAINER="runspace/outputs/logs/run_classification_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

docker exec -d \
  -e TORCH_CUDA_ARCH_LIST=9.0 \
  -e TORCH_EXTENSIONS_DIR=/app/runspace/.torch_extensions \
  qbench bash -c "python runspace/run_all.py --task classification --batch-size 32 --epochs 5 --stop-on-error > ${LOG_CONTAINER} 2>&1"
echo "Started run_all.py (classification) detached in container qbench."
echo "Log: ${LOG_HOST}"
echo "Follow with: tail -f ${LOG_HOST}"
