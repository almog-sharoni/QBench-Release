#!/bin/bash
# Design C' Phase 2b Base2 — warm-start head training in base-2 arithmetic.
#
# Warm start:  runspace/outputs/design_c_t3/checkpoints/best.pt (current
#              Phase 2 best, pose_auc_10 reference). The plan doc spec
#              (ckpt_ep030.pt) is used verbatim when a full Phase 2 run
#              produces it; for now we repoint to best.pt.
# Config:      runspace/inputs/train_configs/design_c_t3_base2.yaml
# Schedule:    10 epochs, 20K pairs/epoch (~18 min/epoch) ~ 3h total
# Output:      runspace/outputs/design_c_t3_base2/checkpoints/
#                {ckpt_epNNN.pt, best.pt, history.json}
#
# Before running this, validate numerical equivalence pre-training:
#   docker exec qbench python runspace/src/scripts/verify_base2_equivalence.py
# The V.1 gate requires max |log_T - log2_T / log2(e)| < 1e-3 on 300 val pairs.
#
# Runs DETACHED inside the qbench container.
# Monitor progress with:   tail -f runspace/outputs/logs/phase2b_base2_train_<ts>.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/runspace/outputs/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_HOST="$LOG_DIR/phase2b_base2_train_${TIMESTAMP}.log"
LOG_CONTAINER="runspace/outputs/logs/phase2b_base2_train_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

docker exec -d qbench bash -c "\
  PYTHONPATH=/app python runspace/train.py \
    --config runspace/inputs/train_configs/design_c_t3_base2.yaml \
    > ${LOG_CONTAINER} 2>&1"

echo "Started Design C' Phase 2b Base2 training, detached in container qbench."
echo "Log: ${LOG_HOST}"
echo ""
echo "Monitor:  tail -f ${LOG_HOST}"
echo "Stop:     docker exec qbench pkill -f 'runspace/train.py'"
