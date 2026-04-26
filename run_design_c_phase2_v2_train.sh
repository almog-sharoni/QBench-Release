#!/bin/bash
# Design C Phase 2 V2 — full head-only training with loss-runaway fix.
#
# Diffs vs run_design_c_phase2_train.sh:
#   - Points at design_c_t3_scannet_v2.yaml:
#       lambda_marg_initial 0.1 -> 1.0, lambda_marg_final 0.01 -> 0.1
#       samples_per_epoch: 20000 (fresh random subset per epoch)
#       out_dir: runspace/outputs/design_c_t3_v2/checkpoints
#   - losses.py now applies marginal_l2_loss on every trace iteration and
#     asserts non-negative match loss (runspace/src/train/losses.py).
#
# Schedule: 30 epochs, 20K pairs/epoch (~18 min/epoch) ~ 9h total at current step rate
# Output:   runspace/outputs/design_c_t3_v2/checkpoints/{ckpt_epNNN.pt, best.pt, history.json}
#
# Runs DETACHED inside the qbench container so the terminal is freed.
# Monitor progress with:   tail -f runspace/outputs/logs/phase2_v2_train_<ts>.log
# Stop with:               docker exec qbench pkill -f 'runspace/train.py'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/runspace/outputs/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_HOST="$LOG_DIR/phase2_v2_train_${TIMESTAMP}.log"
LOG_CONTAINER="runspace/outputs/logs/phase2_v2_train_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

docker exec -d qbench bash -c "\
  PYTHONPATH=/app python runspace/train.py \
    --config runspace/inputs/train_configs/design_c_t3_scannet_v2.yaml \
    > ${LOG_CONTAINER} 2>&1"

echo "Started Design C Phase 2 V2 training, detached in container qbench."
echo "Log: ${LOG_HOST}"
echo ""
echo "Monitor:  tail -f ${LOG_HOST}"
echo "Stop:     docker exec qbench pkill -f 'runspace/train.py'"
