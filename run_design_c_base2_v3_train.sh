#!/bin/bash
# Design C base-2 v3 — cold-start retrain with disjoint train/val/test splits.
#
# Diffs vs run_design_c_phase2_v2_train.sh:
#   - Points at design_c_t3_base2_v3.yaml:
#       target_pipeline: superpoint_superglue_learned_base2 (pure base-2 head,
#         no torch.exp/log/logsumexp/softmax on the runtime tensor path)
#       no resume_from -> cold-start
#       samples_per_epoch: 40000 (was 20000)
#       val pairs: pairs_val_disjoint_1500.txt (no scene overlap with train/test)
#       out_dir: runspace/outputs/design_c_t3_base2_v3/checkpoints
#
# Schedule: 30 epochs, 40K pairs/epoch (~36 min/epoch) ~ 18h total
# Output:   runspace/outputs/design_c_t3_base2_v3/checkpoints/{ckpt_epNNN.pt, best.pt, history.json}
#
# Runs DETACHED inside the qbench container so the terminal is freed.
# Monitor:  tail -f runspace/outputs/logs/design_c_base2_v3_train_<ts>.log
# Stop:     docker exec qbench pkill -f 'runspace/train.py'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/runspace/outputs/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_HOST="$LOG_DIR/design_c_base2_v3_train_${TIMESTAMP}.log"
LOG_CONTAINER="runspace/outputs/logs/design_c_base2_v3_train_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

docker exec -d qbench bash -c "\
  PYTHONPATH=/app python runspace/train.py \
    --config runspace/inputs/train_configs/design_c_t3_base2_v3.yaml \
    > ${LOG_CONTAINER} 2>&1"

echo "Started Design C base-2 v3 training, detached in container qbench."
echo "Log: ${LOG_HOST}"
echo ""
echo "Monitor:  tail -f ${LOG_HOST}"
echo "Stop:     docker exec qbench pkill -f 'runspace/train.py'"
