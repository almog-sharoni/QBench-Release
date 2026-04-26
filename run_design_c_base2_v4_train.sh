#!/bin/bash
# Design C′ base-2 v4 — Step 2: quantized SP+SG retrofit, warm-started from v3.
#
# Diffs vs run_design_c_base2_v3_train.sh:
#   - Points at design_c_t3_base2_v4.yaml:
#       adapter + quantization blocks: FP8 E4M3 chunk-128 on SP and SG-backbone
#         (kenc + gnn + final_proj). Sinkhorn head stays FP base-2 to remain
#         gradient-trainable.
#       resume_from: design_c_t3_base2_v3/checkpoints/best.pt (head + bin_score)
#       lr: 1e-5 (down from 1e-4)
#       lambda_marg flat at 0.03 (anneal already done in v3)
#       out_dir: runspace/outputs/design_c_t3_base2_v4/checkpoints
#
# Schedule: 30 epochs, 40K pairs/epoch. Quantized fwd/bwd is slower than FP32,
# expect ~1.5x v3's ~36 min/epoch -> ~25-30h total wall time.
# Output:   runspace/outputs/design_c_t3_base2_v4/checkpoints/{ckpt_epNNN.pt, best.pt, history.json}
#
# Runs DETACHED inside the qbench container so the terminal is freed.
# Monitor:  tail -f runspace/outputs/logs/design_c_base2_v4_train_<ts>.log
# Stop:     docker exec qbench pkill -f 'runspace/train.py'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/runspace/outputs/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_HOST="$LOG_DIR/design_c_base2_v4_train_${TIMESTAMP}.log"
LOG_CONTAINER="runspace/outputs/logs/design_c_base2_v4_train_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

docker exec -d qbench bash -c "\
  PYTHONPATH=/app python runspace/train.py \
    --config runspace/inputs/train_configs/design_c_t3_base2_v4.yaml \
    > ${LOG_CONTAINER} 2>&1"

echo "Started Design C base-2 v4 (quantized SP+SG) training, detached in container qbench."
echo "Log: ${LOG_HOST}"
echo ""
echo "Monitor:  tail -f ${LOG_HOST}"
echo "Stop:     docker exec qbench pkill -f 'runspace/train.py'"
