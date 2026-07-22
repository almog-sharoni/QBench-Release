#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# ./apptainer.sh runspace/experiments/find_optimal_input_quant/find_optimal_input_quant.py --model_name vit_b_16 --force_rerun
./apptainer.sh runspace/experiments/find_optimal_input_quant/find_optimal_input_quant.py --model_name mobilevit_s --force_rerun
./apptainer.sh runspace/experiments/find_optimal_input_quant/find_optimal_input_quant.py --model_name vit_b_16 --only_dynamic --metric l1 --force_rerun

./apptainer.sh runspace/experiments/find_optimal_hybrid_quant/find_optimal_hybrid_quant.py --model_name vit_b_16 --weight_mode best --input_mode sweep --batch_size 128 --num_workers 32 --limit_batches -1 --force_rerun
./apptainer.sh runspace/experiments/find_optimal_hybrid_quant/find_optimal_hybrid_quant.py --model_name mobilevit_s --weight_mode best --input_mode sweep --batch_size 128 --num_workers 32 --limit_batches -1 --force_rerun
