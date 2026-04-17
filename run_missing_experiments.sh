#!/bin/bash
set -e
cd /app

W_FMTS="fp6_e1m4,fp6_e2m3,fp6_e3m2,fp6_e4m1,fp6_e5m0,fp6_e0m5"
A_FMTS="fp4_e1m2,fp4_e2m1,fp4_e3m0,fp4_e0m3"
DATASET="--dataset_type classification --dataset_name imagenet --dataset_path /data/imagenet/val --num_workers 4"

echo "[$(date)] ===== W6A4: mobilevit_xxs ====="
python3 runspace/experiments/find_optimal_w6a4/find_optimal_w6a4.py \
  --model_name mobilevit_xxs --model_source timm \
  --weight_formats "$W_FMTS" --activation_formats "$A_FMTS" $DATASET

echo "[$(date)] ===== W6A4: vit_b_16 ====="
python3 runspace/experiments/find_optimal_w6a4/find_optimal_w6a4.py \
  --model_name vit_b_16 \
  --weight_formats "$W_FMTS" --activation_formats "$A_FMTS" $DATASET

echo "[$(date)] ===== W6A4: resnet152 ====="
python3 runspace/experiments/find_optimal_w6a4/find_optimal_w6a4.py \
  --model_name resnet152 \
  --weight_formats "$W_FMTS" --activation_formats "$A_FMTS" $DATASET

echo "[$(date)] ===== weight_quant_baseline: mobilevit_xxs ====="
python3 runspace/experiments/find_optimal_weight_quant/find_optimal_weight_quant.py \
  --model_name mobilevit_xxs --run_eval --skip_layer_wise \
  --dataset_type classification --dataset_name imagenet \
  --dataset_path /data/imagenet/val --num_workers 4

echo "[$(date)] ===== weight_quant_baseline: vit_b_16 ====="
python3 runspace/experiments/find_optimal_weight_quant/find_optimal_weight_quant.py \
  --model_name vit_b_16 --run_eval --skip_layer_wise \
  --dataset_type classification --dataset_name imagenet \
  --dataset_path /data/imagenet/val --num_workers 4

echo "[$(date)] ===== weight_quant_baseline: resnet152 ====="
python3 runspace/experiments/find_optimal_weight_quant/find_optimal_weight_quant.py \
  --model_name resnet152 --run_eval --skip_layer_wise \
  --dataset_type classification --dataset_name imagenet \
  --dataset_path /data/imagenet/val --num_workers 4

echo "[$(date)] ===== hybrid_best_combo: mobilevit_xxs ====="
python3 runspace/experiments/find_optimal_hybrid_quant/find_optimal_hybrid_quant.py \
  --model_name mobilevit_xxs \
  --weight_metrics l1,mse --input_metrics mse,l1 \
  --weight_chunk_size 128 --input_chunk_size 128 $DATASET

echo "[$(date)] ===== hybrid_best_combo: vit_b_16 ====="
python3 runspace/experiments/find_optimal_hybrid_quant/find_optimal_hybrid_quant.py \
  --model_name vit_b_16 \
  --weight_metrics l1,mse --input_metrics mse,l1 \
  --weight_chunk_size 128 --input_chunk_size 128 $DATASET

echo "[$(date)] ===== ALL DONE ====="
