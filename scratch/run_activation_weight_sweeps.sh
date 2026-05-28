#!/usr/bin/env bash
set -euo pipefail

# Run separate activation-input and weight-only quantization sweeps.
# Baselines use all 4-8 bit fixed formats. Activation baselines also include an
# fp32 reference first so empty DBs can compute accuracy drop. Dynamic/optimized
# runs are split by bit width and use per-chunk selection only.

MODEL_NAME="${MODEL_NAME:-resnet18}"
MODELS_FILE="${MODELS_FILE:-}"
WEIGHTS="${WEIGHTS:-DEFAULT}"
DATASET_NAME="${DATASET_NAME:-imagenet}"
DATASET_PATH="${DATASET_PATH:-/data/imagenet/val}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-32}"
LIMIT_BATCHES="${LIMIT_BATCHES:--1}"
CHUNK_SIZE="${CHUNK_SIZE:-128}"
INPUT_SIZE="${INPUT_SIZE:-224}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runspace/experiments/baselines_vs_dynamic_runs}"

RUN_ACTIVATIONS="${RUN_ACTIVATIONS:-1}"
RUN_WEIGHTS="${RUN_WEIGHTS:-1}"
RUN_BASELINES="${RUN_BASELINES:-1}"
RUN_DYNAMIC="${RUN_DYNAMIC:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
FORCE_RECALC="${FORCE_RECALC:-0}"
SKIP_INPUT_ERROR_STATS="${SKIP_INPUT_ERROR_STATS:-1}"
DRY_RUN="${DRY_RUN:-0}"

# The weight experiment logs optimized per-chunk runs as weight_dt=opt_chunk_mse.
# Force those split-by-bit runs by default so 8/7/6/5/4 bit candidate groups all
# execute even if a previous opt_chunk_mse row exists in the DB.
WEIGHT_DYNAMIC_FORCE_RERUN="${WEIGHT_DYNAMIC_FORCE_RERUN:-1}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUNNER=("${RUNNER_BIN:-./apptainer.sh}")
INPUT_SCRIPT="runspace/experiments/find_optimal_input_quant/find_optimal_input_quant.py"
WEIGHT_SCRIPT="runspace/experiments/find_optimal_weight_quant/find_optimal_weight_quant.py"

FP8_FORMATS="fp8_e1m6,fp8_e2m5,fp8_e3m4,fp8_e4m3,fp8_e5m2,fp8_e6m1,fp8_e7m0"
FP7_FORMATS="fp7_e1m5,fp7_e2m4,fp7_e3m3,fp7_e4m2,fp7_e5m1,fp7_e6m0"
FP6_FORMATS="fp6_e1m4,fp6_e2m3,fp6_e3m2,fp6_e4m1,fp6_e5m0"
FP5_FORMATS="fp5_e1m3,fp5_e2m2,fp5_e3m1,fp5_e4m0"
FP4_FORMATS="fp4_e1m2,fp4_e2m1,fp4_e3m0"
BASELINE_4_8_FORMATS="${FP8_FORMATS},${FP7_FORMATS},${FP6_FORMATS},${FP5_FORMATS},${FP4_FORMATS}"
ACTIVATION_BASELINE_FORMATS="${ACTIVATION_BASELINE_FORMATS:-fp32,${BASELINE_4_8_FORMATS}}"

BIT_SWEEPS=(
  "8:${FP8_FORMATS}"
  "7:${FP7_FORMATS}"
  "6:${FP6_FORMATS}"
  "5:${FP5_FORMATS}"
  "4:${FP4_FORMATS}"
)

COMMON_ARGS=(
  --weights "$WEIGHTS"
  --dataset_name "$DATASET_NAME"
  --dataset_path "$DATASET_PATH"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --limit_batches "$LIMIT_BATCHES"
)

if [[ -n "$MODELS_FILE" ]]; then
  COMMON_ARGS+=(--models_file "$MODELS_FILE")
else
  COMMON_ARGS+=(--model_name "$MODEL_NAME")
fi

maybe_force_rerun_args=()
if [[ "$FORCE_RERUN" == "1" ]]; then
  maybe_force_rerun_args+=(--force_rerun)
fi

maybe_force_recalc_args=()
if [[ "$FORCE_RECALC" == "1" ]]; then
  maybe_force_recalc_args+=(--force_recalc)
fi

maybe_skip_input_error_stats_args=()
if [[ "$SKIP_INPUT_ERROR_STATS" == "1" ]]; then
  maybe_skip_input_error_stats_args+=(--skip_input_error_stats)
fi

run_cmd() {
  printf '\n>>'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

if [[ "$RUN_ACTIVATIONS" == "1" && "$RUN_BASELINES" == "1" ]]; then
  run_cmd "${RUNNER[@]}" "$INPUT_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --output_dir "${OUTPUT_ROOT}/activations/baselines_4_8" \
    --chunk_size "$CHUNK_SIZE" \
    --input_size "$INPUT_SIZE" \
    --baseline_formats "$ACTIVATION_BASELINE_FORMATS" \
    --candidate_formats "$BASELINE_4_8_FORMATS" \
    --experiment_type "input_quant_baseline_4_8" \
    --only_baselines \
    --unsigned_input_sources "relu,softmax,quantrelu,quantsoftmax" \
    --dynamic_unsigned_input_candidates \
    --fold_input_norm \
    "${maybe_skip_input_error_stats_args[@]}" \
    "${maybe_force_rerun_args[@]}"
fi

if [[ "$RUN_ACTIVATIONS" == "1" && "$RUN_DYNAMIC" == "1" ]]; then
  for sweep in "${BIT_SWEEPS[@]}"; do
    bit="${sweep%%:*}"
    formats="${sweep#*:}"
    run_cmd "${RUNNER[@]}" "$INPUT_SCRIPT" \
      "${COMMON_ARGS[@]}" \
      --output_dir "${OUTPUT_ROOT}/activations/dynamic_${bit}bit" \
      --chunk_size "$CHUNK_SIZE" \
      --input_size "$INPUT_SIZE" \
      --baseline_formats "$formats" \
      --candidate_formats "$formats" \
      --dynamic_experiment_type "input_quant_dynamic_${bit}bit" \
      --only_dynamic \
      --metric "mse" \
      --unsigned_input_sources "relu,softmax,quantrelu,quantsoftmax" \
      --dynamic_unsigned_input_candidates \
      --fold_input_norm \
      "${maybe_force_rerun_args[@]}"
  done
fi

if [[ "$RUN_WEIGHTS" == "1" && "$RUN_BASELINES" == "1" ]]; then
  run_cmd "${RUNNER[@]}" "$WEIGHT_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --output_dir "${OUTPUT_ROOT}/weights/baselines_4_8" \
    --metrics "mse" \
    --weight_chunk_size "$CHUNK_SIZE" \
    --baseline_formats "$BASELINE_4_8_FORMATS" \
    --run_eval \
    --skip_layer_wise \
    --fold_input_norm \
    "${maybe_force_rerun_args[@]}" \
    "${maybe_force_recalc_args[@]}"
fi

if [[ "$RUN_WEIGHTS" == "1" && "$RUN_DYNAMIC" == "1" ]]; then
  weight_dynamic_force_args=()
  if [[ "$WEIGHT_DYNAMIC_FORCE_RERUN" == "1" || "$FORCE_RERUN" == "1" ]]; then
    weight_dynamic_force_args+=(--force_rerun)
  fi

  for sweep in "${BIT_SWEEPS[@]}"; do
    bit="${sweep%%:*}"
    formats="${sweep#*:}"
    run_cmd "${RUNNER[@]}" "$WEIGHT_SCRIPT" \
      "${COMMON_ARGS[@]}" \
      --output_dir "${OUTPUT_ROOT}/weights/dynamic_chunk_${bit}bit" \
      --metrics "mse" \
      --weight_chunk_size "$CHUNK_SIZE" \
      --baseline_formats "$formats" \
      --run_eval \
      --skip_baselines \
      --skip_layer_wise \
      --per_chunk_format \
      --fold_input_norm \
      "${weight_dynamic_force_args[@]}" \
      "${maybe_force_recalc_args[@]}"
  done
fi
