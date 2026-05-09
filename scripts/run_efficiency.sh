#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
DATASET="${DATASET:-realworldqa}"
KEEP_RATIO="${KEEP_RATIO:-0.4}"
DEVICE="${DEVICE:-cuda:0}"
OUT_DIR="${F3A_OUTPUT_ROOT:-outputs}/efficiency"
mkdir -p "$OUT_DIR"
python tools/run_efficiency_profile_one.py \
  --dataset "$DATASET" \
  --model-path "$MODEL_PATH" \
  --device "$DEVICE" \
  --routing-mode foraging \
  --keep-ratio "$KEEP_RATIO" \
  --method-name f3a \
  --save-json "$OUT_DIR/${DATASET}_f3a_${KEEP_RATIO}.json"
