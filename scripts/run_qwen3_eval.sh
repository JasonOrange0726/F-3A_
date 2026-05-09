#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
DATASETS="${DATASETS:-hallusionbench realworldqa pope}"
KEEP_RATIOS="${KEEP_RATIOS:-0.6 0.4 0.2}"
DEVICE="${DEVICE:-cuda:0}"
OUT_ROOT="${F3A_OUTPUT_ROOT:-outputs/qwen3_8b}"
mkdir -p "$OUT_ROOT"

for dataset in $DATASETS; do
  for ratio in $KEEP_RATIOS; do
    tag=$(printf '%.0f' "$(python - <<PY
print(float('$ratio') * 100)
PY
)")
    python -m f3a.image_eval \
      --dataset "$dataset" \
      --model-path "$MODEL_PATH" \
      --device "$DEVICE" \
      --routing-mode foraging \
      --keep-ratio "$ratio" \
      --save-json "$OUT_ROOT/${dataset}_f3a_${tag}.json"
  done
done
