#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-OpenGVLab/InternVL3_5-8B}"
DATASETS="${DATASETS:-realworldqa pope}"
KEEP_RATIOS="${KEEP_RATIOS:-0.6 0.4 0.2}"
DEVICE="${DEVICE:-cuda:0}"
OUT_DIR="${F3A_OUTPUT_ROOT:-outputs}/internvl35_8b"
mkdir -p "$OUT_DIR"
python -m f3a.internvl35.run_internvl35_ordered \
  --model-path "$MODEL_PATH" \
  --device "$DEVICE" \
  --datasets $DATASETS \
  --keep-ratios $KEEP_RATIOS \
  --routing-mode foraging \
  --out-dir "$OUT_DIR"
