#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from f3a.qwen3_235b_fp8_adapter.fp8_wrapper import DEFAULT_FP8_MODEL_PATH, F3AQwenVL235BFP8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One synthetic-image smoke test for Qwen3-VL-235B FP8 + F3A.")
    parser.add_argument("--model-path", default=DEFAULT_FP8_MODEL_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--max-memory", default="")
    parser.add_argument("--offload-folder", default="")
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--routing-mode", default="foraging")
    parser.add_argument("--keep-ratio", type=float, default=0.2)
    parser.add_argument("--final-keep", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--prompt", default="Is there a red square in the image? Answer yes or no.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[smoke] loading {args.model_path}", flush=True)
    model = F3AQwenVL235BFP8.from_pretrained(
        model_path=args.model_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map or None,
        max_memory_spec=args.max_memory,
        offload_folder=args.offload_folder,
        attn_implementation=args.attn_implementation,
        routing_mode=args.routing_mode,
        keep_ratio=args.keep_ratio,
    )
    image = Image.new("RGB", (448, 448), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((120, 120, 320, 320), fill="red")
    draw.text((24, 24), "F3A FP8 smoke", fill="black")

    start = time.perf_counter()
    output = model.generate_answer(
        image_source=image,
        question=args.prompt,
        prompt_text=model.format_yes_no_prompt(args.prompt),
        routed=True,
        final_keep=args.final_keep if args.final_keep > 0 else None,
        keep_ratio=args.keep_ratio,
        max_new_tokens=args.max_new_tokens,
    )
    elapsed = time.perf_counter() - start
    print("[smoke] done", flush=True)
    print(f"text={output.get('text')!r}", flush=True)
    print(
        "tokens="
        f"selected={output.get('selected_visual_tokens')} "
        f"full={output.get('full_visual_tokens')} "
        f"prefill_seq_len={output.get('prefill_seq_len')} "
        f"elapsed={elapsed:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
