import json
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Sequence

import torch

from .image_eval import (
    build_arg_parser,
    build_model,
    load_local_dataset,
    materialize_image_source,
    maybe_select_indices,
    parse_mmbench_message,
    parse_realworldqa_question,
)


def parse_args():
    parser = build_arg_parser(description="Profile visual token count, latency, and memory on image-benchmark image benchmarks.")
    parser.add_argument("--warmup-samples", type=int, default=5)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def make_sample_runner(model, row: dict[str, Any], args, routed: bool) -> Callable[[], dict[str, Any]]:
    image_source = materialize_image_source(row["image"])
    system_prompt = args.system_prompt or None
    final_keep = args.final_keep if args.final_keep > 0 else None

    if args.dataset == "pope":
        question = str(row["question"])
        prompt_text = model.format_yes_no_prompt(question)

        def run():
            return model.generate_answer(
                image_source=image_source,
                question=question,
                prompt_text=prompt_text,
                routed=routed,
                final_keep=final_keep,
                keep_ratio=args.keep_ratio,
                system_prompt=system_prompt,
                max_new_tokens=min(args.max_new_tokens, 4),
            )

        return run

    if args.dataset == "chartqa":
        question = str(row["question"])
        prompt_text = model.format_short_answer_prompt(question)

        def run():
            return model.generate_answer(
                image_source=image_source,
                question=question,
                prompt_text=prompt_text,
                routed=routed,
                final_keep=final_keep,
                keep_ratio=args.keep_ratio,
                system_prompt=system_prompt,
                max_new_tokens=args.max_new_tokens,
            )

        return run

    if args.dataset == "ai2d":
        instruction = str(row["question"])
        choices = [str(option) for option in row["options"]]
        prompt_text = model.format_multiple_choice_prompt(instruction=instruction, choices=choices)

        def run():
            return model.predict_choice(
                image_source=image_source,
                instruction=instruction,
                choices=choices,
                routed=routed,
                final_keep=final_keep,
                keep_ratio=args.keep_ratio,
                system_prompt=system_prompt,
            )

        return run

    if args.dataset == "hallusionbench":
        question = str(row["question"])
        prompt_text = model.format_yes_no_prompt(question)

        def run():
            return model.generate_answer(
                image_source=image_source,
                question=question,
                prompt_text=prompt_text,
                routed=routed,
                final_keep=final_keep,
                keep_ratio=args.keep_ratio,
                system_prompt=system_prompt,
                max_new_tokens=min(args.max_new_tokens, 4),
            )

        return run

    if args.dataset == "mme":
        question = str(row["question"])
        prompt_text = model.format_yes_no_prompt(question)

        def run():
            return model.generate_answer(
                image_source=image_source,
                question=question,
                prompt_text=prompt_text,
                routed=routed,
                final_keep=final_keep,
                keep_ratio=args.keep_ratio,
                system_prompt=system_prompt,
                max_new_tokens=min(args.max_new_tokens, 4),
            )

        return run

    if args.dataset == "scienceqa":
        choices = [str(choice) for choice in row["choices"]]
        prompt_parts = []
        hint = str(row.get("hint", "")).strip()
        if hint:
            prompt_parts.append(f"Context: {hint}")
        prompt_parts.append(str(row["question"]))
        instruction = "\n".join(prompt_parts)
        prompt_text = model.format_multiple_choice_prompt(instruction=instruction, choices=choices)

        def run():
            return model.predict_choice(
                image_source=image_source,
                instruction=instruction,
                choices=choices,
                routed=routed,
                final_keep=final_keep,
                keep_ratio=args.keep_ratio,
                system_prompt=system_prompt,
            )

        return run

    if args.dataset == "realworldqa":
        instruction, _, choices = parse_realworldqa_question(str(row["question"]))
        if choices:
            prompt_text = model.format_multiple_choice_prompt(instruction=instruction, choices=choices)

            def run():
                return model.predict_choice(
                    image_source=image_source,
                    instruction=instruction,
                    choices=choices,
                    routed=routed,
                    final_keep=final_keep,
                    keep_ratio=args.keep_ratio,
                    system_prompt=system_prompt,
                )

            return run

        prompt_text = model.format_short_answer_prompt(instruction)

        def run():
            return model.generate_answer(
                image_source=image_source,
                question=instruction,
                prompt_text=prompt_text,
                routed=routed,
                final_keep=final_keep,
                keep_ratio=args.keep_ratio,
                system_prompt=system_prompt,
                max_new_tokens=args.max_new_tokens,
            )

        return run

    if args.dataset == "textvqa":
        question = str(row["question"])
        prompt_text = model.format_short_answer_prompt(question)

        def run():
            return model.generate_answer(
                image_source=image_source,
                question=question,
                prompt_text=prompt_text,
                routed=routed,
                final_keep=final_keep,
                keep_ratio=args.keep_ratio,
                system_prompt=system_prompt,
                max_new_tokens=args.max_new_tokens,
            )

        return run

    if args.dataset == "mmbench":
        parsed = parse_mmbench_message(row["messages"])
        image_source = materialize_image_source(row["media"])
        instruction = parsed["instruction"]
        choices = parsed["choices"]
        prompt_text = model.format_multiple_choice_prompt(instruction=instruction, choices=choices)

        def run():
            return model.predict_choice(
                image_source=image_source,
                instruction=instruction,
                choices=choices,
                routed=routed,
                final_keep=final_keep,
                keep_ratio=args.keep_ratio,
                system_prompt=system_prompt,
            )

        return run

    if args.dataset == "vsr":
        question = f'Is the following statement true about the image? "{str(row["caption"]).strip()}"'
        image_source = row["image_path"]
        prompt_text = model.format_yes_no_prompt(question)

        def run():
            return model.generate_answer(
                image_source=image_source,
                question=question,
                prompt_text=prompt_text,
                routed=routed,
                final_keep=final_keep,
                keep_ratio=args.keep_ratio,
                system_prompt=system_prompt,
                max_new_tokens=min(args.max_new_tokens, 4),
            )

        return run

    if args.dataset == "visual7w":
        image_source = row["image_path"]
        instruction = str(row["question"])
        choices = list(row["choices"])
        prompt_text = model.format_multiple_choice_prompt(instruction=instruction, choices=choices)

        def run():
            return model.predict_choice(
                image_source=image_source,
                instruction=instruction,
                choices=choices,
                routed=routed,
                final_keep=final_keep,
                keep_ratio=args.keep_ratio,
                system_prompt=system_prompt,
            )

        return run

    raise ValueError(f"Unsupported dataset: {args.dataset}")


def profile_mode(model, dataset, args, routed: bool) -> dict[str, Any]:
    device = torch.device(args.device)
    indices = maybe_select_indices(len(dataset), args.max_samples, args.shuffle_seed)
    warmup = min(args.warmup_samples, len(indices))

    latencies_ms: list[float] = []
    peak_extra_mb: list[float] = []
    full_visual_tokens: list[int] = []
    selected_visual_tokens: list[int] = []
    prefill_seq_lens: list[int] = []
    final_seq_lens: list[int] = []
    generated_tokens: list[int] = []
    prefill_kv_cache_mb: list[float] = []
    final_kv_cache_mb: list[float] = []

    for step_idx, dataset_idx in enumerate(indices):
        row = dataset[dataset_idx]
        run = make_sample_runner(model, row, args, routed=routed)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            before_alloc = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        output = run()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if step_idx < warmup:
            continue

        full_tokens = output.get("full_visual_tokens")
        selected_tokens = output.get("selected_visual_tokens")
        prefill_seq_len = output.get("prefill_seq_len")
        final_seq_len = output.get("final_seq_len")
        generated = output.get("generated_tokens", 0)
        prefill_kv_mb = output.get("prefill_kv_cache_mb_estimate")
        final_kv_mb = output.get("kv_cache_mb_estimate")

        if full_tokens is None:
            raise RuntimeError("Model output is missing full_visual_tokens; update wrapper efficiency stats first.")
        if selected_tokens is None:
            selected_tokens = output.get("selected_count", full_tokens)
        if prefill_seq_len is None or final_seq_len is None:
            raise RuntimeError("Model output is missing sequence length efficiency stats.")

        full_visual_tokens.append(int(full_tokens))
        selected_visual_tokens.append(int(selected_tokens))
        prefill_seq_lens.append(int(prefill_seq_len))
        final_seq_lens.append(int(final_seq_len))
        generated_tokens.append(int(generated))
        if prefill_kv_mb is not None:
            prefill_kv_cache_mb.append(float(prefill_kv_mb))
        if final_kv_mb is not None:
            final_kv_cache_mb.append(float(final_kv_mb))
        latencies_ms.append(elapsed_ms)

        if device.type == "cuda":
            peak_alloc = torch.cuda.max_memory_allocated(device)
            peak_extra_mb.append(max(0.0, (peak_alloc - before_alloc) / (1024**2)))

    if not latencies_ms:
        raise RuntimeError("No profiled samples. Reduce warmup_samples or increase max_samples.")

    keep_ratios = [sel / full for sel, full in zip(selected_visual_tokens, full_visual_tokens)]
    summary = {
        "num_profiled_samples": len(latencies_ms),
        "warmup_samples": warmup,
        "avg_full_visual_tokens": mean(full_visual_tokens),
        "median_full_visual_tokens": median(full_visual_tokens),
        "avg_selected_visual_tokens": mean(selected_visual_tokens),
        "median_selected_visual_tokens": median(selected_visual_tokens),
        "avg_keep_ratio_actual": mean(keep_ratios),
        "avg_prefill_seq_len": mean(prefill_seq_lens),
        "median_prefill_seq_len": median(prefill_seq_lens),
        "avg_final_seq_len": mean(final_seq_lens),
        "median_final_seq_len": median(final_seq_lens),
        "avg_generated_tokens": mean(generated_tokens),
        "latency_ms_mean": mean(latencies_ms),
        "latency_ms_p50": percentile(latencies_ms, 0.5),
        "latency_ms_p90": percentile(latencies_ms, 0.9),
        "latency_ms_max": max(latencies_ms),
    }
    if prefill_kv_cache_mb:
        summary["prefill_kv_cache_mb_mean"] = mean(prefill_kv_cache_mb)
        summary["prefill_kv_cache_mb_p50"] = percentile(prefill_kv_cache_mb, 0.5)
        summary["prefill_kv_cache_mb_p90"] = percentile(prefill_kv_cache_mb, 0.9)
    if final_kv_cache_mb:
        summary["kv_cache_mb_mean"] = mean(final_kv_cache_mb)
        summary["kv_cache_mb_p50"] = percentile(final_kv_cache_mb, 0.5)
        summary["kv_cache_mb_p90"] = percentile(final_kv_cache_mb, 0.9)
    if peak_extra_mb:
        summary.update(
            {
                "peak_extra_memory_mb_mean": mean(peak_extra_mb),
                "peak_extra_memory_mb_p50": percentile(peak_extra_mb, 0.5),
                "peak_extra_memory_mb_p90": percentile(peak_extra_mb, 0.9),
                "peak_extra_memory_mb_max": max(peak_extra_mb),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    dataset = load_local_dataset(args)
    model = build_model(args)

    routed_summary = profile_mode(model=model, dataset=dataset, args=args, routed=True)
    print("routed_efficiency", json.dumps(routed_summary, ensure_ascii=False))

    baseline_summary = None
    if args.compare_baseline:
        baseline_summary = profile_mode(model=model, dataset=dataset, args=args, routed=False)
        print("baseline_efficiency", json.dumps(baseline_summary, ensure_ascii=False))

    if args.save_json:
        payload = {
            "config": vars(args),
            "dataset_size": len(dataset),
            "routed_efficiency": routed_summary,
            "baseline_efficiency": baseline_summary,
        }
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved_json={save_path}")


if __name__ == "__main__":
    main()
