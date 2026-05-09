#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from f3a.internvl35.internvl35_wrapper import (
    DEFAULT_INTERNVL35_38B_MODEL_PATH,
    DEFAULT_INTERNVL35_8B_MODEL_PATH,
    build_internvl35_model_from_args,
)
from f3a.image_eval import build_arg_parser, load_local_dataset, run_eval

METRIC_PRIORITY = ["accuracy", "relaxed_accuracy", "vqa_accuracy", "f1", "score"]


def pick_metric(summary: dict[str, Any] | None) -> tuple[str, float | None]:
    if not isinstance(summary, dict):
        return "", None
    for key in METRIC_PRIORITY:
        value = summary.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)
    for key, value in summary.items():
        if key == "num_samples" or key.startswith("avg_"):
            continue
        if isinstance(value, (int, float)):
            return key, float(value)
    return "", None


def parse_main_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run F3A on InternVL3.5 models without touching existing Qwen paths.")
    parser.add_argument("--model-path", default=DEFAULT_INTERNVL35_8B_MODEL_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--device-map", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=["hallusionbench", "realworldqa", "scienceqa", "ai2d", "pope"])
    parser.add_argument("--keep-ratios", nargs="+", type=float, default=[0.6, 0.4, 0.2])
    parser.add_argument("--final-keep", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--routing-mode", default="foraging", choices=["foraging", "legacy_topk", "divprune", "cdprune"])
    parser.add_argument("--text-conditioning-mode", default="universal_three_cue")
    parser.add_argument("--max-num-tiles", type=int, default=12)
    parser.add_argument("--disable-thumbnail", action="store_true")
    parser.add_argument("--disable-flash-attn", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true", default=True)
    parser.add_argument("--run-baseline", dest="skip_baseline", action="store_false")
    parser.add_argument("--only-baseline", action="store_true", help="Run only full-token baseline and skip routed/pruned configs.")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def make_eval_args(main_args: argparse.Namespace, dataset: str, keep_ratio: float, save_json: Path):
    parser = build_arg_parser()
    argv = [
        "--dataset", dataset,
        "--model-path", main_args.model_path,
        "--device", main_args.device,
        "--torch-dtype", main_args.torch_dtype,
        "--routing-mode", main_args.routing_mode,
        "--text-conditioning-mode", main_args.text_conditioning_mode,
        "--keep-ratio", str(keep_ratio),
        "--max-new-tokens", str(main_args.max_new_tokens),
        "--save-json", str(save_json),
    ]
    if main_args.device_map:
        argv += ["--device-map", main_args.device_map]
    if main_args.final_keep > 0:
        argv += ["--final-keep", str(main_args.final_keep)]
    if main_args.max_samples > 0:
        argv += ["--max-samples", str(main_args.max_samples), "--shuffle-seed", str(main_args.shuffle_seed)]
    args = parser.parse_args(argv)
    args.max_num_tiles = main_args.max_num_tiles
    args.disable_thumbnail = main_args.disable_thumbnail
    args.disable_flash_attn = main_args.disable_flash_attn
    return args


def has_completed_summary(path: Path, summary_key: str) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return pick_metric(payload.get(summary_key))[1] is not None


def write_payload(save_path: Path, args, dataset, routed_summary, routed_results, baseline_summary=None, baseline_results=None):
    payload = {
        "config": vars(args),
        "adapter": "internvl35",
        "dataset_size": len(dataset),
        "routed_summary": routed_summary,
        "baseline_summary": baseline_summary,
        "routed_results": routed_results,
        "baseline_results": baseline_results,
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    main_args = parse_main_args()
    out_dir = Path(main_args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = Path(main_args.model_path.rstrip("/")).name

    first_args = make_eval_args(main_args, main_args.datasets[0], main_args.keep_ratios[0], out_dir / "_warmup.json")
    print(f"[load] model={main_args.model_path} device={main_args.device}", flush=True)
    model = build_internvl35_model_from_args(first_args)
    print(f"[load] done model={main_args.model_path}", flush=True)

    rows: list[dict[str, Any]] = []
    for dataset_name in main_args.datasets:
        dataset_args = make_eval_args(main_args, dataset_name, main_args.keep_ratios[0], out_dir / "_dataset_load.json")
        dataset = load_local_dataset(dataset_args)
        print(f"[dataset] {dataset_name} size={len(dataset)}", flush=True)

        baseline_summary = None
        baseline_score = None
        if main_args.skip_baseline:
            print(f"[skip] dataset={dataset_name} config=baseline_k100", flush=True)
        else:
            baseline_path = out_dir / f"{dataset_name}_{model_tag}_k100_baseline.json"
            if main_args.skip_existing and has_completed_summary(baseline_path, "baseline_summary"):
                payload = json.loads(baseline_path.read_text(encoding="utf-8"))
                baseline_summary = payload.get("baseline_summary")
                metric_name, baseline_score = pick_metric(baseline_summary)
                print(f"[skip-existing] dataset={dataset_name} config=baseline_k100", flush=True)
            else:
                print(f"[run] dataset={dataset_name} config=baseline_k100", flush=True)
                baseline_results, baseline_summary = run_eval(model=model, dataset=dataset, args=dataset_args, routed=False)
                metric_name, baseline_score = pick_metric(baseline_summary)
                write_payload(baseline_path, dataset_args, dataset, None, None, baseline_summary, baseline_results)
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_tag,
                    "config": "baseline_k100",
                    "keep_ratio": 1.0,
                    "metric": metric_name,
                    "score": baseline_score,
                    "baseline_score": baseline_score,
                    "retention": 1.0 if baseline_score not in (None, 0) else None,
                    "json": str(baseline_path),
                }
            )

        if main_args.only_baseline:
            continue

        for keep_ratio in main_args.keep_ratios:
            suffix = f"k{int(round(keep_ratio * 100))}"
            save_path = out_dir / f"{dataset_name}_{model_tag}_{suffix}.json"
            if main_args.skip_existing and has_completed_summary(save_path, "routed_summary"):
                print(f"[skip-existing] dataset={dataset_name} config={suffix}", flush=True)
                continue
            eval_args = make_eval_args(main_args, dataset_name, keep_ratio, save_path)
            print(f"[run] dataset={dataset_name} config={suffix}", flush=True)
            routed_results, routed_summary = run_eval(model=model, dataset=dataset, args=eval_args, routed=True)
            write_payload(save_path, eval_args, dataset, routed_summary, routed_results, baseline_summary, None)
            metric, score = pick_metric(routed_summary)
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_tag,
                    "config": suffix,
                    "keep_ratio": keep_ratio,
                    "metric": metric,
                    "score": score,
                    "baseline_score": baseline_score,
                    "retention": None if baseline_score in (None, 0) or score is None else score / baseline_score,
                    "avg_keep_ratio_actual": routed_summary.get("avg_keep_ratio_actual"),
                    "avg_selected_visual_tokens": routed_summary.get("avg_selected_visual_tokens"),
                    "avg_full_visual_tokens": routed_summary.get("avg_full_visual_tokens"),
                    "json": str(save_path),
                }
            )

    summary_path = out_dir / f"{model_tag}_ordered_summary.csv"
    fieldnames = [
        "dataset",
        "model",
        "config",
        "keep_ratio",
        "metric",
        "score",
        "baseline_score",
        "retention",
        "avg_keep_ratio_actual",
        "avg_selected_visual_tokens",
        "avg_full_visual_tokens",
        "json",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
