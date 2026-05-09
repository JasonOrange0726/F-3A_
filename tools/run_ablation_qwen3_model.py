#!/usr/bin/env python3
"""Run one F3A ablation variant across datasets/ratios with one model load."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f3a.image_eval import build_arg_parser, build_model, load_local_dataset, run_eval

METRIC_PRIORITY = ["accuracy", "relaxed_accuracy", "vqa_accuracy", "f1", "score"]


def pick_metric(summary: dict[str, Any]) -> tuple[str, float | None]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL F3A ablation variants.")
    parser.add_argument("--variant-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--device-map", default="")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--datasets", nargs="+", default=["hallusionbench", "realworldqa"])
    parser.add_argument("--keep-ratios", nargs="+", type=float, default=[0.4, 0.2])
    parser.add_argument("--final-keep", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")

    # Router/cue switches used by main ablations.
    parser.add_argument("--text-conditioning-mode", default="universal_three_cue")
    parser.add_argument("--routing-mode", default="foraging", choices=["foraging"])
    parser.add_argument("--hypothesis-weight", type=float, default=0.35)
    parser.add_argument("--contrast-weight", type=float, default=0.25)
    parser.add_argument("--agreement-weight", type=float, default=0.45)
    parser.add_argument("--repulsion-weight", type=float, default=0.35)
    parser.add_argument("--lockon-weight", type=float, default=0.35)
    parser.add_argument("--visit-weight", type=float, default=0.25)
    parser.add_argument("--uncertainty-weight", type=float, default=0.25)
    parser.add_argument("--jump-ratio", type=float, default=0.15)
    return parser.parse_args()


def has_completed_summary(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    summary = payload.get("routed_summary")
    return isinstance(summary, dict) and pick_metric(summary)[1] is not None


def make_eval_args(main_args: argparse.Namespace, dataset: str, keep_ratio: float, save_json: Path) -> argparse.Namespace:
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
        "--hypothesis-weight", str(main_args.hypothesis_weight),
        "--contrast-weight", str(main_args.contrast_weight),
        "--agreement-weight", str(main_args.agreement_weight),
        "--repulsion-weight", str(main_args.repulsion_weight),
        "--lockon-weight", str(main_args.lockon_weight),
        "--visit-weight", str(main_args.visit_weight),
        "--uncertainty-weight", str(main_args.uncertainty_weight),
        "--jump-ratio", str(main_args.jump_ratio),
        "--save-json", str(save_json),
    ]
    if main_args.final_keep > 0:
        argv += ["--final-keep", str(main_args.final_keep)]
    if main_args.device_map:
        argv += ["--device-map", main_args.device_map]
    if main_args.max_samples > 0:
        argv += ["--max-samples", str(main_args.max_samples), "--shuffle-seed", str(main_args.shuffle_seed)]
    return parser.parse_args(argv)


def model_names(model_path: str) -> tuple[str, str]:
    last = model_path.rstrip("/").split("/")[-1]
    clean = last.replace("models--Qwen--", "")
    return last, clean


def main() -> None:
    args = parse_args()
    model_tag, model_clean = model_names(args.model_path)
    out_dir = Path(args.out_root) / args.variant_name / model_clean
    out_dir.mkdir(parents=True, exist_ok=True)

    first_args = make_eval_args(args, args.datasets[0], args.keep_ratios[0], out_dir / "_warmup.json")
    print(f"[load] variant={args.variant_name} model={args.model_path} device={args.device}", flush=True)
    model = build_model(first_args)
    print(f"[load] done variant={args.variant_name}", flush=True)

    rows: list[dict[str, Any]] = []
    for dataset_name in args.datasets:
        dataset_args = make_eval_args(args, dataset_name, args.keep_ratios[0], out_dir / "_dataset_load.json")
        dataset = load_local_dataset(dataset_args)
        print(f"[dataset] {dataset_name} size={len(dataset)}", flush=True)
        for keep_ratio in args.keep_ratios:
            suffix = f"k{int(round(keep_ratio * 100))}"
            save_path = out_dir / f"{dataset_name}_{model_tag}_{suffix}.json"
            if args.skip_existing and has_completed_summary(save_path):
                print(f"[skip-existing] variant={args.variant_name} dataset={dataset_name} config={suffix}", flush=True)
                payload = json.loads(save_path.read_text(encoding="utf-8"))
                metric, score = pick_metric(payload["routed_summary"])
            else:
                eval_args = make_eval_args(args, dataset_name, keep_ratio, save_path)
                print(f"[run] variant={args.variant_name} dataset={dataset_name} config={suffix}", flush=True)
                routed_results, routed_summary = run_eval(model=model, dataset=dataset, args=eval_args, routed=True)
                payload = {
                    "config": vars(eval_args),
                    "variant_name": args.variant_name,
                    "dataset_size": len(dataset),
                    "routed_summary": routed_summary,
                    "baseline_summary": None,
                    "routed_results": routed_results,
                    "baseline_results": None,
                }
                save_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
                metric, score = pick_metric(routed_summary)
            rows.append(
                {
                    "variant": args.variant_name,
                    "dataset": dataset_name,
                    "model": model_clean,
                    "keep_ratio": keep_ratio,
                    "config": suffix,
                    "metric": metric,
                    "score": score,
                    "json": str(save_path),
                }
            )

    summary_csv = out_dir / f"{args.variant_name}_ablation_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "dataset", "model", "keep_ratio", "config", "metric", "score", "json"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summary] {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
