#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from f3a.datasets import choice_labels
from f3a.internvl35.internvl35_wrapper import (
    DEFAULT_INTERNVL35_8B_INSTRUCT_MODEL_PATH,
    build_internvl35_model_from_args,
)
from f3a.image_eval import (
    build_arg_parser,
    compact_result,
    materialize_image_source,
    maybe_select_indices,
    parse_choice_letter,
    parse_mmbench_message,
    summarize_results,
)

ALIASES = {
    "mmbench_en_dev": ("MMBench-en", "dev-*.parquet"),
    "mmbench_en_test": ("MMBench-en", "test-*.parquet"),
    "mmbench_en_v11_test": ("MMBench-en-V11", "test-*.parquet"),
    "mmbench_cn_dev": ("MMBench-CN/data", "dev-*.parquet"),
    "mmbench_cn_test": ("MMBench-CN/data", "test-*.parquet"),
    "ccbench": ("MMBench-CN/chinese_culture", "test-*.parquet"),
    "mmbench_cn_culture_test": ("MMBench-CN/chinese_culture", "test-*.parquet"),
}
DEFAULT_DATASET_ROOT = os.environ.get("F3A_DATASET_ROOT", "data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MMBench-style datasets with InternVL3.5 + F3A.")
    parser.add_argument("--model-path", default=DEFAULT_INTERNVL35_8B_INSTRUCT_MODEL_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--device-map", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-alias", choices=sorted(ALIASES), required=True)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--keep-ratios", nargs="+", type=float, default=[0.6, 0.4, 0.2])
    parser.add_argument("--routing-mode", default="foraging", choices=["foraging", "legacy_topk", "divprune", "cdprune"])
    parser.add_argument("--text-conditioning-mode", default="universal_three_cue")
    parser.add_argument("--final-keep", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--max-num-tiles", type=int, default=12)
    parser.add_argument("--disable-thumbnail", action="store_true")
    parser.add_argument("--disable-flash-attn", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true", default=True)
    parser.add_argument("--run-baseline", dest="skip_baseline", action="store_false")
    parser.add_argument("--only-baseline", action="store_true", help="Run only full-token baseline and skip routed/pruned configs.")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def make_eval_args(main_args: argparse.Namespace, keep_ratio: float, save_json: Path):
    parser = build_arg_parser()
    argv = [
        "--dataset", "mmbench",
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


def load_mmbench_alias(alias: str, dataset_root: str):
    dirname, pattern = ALIASES[alias]
    root = Path(dataset_root).expanduser() / dirname
    data_dir = root if root.name in {"data", "chinese_culture"} else root / "data"
    data_files = sorted(str(path) for path in data_dir.glob(pattern))
    if not data_files:
        raise FileNotFoundError(f"No files for {alias}: {data_dir / pattern}")
    return load_dataset("parquet", data_files={"test": data_files}, split="test")


def has_completed_summary(path: Path, summary_key: str) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    summary = payload.get(summary_key)
    return isinstance(summary, dict) and bool(summary.get("has_ground_truth") is False or isinstance(summary.get("accuracy"), (int, float)))


def parse_mmbench_like_row(row: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    if "messages" in row:
        return parse_mmbench_message(row["messages"]), row["media"]

    labels: list[str] = []
    choices: list[str] = []
    for label in ["A", "B", "C", "D"]:
        value = row.get(label)
        text = "" if value is None else str(value).strip()
        if not text or text.lower() == "nan":
            continue
        labels.append(label)
        choices.append(text)

    answer = "" if row.get("answer") is None else str(row.get("answer", "")).strip().upper()
    answer_index = labels.index(answer) if answer in labels else None
    hint = "" if row.get("hint") is None else str(row.get("hint", "")).strip()
    question = str(row.get("question", "")).strip()
    instruction = "\n".join(part for part in [f"Hint: {hint}" if hint and hint.lower() != "nan" else "", question] if part)
    parsed = {
        "instruction": instruction,
        "question": question,
        "hint": hint if hint.lower() != "nan" else "",
        "choice_labels": labels,
        "choices": choices,
        "answer": answer,
        "answer_index": answer_index,
    }
    return parsed, row["image"]


def mmbench_prompt(parsed: dict[str, Any]) -> str:
    labels = parsed["choice_labels"] or choice_labels(len(parsed["choices"]))
    lines = ["You are given an image and a multiple-choice question."]
    if parsed.get("hint"):
        lines.append(f"Hint: {parsed['hint']}")
    lines.append(str(parsed.get("question", "")).strip())
    lines.append("Choices:")
    for label, choice in zip(labels, parsed["choices"]):
        lines.append(f"({label}) {choice}")
    lines.append("Answer with only the letter of the correct choice.")
    return "\n".join(lines)


def parse_prediction_index(text: str, labels: list[str]) -> int:
    letter = parse_choice_letter(text)
    if letter in labels:
        return labels.index(letter)
    cleaned = text.strip().upper()
    for idx, label in enumerate(labels):
        if cleaned.startswith(label) or cleaned.startswith(f"({label})"):
            return idx
    return -1


def run_mmbench_generate(model, dataset, args, routed: bool):
    results: list[dict[str, Any]] = []
    labeled_samples = 0
    indices = maybe_select_indices(len(dataset), args.max_samples, args.shuffle_seed)
    iterator = tqdm(indices, desc=f"internvl-mmbench-{'routed' if routed else 'full'}", ncols=100)
    for idx in iterator:
        row = dataset[idx]
        parsed, image_source = parse_mmbench_like_row(row)
        labels = parsed["choice_labels"] or choice_labels(len(parsed["choices"]))
        prompt = mmbench_prompt(parsed)
        output = model.generate_answer(
            image_source=materialize_image_source(image_source),
            question=parsed["question"],
            prompt_text=prompt,
            routed=routed,
            final_keep=args.final_keep if args.final_keep > 0 else None,
            keep_ratio=args.keep_ratio,
            system_prompt=args.system_prompt or None,
            max_new_tokens=args.max_new_tokens,
        )
        prediction = parse_prediction_index(output.get("text", ""), labels)
        extra = {
            "sample_id": str(row.get("id", idx)),
            "question": parsed["question"],
            "hint": parsed["hint"],
            "choice_labels": labels,
            "choices": parsed["choices"],
            "prediction": prediction,
            "prediction_label": labels[prediction] if prediction >= 0 else "",
            "prediction_text": output.get("text", ""),
        }
        if parsed["answer_index"] is not None:
            labeled_samples += 1
            extra["answer"] = parsed["answer"]
            extra["answer_index"] = parsed["answer_index"]
            extra["accuracy"] = int(prediction == parsed["answer_index"])
        results.append(compact_result(output, extra))
    summary = summarize_results(results, metric_name="accuracy")
    summary["labeled_samples"] = labeled_samples
    summary["has_ground_truth"] = labeled_samples > 0
    return results, summary


def write_payload(save_path: Path, args, dataset, routed_summary, routed_results, baseline_summary=None, baseline_results=None):
    payload = {
        "config": vars(args),
        "adapter": "internvl35",
        "eval_style": "internvl35_vlmeval_generation_parse_choice",
        "dataset_alias": args.dataset_alias,
        "dataset_size": len(dataset),
        "routed_summary": routed_summary,
        "baseline_summary": baseline_summary,
        "routed_results": routed_results,
        "baseline_results": baseline_results,
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    main_args = parse_args()
    out_dir = Path(main_args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = Path(main_args.model_path.rstrip("/")).name

    first_args = make_eval_args(main_args, main_args.keep_ratios[0], out_dir / "_warmup_mmbench_vlmeval.json")
    print(f"[load] model={main_args.model_path} device={main_args.device}", flush=True)
    model = build_internvl35_model_from_args(first_args)
    print(f"[load] done model={main_args.model_path}", flush=True)
    dataset = load_mmbench_alias(main_args.dataset_alias, main_args.dataset_root)
    print(f"[dataset] {main_args.dataset_alias} size={len(dataset)}", flush=True)

    rows: list[dict[str, Any]] = []
    baseline_summary = None
    baseline_score = None
    if main_args.skip_baseline:
        print(f"[skip] dataset={main_args.dataset_alias} config=baseline_k100", flush=True)
    else:
        baseline_path = out_dir / f"{main_args.dataset_alias}_{model_tag}_k100_baseline.json"
        if main_args.skip_existing and has_completed_summary(baseline_path, "baseline_summary"):
            payload = json.loads(baseline_path.read_text(encoding="utf-8"))
            baseline_summary = payload.get("baseline_summary")
            baseline_score = baseline_summary.get("accuracy") if isinstance(baseline_summary, dict) else None
            print(f"[skip-existing] dataset={main_args.dataset_alias} config=baseline_k100", flush=True)
        else:
            print(f"[run] dataset={main_args.dataset_alias} config=baseline_k100", flush=True)
            baseline_results, baseline_summary = run_mmbench_generate(model, dataset, first_args, routed=False)
            baseline_score = baseline_summary.get("accuracy")
            write_payload(baseline_path, first_args, dataset, None, None, baseline_summary, baseline_results)
        rows.append(
            {
                "dataset": main_args.dataset_alias,
                "model": model_tag,
                "config": "baseline_k100",
                "keep_ratio": 1.0,
                "metric": "accuracy" if isinstance(baseline_summary, dict) and baseline_summary.get("has_ground_truth") else "",
                "score": baseline_score,
                "baseline_score": baseline_score,
                "retention": 1.0 if baseline_score not in (None, 0) else None,
                "avg_keep_ratio_actual": None,
                "avg_selected_visual_tokens": None,
                "avg_full_visual_tokens": None,
                "json": str(baseline_path),
            }
        )

    if main_args.only_baseline:
        summary_path = out_dir / f"{main_args.dataset_alias}_{model_tag}_mmbench_vlmeval_summary.csv"
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
        return

    for keep_ratio in main_args.keep_ratios:
        suffix = f"k{int(round(keep_ratio * 100))}"
        save_path = out_dir / f"{main_args.dataset_alias}_{model_tag}_{suffix}_vlmeval.json"
        if main_args.skip_existing and has_completed_summary(save_path, "routed_summary"):
            print(f"[skip-existing] dataset={main_args.dataset_alias} config={suffix}", flush=True)
            continue
        eval_args = make_eval_args(main_args, keep_ratio, save_path)
        print(f"[run] dataset={main_args.dataset_alias} config={suffix}", flush=True)
        routed_results, routed_summary = run_mmbench_generate(model, dataset, eval_args, routed=True)
        write_payload(save_path, eval_args, dataset, routed_summary, routed_results, baseline_summary, None)
        score = routed_summary.get("accuracy")
        rows.append(
            {
                "dataset": main_args.dataset_alias,
                "model": model_tag,
                "config": suffix,
                "keep_ratio": keep_ratio,
                "metric": "accuracy" if routed_summary.get("has_ground_truth") else "",
                "score": score,
                "baseline_score": baseline_score,
                "retention": None if baseline_score in (None, 0) or score is None else score / baseline_score,
                "avg_keep_ratio_actual": routed_summary.get("avg_keep_ratio_actual"),
                "avg_selected_visual_tokens": routed_summary.get("avg_selected_visual_tokens"),
                "avg_full_visual_tokens": routed_summary.get("avg_full_visual_tokens"),
                "json": str(save_path),
            }
        )

    summary_path = out_dir / f"{main_args.dataset_alias}_{model_tag}_mmbench_vlmeval_summary.csv"
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
