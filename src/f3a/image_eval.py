import argparse
from collections import defaultdict
from io import BytesIO
import json
from pathlib import Path
import random
import re
import string
from statistics import mean
from typing import Any

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from .defaults import DEFAULT_MODEL_PATH, DEFAULT_ROUTER_CONFIG, DEFAULT_TEXT_CONDITIONING_MODE
from .paths import require_files, resolve_dataset_root
from .wrapper import F3AQwenVL


OPEN_DATASETS = {"hallusionbench", "mme", "textvqa", "vsr"}
MULTIPLE_CHOICE_DATASETS = {"ai2d", "scienceqa", "realworldqa", "mmbench", "visual7w"}


def build_arg_parser(description: str = "Evaluate F3A on image-benchmark image benchmarks with Qwen-VL.") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset",
        choices=["pope", "chartqa", "ai2d", "hallusionbench", "mme", "scienceqa", "realworldqa", "textvqa", "mmbench", "vsr", "visual7w"],
        required=True,
    )
    parser.add_argument("--dataset-root", type=str, default="")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--torch-dtype", type=str, default="auto")
    parser.add_argument("--device-map", type=str, default="")
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--keep-ratio", type=float, default=DEFAULT_ROUTER_CONFIG["keep_ratio"])
    parser.add_argument("--final-keep", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int, default=-1)
    parser.add_argument("--system-prompt", type=str, default="")
    parser.add_argument("--router-heads", type=int, default=DEFAULT_ROUTER_CONFIG["router_heads"])
    parser.add_argument("--token-nonzero", type=int, default=DEFAULT_ROUTER_CONFIG["token_nonzero"])
    parser.add_argument("--odor-nonzero", type=int, default=DEFAULT_ROUTER_CONFIG["odor_nonzero"])
    parser.add_argument("--head-topk", type=int, default=DEFAULT_ROUTER_CONFIG["head_topk"])
    parser.add_argument("--odor-topk", type=int, default=DEFAULT_ROUTER_CONFIG["odor_topk"])
    parser.add_argument("--local-window-size", type=int, default=DEFAULT_ROUTER_CONFIG["local_window_size"])
    parser.add_argument("--scaffold-keep", type=int, default=DEFAULT_ROUTER_CONFIG["scaffold_keep"])
    parser.add_argument("--smell-weight", type=float, default=DEFAULT_ROUTER_CONFIG["smell_weight"])
    parser.add_argument("--odor-gate-scale", type=float, default=DEFAULT_ROUTER_CONFIG["odor_gate_scale"])
    parser.add_argument("--odor-temperature", type=float, default=DEFAULT_ROUTER_CONFIG["odor_temperature"])
    parser.add_argument("--hypothesis-weight", type=float, default=DEFAULT_ROUTER_CONFIG["hypothesis_weight"])
    parser.add_argument("--contrast-weight", type=float, default=DEFAULT_ROUTER_CONFIG["contrast_weight"])
    parser.add_argument("--agreement-weight", type=float, default=DEFAULT_ROUTER_CONFIG["agreement_weight"])
    parser.add_argument("--repulsion-weight", type=float, default=DEFAULT_ROUTER_CONFIG["repulsion_weight"])
    parser.add_argument(
        "--routing-mode",
        choices=["foraging", "legacy_topk", "divprune", "cdprune", "fastv", "visionzip"],
        default="foraging",
    )
    parser.add_argument("--region-agreement-weight", type=float, default=DEFAULT_ROUTER_CONFIG["region_agreement_weight"])
    parser.add_argument("--lockon-weight", type=float, default=DEFAULT_ROUTER_CONFIG["lockon_weight"])
    parser.add_argument("--visit-weight", type=float, default=DEFAULT_ROUTER_CONFIG["visit_weight"])
    parser.add_argument("--uncertainty-weight", type=float, default=DEFAULT_ROUTER_CONFIG["uncertainty_weight"])
    parser.add_argument("--jump-ratio", type=float, default=DEFAULT_ROUTER_CONFIG["jump_ratio"])
    parser.add_argument("--ivc-keep-ratio", type=float, default=DEFAULT_ROUTER_CONFIG["ivc_keep_ratio"])
    parser.add_argument("--ivc-window-bonus", type=float, default=DEFAULT_ROUTER_CONFIG["ivc_window_bonus"])
    parser.add_argument("--coarse-pos-weight", type=float, default=DEFAULT_ROUTER_CONFIG["coarse_pos_weight"])
    parser.add_argument("--local-pos-weight", type=float, default=DEFAULT_ROUTER_CONFIG["local_pos_weight"])
    parser.add_argument("--anchor-pos-weight", type=float, default=DEFAULT_ROUTER_CONFIG["anchor_pos_weight"])
    parser.add_argument(
        "--text-conditioning-mode",
        choices=["pooled_prompt", "multi_cue", "universal_three_cue", "universal_vqa"],
        default=DEFAULT_TEXT_CONDITIONING_MODE,
    )
    parser.add_argument("--fastv-prune-layer", type=int, default=DEFAULT_ROUTER_CONFIG["fastv_prune_layer"])
    parser.add_argument("--router-seed", type=int, default=DEFAULT_ROUTER_CONFIG["router_seed"])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--pope-category", choices=["all", "random", "popular", "adversarial"], default="all")
    parser.add_argument("--vsr-split-family", choices=["random", "zeroshot"], default="zeroshot")
    parser.add_argument("--vsr-split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--visual7w-split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--save-json", type=str, default="")
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\.,;:!?\"'`]", "", text)
    return text


def extract_first_line(text: str) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line.strip().strip(".")


def parse_yes_no(text: str) -> str:
    lowered = normalize_text(text)
    yes_match = re.search(r"\byes\b", lowered)
    no_match = re.search(r"\bno\b", lowered)
    if yes_match and no_match:
        return "yes" if yes_match.start() < no_match.start() else "no"
    if yes_match:
        return "yes"
    if no_match:
        return "no"
    if lowered.startswith("y"):
        return "yes"
    if lowered.startswith("n"):
        return "no"
    return ""


def parse_number(text: str) -> float | None:
    candidate = extract_first_line(text).replace(",", "").strip()
    if candidate.endswith("%"):
        candidate = candidate[:-1].strip()
    match = re.search(r"[-+]?\d*\.?\d+", candidate)
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def chartqa_relaxed_correct(prediction: str, answer: str) -> bool:
    pred_num = parse_number(prediction)
    gold_num = parse_number(answer)
    if pred_num is not None and gold_num is not None:
        if gold_num == 0:
            return abs(pred_num - gold_num) <= 1e-6
        return abs(pred_num - gold_num) / abs(gold_num) <= 0.05
    return normalize_text(extract_first_line(prediction)) == normalize_text(answer)


def normalize_vqa_answer(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("\n", " ")
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def textvqa_score(prediction: str, answers: list[str]) -> float:
    norm_pred = normalize_vqa_answer(extract_first_line(prediction))
    if not norm_pred:
        return 0.0
    norm_answers = [normalize_vqa_answer(answer) for answer in answers if str(answer).strip()]
    if not norm_answers:
        return 0.0
    matches = sum(1 for answer in norm_answers if answer == norm_pred)
    return min(1.0, matches / 3.0)


def parse_choice_letter(text: str) -> str:
    candidate = extract_first_line(text).upper()
    match = re.search(r"\b([A-Z])\b", candidate)
    return match.group(1) if match else ""


def parse_realworldqa_question(question: str) -> tuple[str, list[str], list[str]]:
    lines = [line.strip() for line in question.splitlines() if line.strip()]
    instruction_lines: list[str] = []
    choice_labels: list[str] = []
    choices: list[str] = []
    for line in lines:
        match = re.match(r"^([A-E])(?:[.)]|\s)\s*(.+)$", line)
        if match:
            choice_labels.append(match.group(1))
            choices.append(match.group(2).strip())
            continue
        if "please answer directly with only the letter" in line.lower():
            continue
        if "please answer directly with a single word or number" in line.lower():
            continue
        instruction_lines.append(line)
    return "\n".join(instruction_lines), choice_labels, choices


def realworldqa_open_correct(prediction: str, answer: str) -> bool:
    answer = str(answer).strip()
    gold_yes_no = parse_yes_no(answer)
    if gold_yes_no:
        return parse_yes_no(prediction) == gold_yes_no

    gold_number = parse_number(answer)
    pred_number = parse_number(prediction)
    if gold_number is not None and pred_number is not None:
        return abs(pred_number - gold_number) <= 1e-6

    return normalize_text(extract_first_line(prediction)) == normalize_text(answer)


def resolve_realworldqa_answer_index(answer: str, choice_labels: list[str], choices: list[str]) -> int:
    candidate = str(answer).strip()
    if not candidate:
        raise ValueError("Empty RealWorldQA answer for multiple-choice sample")

    parsed_letter = parse_choice_letter(candidate)
    if parsed_letter:
        if parsed_letter in choice_labels:
            return choice_labels.index(parsed_letter)

    norm_candidate = normalize_text(candidate)
    for idx, choice in enumerate(choices):
        if normalize_text(choice) == norm_candidate:
            return idx

    raise ValueError(
        f"Failed to map RealWorldQA answer to choices: answer={answer!r} labels={choice_labels!r} choices={choices!r}"
    )


def parse_mmbench_message(raw_messages: Any) -> dict[str, Any]:
    if isinstance(raw_messages, str):
        messages = json.loads(raw_messages)
    else:
        messages = raw_messages

    if isinstance(messages, dict):
        message = messages
    elif isinstance(messages, list) and messages:
        message = messages[0]
    else:
        raise ValueError(f"Unsupported MMBench messages payload: {type(raw_messages)}")

    if not isinstance(message, dict):
        raise ValueError(f"Unsupported MMBench message entry: {type(message)}")

    question = str(message.get("question", "")).strip()
    hint = str(message.get("hint", "")).strip()
    options_map = message.get("options", {}) or {}
    choice_labels = [str(label).strip() for label in list(message.get("choices", []))]
    if not choice_labels and isinstance(options_map, dict):
        choice_labels = sorted(str(label).strip() for label in options_map.keys())
    choices = [str(options_map[label]).strip() for label in choice_labels if str(label).strip() in options_map]

    answer = str(message.get("answer", "")).strip()
    answer_index = None
    if answer:
        parsed_letter = parse_choice_letter(answer)
        if parsed_letter and parsed_letter in choice_labels:
            answer_index = choice_labels.index(parsed_letter)
        else:
            norm_answer = normalize_text(answer)
            for idx, choice in enumerate(choices):
                if normalize_text(choice) == norm_answer:
                    answer_index = idx
                    break

    prompt_parts: list[str] = []
    if hint:
        prompt_parts.append(f"Hint: {hint}")
    prompt_parts.append(question)
    instruction = "\n".join(part for part in prompt_parts if part)
    return {
        "instruction": instruction,
        "question": question,
        "hint": hint,
        "choice_labels": choice_labels,
        "choices": choices,
        "answer": answer,
        "answer_index": answer_index,
    }


def format_vsr_question(statement: str) -> str:
    statement = statement.strip()
    return f'Is the following statement true about the image? "{statement}"'


def build_visual7w_choices(answer: str, distractors: list[str]) -> tuple[list[str], int]:
    choices = [str(answer).strip()] + [str(choice).strip() for choice in distractors]
    if len(choices) != 4:
        raise ValueError(f"Visual7W expects 4 choices, got {len(choices)}")
    # Deterministic alphabetical order avoids answer-position bias without any randomness.
    ordered = sorted(enumerate(choices), key=lambda item: item[1].lower())
    sorted_choices = [choice for _, choice in ordered]
    answer_index = next(idx for idx, (orig_idx, _) in enumerate(ordered) if orig_idx == 0)
    return sorted_choices, answer_index


def compact_result(result: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    compact = dict(extra)
    compact["selected_count"] = result.get("selected_count")
    for key in [
        "full_visual_tokens",
        "selected_visual_tokens",
        "prefill_seq_len",
        "generated_tokens",
        "final_seq_len",
        "prefill_kv_cache_mb_estimate",
        "kv_cache_mb_estimate",
    ]:
        if key in result:
            compact[key] = result.get(key)
    if "text" in result:
        compact["text"] = result["text"]
    if "prediction" in result:
        compact["prediction"] = result["prediction"]
        compact["scores"] = result.get("scores")
    return compact


def materialize_image_source(value: Any) -> Any:
    if isinstance(value, dict):
        blob = value.get("bytes")
        if isinstance(blob, (bytes, bytearray)):
            return Image.open(BytesIO(blob)).convert("RGB")
        path = value.get("path")
        if isinstance(path, str) and path:
            return Image.open(path).convert("RGB")
    return value


def _data_files(root: Path, pattern: str) -> list[str]:
    return sorted(str(path) for path in root.glob(pattern))


def load_local_dataset(args: argparse.Namespace):
    if args.dataset == "pope":
        root = resolve_dataset_root("pope", args.dataset_root)
        data_files = require_files(_data_files(root / "data", "*.parquet"), f"POPE under {root}")
        dataset = load_dataset("parquet", data_files={"test": data_files}, split="test")
        if args.pope_category != "all":
            dataset = dataset.filter(lambda row: row["category"] == args.pope_category)
        return dataset

    if args.dataset == "chartqa":
        root = resolve_dataset_root("chartqa", args.dataset_root)
        data_files = require_files(_data_files(root / "data", "*.parquet"), f"ChartQA under {root}")
        return load_dataset("parquet", data_files={"test": data_files}, split="test")

    if args.dataset == "ai2d":
        root = resolve_dataset_root("ai2d", args.dataset_root)
        data_files = require_files(_data_files(root / "data", "*.parquet"), f"AI2D under {root}")
        return load_dataset("parquet", data_files={"test": data_files}, split="test")

    if args.dataset == "hallusionbench":
        root = resolve_dataset_root("hallusionbench", args.dataset_root)
        # HallusionBench ships both image and non-image parquet shards. We only
        # evaluate the image subset here; the non-image shard contains duplicated
        # text-only rows with `image=None`, which breaks visual evaluation.
        data_files = require_files(_data_files(root / "data", "image-*.parquet"), f"HallusionBench image shards under {root}")
        return load_dataset("parquet", data_files={"test": data_files}, split="test")

    if args.dataset == "mme":
        root = resolve_dataset_root("mme", args.dataset_root)
        data_files = require_files(_data_files(root / "data", "*.parquet"), f"MME under {root}")
        return load_dataset("parquet", data_files={"test": data_files}, split="test")

    if args.dataset == "scienceqa":
        root = resolve_dataset_root("scienceqa", args.dataset_root)
        data_files = require_files(_data_files(root / "data", "test-*.parquet"), f"ScienceQA test shards under {root}")
        dataset = load_dataset("parquet", data_files={"test": data_files}, split="test")
        dataset = dataset.filter(lambda row: row["image"] is not None and row["task"] == "closed choice")
        return dataset

    if args.dataset == "realworldqa":
        root = resolve_dataset_root("realworldqa", args.dataset_root)
        data_files = require_files(_data_files(root / "data", "*.parquet"), f"RealWorldQA under {root}")
        return load_dataset("parquet", data_files={"test": data_files}, split="test")

    if args.dataset == "mmbench":
        root = resolve_dataset_root("mmbench", args.dataset_root)
        dev_files = _data_files(root / "data", "dev-*.parquet")
        if dev_files:
            return load_dataset("parquet", data_files={"dev": dev_files}, split="dev")
        test_files = _data_files(root / "data", "test-*.parquet")
        if test_files:
            return load_dataset("parquet", data_files={"test": test_files}, split="test")
        fallback_root = resolve_dataset_root("mmbench_v11", "")
        data_files = require_files(_data_files(fallback_root / "data", "*.parquet"), f"MMBench under {root} or {fallback_root}")
        return load_dataset("parquet", data_files={"test": data_files}, split="test")

    if args.dataset == "vsr":
        root = resolve_dataset_root("vsr", args.dataset_root)
        split_path = root / "splits" / args.vsr_split_family / f"{args.vsr_split}.jsonl"
        image_root = root / "images" / "images"
        rows: list[dict[str, Any]] = []
        with split_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                row = json.loads(line)
                row["sample_id"] = f"{args.vsr_split_family}_{args.vsr_split}_{idx}"
                row["image_path"] = str(image_root / str(row["image"]))
                rows.append(row)
        return rows

    if args.dataset == "visual7w":
        root = resolve_dataset_root("visual7w", args.dataset_root)
        annotation_path = root / "telling" / "dataset_v7w_telling.json"
        image_root = root / "images" / "images"
        with annotation_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows: list[dict[str, Any]] = []
        for image_entry in payload["images"]:
            if str(image_entry.get("split")) != args.visual7w_split:
                continue
            image_path = str(image_root / str(image_entry["filename"]))
            for qa in image_entry.get("qa_pairs", []):
                choices, answer_index = build_visual7w_choices(
                    answer=str(qa["answer"]),
                    distractors=list(qa.get("multiple_choices", [])),
                )
                rows.append(
                    {
                        "sample_id": str(qa["qa_id"]),
                        "image_path": image_path,
                        "question": str(qa["question"]),
                        "choices": choices,
                        "answer_index": answer_index,
                        "question_type": str(qa.get("type", "")),
                        "image_id": str(image_entry.get("image_id", "")),
                        "filename": str(image_entry.get("filename", "")),
                        "split": str(image_entry.get("split", "")),
                    }
                )
        return rows

    root = resolve_dataset_root("textvqa", args.dataset_root)
    data_files = require_files(_data_files(root / "data", "validation-*.parquet"), f"TextVQA validation shards under {root}")
    return load_dataset("parquet", data_files={"validation": data_files}, split="validation")


def maybe_select_indices(length: int, max_samples: int, shuffle_seed: int) -> list[int]:
    indices = list(range(length))
    if max_samples <= 0 or max_samples >= length:
        return indices
    rng = random.Random(42 if shuffle_seed < 0 else shuffle_seed)
    rng.shuffle(indices)
    return indices[:max_samples]


def build_model(args: argparse.Namespace) -> F3AQwenVL:
    return F3AQwenVL.from_pretrained(
        model_path=args.model_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map or None,
        router_heads=args.router_heads,
        token_nonzero=args.token_nonzero,
        odor_nonzero=args.odor_nonzero,
        head_topk=args.head_topk,
        odor_topk=args.odor_topk,
        local_window_size=args.local_window_size,
        scaffold_keep=args.scaffold_keep,
        keep_ratio=args.keep_ratio,
        smell_weight=args.smell_weight,
        odor_gate_scale=args.odor_gate_scale,
        odor_temperature=args.odor_temperature,
        hypothesis_weight=args.hypothesis_weight,
        contrast_weight=args.contrast_weight,
        agreement_weight=args.agreement_weight,
        repulsion_weight=args.repulsion_weight,
        routing_mode=args.routing_mode,
        region_agreement_weight=args.region_agreement_weight,
        lockon_weight=args.lockon_weight,
        visit_weight=args.visit_weight,
        uncertainty_weight=args.uncertainty_weight,
        jump_ratio=args.jump_ratio,
        ivc_keep_ratio=args.ivc_keep_ratio,
        ivc_window_bonus=args.ivc_window_bonus,
        coarse_pos_weight=args.coarse_pos_weight,
        local_pos_weight=args.local_pos_weight,
        anchor_pos_weight=args.anchor_pos_weight,
        text_conditioning_mode=args.text_conditioning_mode,
        fastv_prune_layer=args.fastv_prune_layer,
        router_seed=args.router_seed,
    )


def summarize_results(results: list[dict[str, Any]], metric_name: str) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_samples": len(results)}
    metric_values = [item[metric_name] for item in results if item.get(metric_name) is not None]
    if metric_values:
        summary[metric_name] = 100.0 * mean(metric_values)
    selected = [item["selected_count"] for item in results if item.get("selected_count") is not None]
    if selected:
        summary["avg_selected_count"] = mean(selected)
    aggregate_fields = {
        "full_visual_tokens": "avg_full_visual_tokens",
        "selected_visual_tokens": "avg_selected_visual_tokens",
        "prefill_seq_len": "avg_prefill_seq_len",
        "generated_tokens": "avg_generated_tokens",
        "final_seq_len": "avg_final_seq_len",
        "prefill_kv_cache_mb_estimate": "avg_prefill_kv_cache_mb_estimate",
        "kv_cache_mb_estimate": "avg_kv_cache_mb_estimate",
    }
    for source_key, summary_key in aggregate_fields.items():
        values = [item[source_key] for item in results if item.get(source_key) is not None]
        if values:
            summary[summary_key] = mean(values)
    full_visual = [item["full_visual_tokens"] for item in results if item.get("full_visual_tokens") is not None]
    selected_visual = [
        item["selected_visual_tokens"] for item in results if item.get("selected_visual_tokens") is not None
    ]
    if full_visual and selected_visual and len(full_visual) == len(selected_visual):
        keep_ratios = [selected / full for selected, full in zip(selected_visual, full_visual) if full > 0]
        if keep_ratios:
            summary["avg_keep_ratio_actual"] = mean(keep_ratios)
    return summary


def eval_pope(model: F3AQwenVL, dataset, args: argparse.Namespace, routed: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    per_category: dict[str, list[int]] = defaultdict(list)
    indices = maybe_select_indices(len(dataset), args.max_samples, args.shuffle_seed)
    iterator = tqdm(indices, desc=f"pope-{'routed' if routed else 'full'}", ncols=100)
    for idx in iterator:
        row = dataset[idx]
        image_source = materialize_image_source(row["image"])
        output = model.generate_answer(
            image_source=image_source,
            question=row["question"],
            prompt_text=model.format_yes_no_prompt(row["question"]),
            routed=routed,
            final_keep=args.final_keep if args.final_keep > 0 else None,
            keep_ratio=args.keep_ratio,
            system_prompt=args.system_prompt or None,
            max_new_tokens=min(args.max_new_tokens, 4),
        )
        prediction = parse_yes_no(output["text"])
        answer = parse_yes_no(str(row["answer"]))
        correct = int(prediction == answer)
        per_category[str(row["category"])].append(correct)
        results.append(
            compact_result(
                output,
                {
                    "sample_id": str(row.get("question_id", row.get("id", idx))),
                    "question": row["question"],
                    "answer": answer,
                    "prediction_text": prediction,
                    "category": row["category"],
                    "accuracy": correct,
                },
            )
        )
    summary = summarize_results(results, metric_name="accuracy")
    summary["per_category_accuracy"] = {
        category: 100.0 * mean(values) for category, values in sorted(per_category.items())
    }
    return results, summary


def eval_chartqa(model: F3AQwenVL, dataset, args: argparse.Namespace, routed: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    by_type: dict[str, list[int]] = defaultdict(list)
    indices = maybe_select_indices(len(dataset), args.max_samples, args.shuffle_seed)
    iterator = tqdm(indices, desc=f"chartqa-{'routed' if routed else 'full'}", ncols=100)
    for idx in iterator:
        row = dataset[idx]
        image_source = materialize_image_source(row["image"])
        output = model.generate_answer(
            image_source=image_source,
            question=row["question"],
            prompt_text=model.format_short_answer_prompt(row["question"]),
            routed=routed,
            final_keep=args.final_keep if args.final_keep > 0 else None,
            keep_ratio=args.keep_ratio,
            system_prompt=args.system_prompt or None,
            max_new_tokens=args.max_new_tokens,
        )
        correct = int(chartqa_relaxed_correct(output["text"], str(row["answer"])))
        question_type = str(row.get("type", "unknown"))
        by_type[question_type].append(correct)
        results.append(
            compact_result(
                output,
                {
                    "sample_id": str(idx),
                    "question": row["question"],
                    "answer": str(row["answer"]),
                    "type": question_type,
                    "relaxed_accuracy": correct,
                },
            )
        )
    summary = summarize_results(results, metric_name="relaxed_accuracy")
    summary["per_type_relaxed_accuracy"] = {
        question_type: 100.0 * mean(values) for question_type, values in sorted(by_type.items())
    }
    return results, summary


def eval_ai2d(model: F3AQwenVL, dataset, args: argparse.Namespace, routed: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    indices = maybe_select_indices(len(dataset), args.max_samples, args.shuffle_seed)
    iterator = tqdm(indices, desc=f"ai2d-{'routed' if routed else 'full'}", ncols=100)
    for idx in iterator:
        row = dataset[idx]
        options = [str(option) for option in row["options"]]
        answer_index = int(row["answer"])
        image_source = materialize_image_source(row["image"])
        output = model.predict_choice(
            image_source=image_source,
            instruction=row["question"],
            choices=options,
            routed=routed,
            final_keep=args.final_keep if args.final_keep > 0 else None,
            keep_ratio=args.keep_ratio,
            system_prompt=args.system_prompt or None,
        )
        correct = int(output["prediction"] == answer_index)
        results.append(
            compact_result(
                output,
                {
                    "sample_id": str(idx),
                    "question": row["question"],
                    "answer_index": answer_index,
                    "accuracy": correct,
                },
            )
        )
    summary = summarize_results(results, metric_name="accuracy")
    return results, summary


def eval_hallusionbench(model: F3AQwenVL, dataset, args: argparse.Namespace, routed: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    per_category: dict[str, list[int]] = defaultdict(list)
    indices = maybe_select_indices(len(dataset), args.max_samples, args.shuffle_seed)
    iterator = tqdm(indices, desc=f"hallusion-{'routed' if routed else 'full'}", ncols=100)
    for idx in iterator:
        row = dataset[idx]
        image_source = materialize_image_source(row["image"])
        output = model.generate_answer(
            image_source=image_source,
            question=row["question"],
            prompt_text=model.format_yes_no_prompt(row["question"]),
            routed=routed,
            final_keep=args.final_keep if args.final_keep > 0 else None,
            keep_ratio=args.keep_ratio,
            system_prompt=args.system_prompt or None,
            max_new_tokens=min(args.max_new_tokens, 4),
        )
        prediction = parse_yes_no(output["text"])
        answer = "yes" if str(row["gt_answer"]) == "1" else "no"
        correct = int(prediction == answer)
        category = f"{row['category']}/{row['subcategory']}"
        per_category[category].append(correct)
        results.append(
            compact_result(
                output,
                {
                    "sample_id": str(row.get("question_id", idx)),
                    "question": row["question"],
                    "answer": answer,
                    "prediction_text": prediction,
                    "category": category,
                    "accuracy": correct,
                },
            )
        )
    summary = summarize_results(results, metric_name="accuracy")
    summary["per_category_accuracy"] = {
        category: 100.0 * mean(values) for category, values in sorted(per_category.items())
    }
    return results, summary


def eval_mme(model: F3AQwenVL, dataset, args: argparse.Namespace, routed: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    per_category: dict[str, list[int]] = defaultdict(list)
    per_category_per_image: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    indices = maybe_select_indices(len(dataset), args.max_samples, args.shuffle_seed)
    iterator = tqdm(indices, desc=f"mme-{'routed' if routed else 'full'}", ncols=100)
    for idx in iterator:
        row = dataset[idx]
        image_source = materialize_image_source(row["image"])
        output = model.generate_answer(
            image_source=image_source,
            question=row["question"],
            prompt_text=model.format_yes_no_prompt(row["question"]),
            routed=routed,
            final_keep=args.final_keep if args.final_keep > 0 else None,
            keep_ratio=args.keep_ratio,
            system_prompt=args.system_prompt or None,
            max_new_tokens=min(args.max_new_tokens, 4),
        )
        prediction = parse_yes_no(output["text"])
        answer = parse_yes_no(str(row["answer"]))
        correct = int(prediction == answer)
        category = str(row["category"])
        image_id = str(row.get("question_id", idx))
        per_category[category].append(correct)
        per_category_per_image[category][image_id].append(correct)
        results.append(
            compact_result(
                output,
                {
                    "sample_id": image_id,
                    "question": row["question"],
                    "answer": answer,
                    "prediction_text": prediction,
                    "category": category,
                    "accuracy": correct,
                },
            )
        )
    summary = summarize_results(results, metric_name="accuracy")
    summary["per_category_accuracy"] = {
        category: 100.0 * mean(values) for category, values in sorted(per_category.items())
    }
    per_category_score: dict[str, float] = {}
    for category, image_to_scores in sorted(per_category_per_image.items()):
        normal_acc = mean(score for scores in image_to_scores.values() for score in scores)
        pair_acc = mean(int(all(scores)) for scores in image_to_scores.values())
        per_category_score[category] = 100.0 * (normal_acc + pair_acc)
    summary["per_category_score"] = per_category_score
    summary["score"] = sum(per_category_score.values())
    return results, summary


def eval_scienceqa(model: F3AQwenVL, dataset, args: argparse.Namespace, routed: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    indices = maybe_select_indices(len(dataset), args.max_samples, args.shuffle_seed)
    iterator = tqdm(indices, desc=f"scienceqa-{'routed' if routed else 'full'}", ncols=100)
    for idx in iterator:
        row = dataset[idx]
        choices = [str(choice) for choice in row["choices"]]
        answer_index = int(row["answer"])
        prompt_parts = []
        hint = str(row.get("hint", "")).strip()
        if hint:
            prompt_parts.append(f"Context: {hint}")
        prompt_parts.append(str(row["question"]))
        image_source = materialize_image_source(row["image"])
        output = model.predict_choice(
            image_source=image_source,
            instruction="\n".join(prompt_parts),
            choices=choices,
            routed=routed,
            final_keep=args.final_keep if args.final_keep > 0 else None,
            keep_ratio=args.keep_ratio,
            system_prompt=args.system_prompt or None,
        )
        correct = int(output["prediction"] == answer_index)
        results.append(
            compact_result(
                output,
                {
                    "sample_id": str(idx),
                    "question": row["question"],
                    "answer_index": answer_index,
                    "subject": str(row.get("subject", "")),
                    "grade": str(row.get("grade", "")),
                    "accuracy": correct,
                },
            )
        )
    summary = summarize_results(results, metric_name="accuracy")
    return results, summary


def eval_realworldqa(model: F3AQwenVL, dataset, args: argparse.Namespace, routed: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    indices = maybe_select_indices(len(dataset), args.max_samples, args.shuffle_seed)
    iterator = tqdm(indices, desc=f"realworldqa-{'routed' if routed else 'full'}", ncols=100)
    for idx in iterator:
        row = dataset[idx]
        image_source = materialize_image_source(row["image"])
        instruction, choice_labels, choices = parse_realworldqa_question(str(row["question"]))
        answer = str(row["answer"]).strip()
        if choices:
            answer_index = resolve_realworldqa_answer_index(answer, choice_labels, choices)
            output = model.predict_choice(
                image_source=image_source,
                instruction=instruction,
                choices=choices,
                routed=routed,
                final_keep=args.final_keep if args.final_keep > 0 else None,
                keep_ratio=args.keep_ratio,
                system_prompt=args.system_prompt or None,
            )
            correct = int(output["prediction"] == answer_index)
            extra = {
                "sample_id": str(idx),
                "question": instruction,
                "question_type": "multiple_choice",
                "answer": answer,
                "answer_index": answer_index,
                "accuracy": correct,
            }
        else:
            output = model.generate_answer(
                image_source=image_source,
                question=instruction,
                prompt_text=model.format_short_answer_prompt(instruction),
                routed=routed,
                final_keep=args.final_keep if args.final_keep > 0 else None,
                keep_ratio=args.keep_ratio,
                system_prompt=args.system_prompt or None,
                max_new_tokens=args.max_new_tokens,
            )
            correct = int(realworldqa_open_correct(output["text"], answer))
            extra = {
                "sample_id": str(idx),
                "question": instruction,
                "question_type": "open",
                "answer": answer,
                "accuracy": correct,
            }
        results.append(
            compact_result(
                output,
                extra,
            )
        )
    summary = summarize_results(results, metric_name="accuracy")
    return results, summary


def eval_mmbench(model: F3AQwenVL, dataset, args: argparse.Namespace, routed: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    indices = maybe_select_indices(len(dataset), args.max_samples, args.shuffle_seed)
    iterator = tqdm(indices, desc=f"mmbench-{'routed' if routed else 'full'}", ncols=100)
    labeled_samples = 0
    for idx in iterator:
        row = dataset[idx]
        parsed = parse_mmbench_message(row["messages"])
        image_source = materialize_image_source(row["media"])
        output = model.predict_choice(
            image_source=image_source,
            instruction=parsed["instruction"],
            choices=parsed["choices"],
            routed=routed,
            final_keep=args.final_keep if args.final_keep > 0 else None,
            keep_ratio=args.keep_ratio,
            system_prompt=args.system_prompt or None,
        )
        extra = {
            "sample_id": str(row.get("id", idx)),
            "question": parsed["question"],
            "hint": parsed["hint"],
            "choice_labels": parsed["choice_labels"],
            "choices": parsed["choices"],
            "prediction_label": parsed["choice_labels"][output["prediction"]],
            "prediction_text": parsed["choices"][output["prediction"]],
        }
        if parsed["answer_index"] is not None:
            labeled_samples += 1
            extra["answer"] = parsed["answer"]
            extra["answer_index"] = parsed["answer_index"]
            extra["accuracy"] = int(output["prediction"] == parsed["answer_index"])
        results.append(compact_result(output, extra))

    summary = summarize_results(results, metric_name="accuracy")
    summary["labeled_samples"] = labeled_samples
    summary["has_ground_truth"] = labeled_samples > 0
    return results, summary


def eval_vsr(model: F3AQwenVL, dataset, args: argparse.Namespace, routed: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    per_relation: dict[str, list[int]] = defaultdict(list)
    indices = maybe_select_indices(len(dataset), args.max_samples, args.shuffle_seed)
    iterator = tqdm(indices, desc=f"vsr-{args.vsr_split_family}-{args.vsr_split}-{'routed' if routed else 'full'}", ncols=100)
    for idx in iterator:
        row = dataset[idx]
        question = format_vsr_question(str(row["caption"]))
        image_source = row["image_path"]
        output = model.generate_answer(
            image_source=image_source,
            question=question,
            prompt_text=model.format_yes_no_prompt(question),
            routed=routed,
            final_keep=args.final_keep if args.final_keep > 0 else None,
            keep_ratio=args.keep_ratio,
            system_prompt=args.system_prompt or None,
            max_new_tokens=min(args.max_new_tokens, 4),
        )
        prediction = parse_yes_no(output["text"])
        answer = "yes" if int(row["label"]) == 1 else "no"
        correct = int(prediction == answer)
        relation = str(row.get("relation", "unknown"))
        per_relation[relation].append(correct)
        results.append(
            compact_result(
                output,
                {
                    "sample_id": str(row.get("sample_id", idx)),
                    "image": str(row.get("image", "")),
                    "statement": str(row["caption"]),
                    "question": question,
                    "answer": answer,
                    "prediction_text": prediction,
                    "relation": relation,
                    "subject": str(row.get("subj", "")),
                    "object": str(row.get("obj", "")),
                    "accuracy": correct,
                },
            )
        )
    summary = summarize_results(results, metric_name="accuracy")
    summary["split_family"] = args.vsr_split_family
    summary["split"] = args.vsr_split
    summary["per_relation_accuracy"] = {
        relation: 100.0 * mean(values) for relation, values in sorted(per_relation.items())
    }
    return results, summary


def eval_visual7w(model: F3AQwenVL, dataset, args: argparse.Namespace, routed: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    per_type: dict[str, list[int]] = defaultdict(list)
    indices = maybe_select_indices(len(dataset), args.max_samples, args.shuffle_seed)
    iterator = tqdm(indices, desc=f"visual7w-{args.visual7w_split}-{'routed' if routed else 'full'}", ncols=100)
    for idx in iterator:
        row = dataset[idx]
        output = model.predict_choice(
            image_source=row["image_path"],
            instruction=row["question"],
            choices=row["choices"],
            routed=routed,
            final_keep=args.final_keep if args.final_keep > 0 else None,
            keep_ratio=args.keep_ratio,
            system_prompt=args.system_prompt or None,
        )
        correct = int(output["prediction"] == int(row["answer_index"]))
        question_type = str(row.get("question_type", "unknown"))
        per_type[question_type].append(correct)
        results.append(
            compact_result(
                output,
                {
                    "sample_id": str(row["sample_id"]),
                    "image_id": str(row.get("image_id", "")),
                    "filename": str(row.get("filename", "")),
                    "question": row["question"],
                    "choices": row["choices"],
                    "answer_index": int(row["answer_index"]),
                    "question_type": question_type,
                    "accuracy": correct,
                },
            )
        )
    summary = summarize_results(results, metric_name="accuracy")
    summary["split"] = args.visual7w_split
    summary["per_type_accuracy"] = {
        question_type: 100.0 * mean(values) for question_type, values in sorted(per_type.items())
    }
    return results, summary


def eval_textvqa(model: F3AQwenVL, dataset, args: argparse.Namespace, routed: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    indices = maybe_select_indices(len(dataset), args.max_samples, args.shuffle_seed)
    iterator = tqdm(indices, desc=f"textvqa-{'routed' if routed else 'full'}", ncols=100)
    for idx in iterator:
        row = dataset[idx]
        answers = [str(answer) for answer in row["answers"]]
        image_source = materialize_image_source(row["image"])
        output = model.generate_answer(
            image_source=image_source,
            question=row["question"],
            prompt_text=model.format_short_answer_prompt(str(row["question"])),
            routed=routed,
            final_keep=args.final_keep if args.final_keep > 0 else None,
            keep_ratio=args.keep_ratio,
            system_prompt=args.system_prompt or None,
            max_new_tokens=args.max_new_tokens,
        )
        score = textvqa_score(output["text"], answers)
        results.append(
            compact_result(
                output,
                {
                    "sample_id": str(row.get("question_id", idx)),
                    "question": row["question"],
                    "answers": answers,
                    "set_name": str(row.get("set_name", "")),
                    "vqa_accuracy": score,
                },
            )
        )
    summary = summarize_results(results, metric_name="vqa_accuracy")
    return results, summary


def run_eval(model: F3AQwenVL, dataset, args: argparse.Namespace, routed: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.dataset == "pope":
        return eval_pope(model=model, dataset=dataset, args=args, routed=routed)
    if args.dataset == "chartqa":
        return eval_chartqa(model=model, dataset=dataset, args=args, routed=routed)
    if args.dataset == "ai2d":
        return eval_ai2d(model=model, dataset=dataset, args=args, routed=routed)
    if args.dataset == "hallusionbench":
        return eval_hallusionbench(model=model, dataset=dataset, args=args, routed=routed)
    if args.dataset == "mme":
        return eval_mme(model=model, dataset=dataset, args=args, routed=routed)
    if args.dataset == "scienceqa":
        return eval_scienceqa(model=model, dataset=dataset, args=args, routed=routed)
    if args.dataset == "realworldqa":
        return eval_realworldqa(model=model, dataset=dataset, args=args, routed=routed)
    if args.dataset == "mmbench":
        return eval_mmbench(model=model, dataset=dataset, args=args, routed=routed)
    if args.dataset == "vsr":
        return eval_vsr(model=model, dataset=dataset, args=args, routed=routed)
    if args.dataset == "visual7w":
        return eval_visual7w(model=model, dataset=dataset, args=args, routed=routed)
    return eval_textvqa(model=model, dataset=dataset, args=args, routed=routed)


def main() -> None:
    args = parse_args()
    dataset = load_local_dataset(args)
    model = build_model(args)

    routed_results, routed_summary = run_eval(model=model, dataset=dataset, args=args, routed=True)
    print("routed_summary", json.dumps(routed_summary, ensure_ascii=False))

    baseline_results = None
    baseline_summary = None
    if args.compare_baseline:
        baseline_results, baseline_summary = run_eval(model=model, dataset=dataset, args=args, routed=False)
        print("baseline_summary", json.dumps(baseline_summary, ensure_ascii=False))

    if args.save_json:
        payload = {
            "config": vars(args),
            "dataset_size": len(dataset),
            "routed_summary": routed_summary,
            "baseline_summary": baseline_summary,
            "routed_results": routed_results,
            "baseline_results": baseline_results,
        }
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved_json={save_path}")


if __name__ == "__main__":
    main()
