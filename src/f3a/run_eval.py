import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
from statistics import mean

from tqdm import tqdm

from .defaults import DEFAULT_MODEL_PATH, DEFAULT_ROUTER_CONFIG, DEFAULT_TEXT_CONDITIONING_MODE
from .datasets import (
    ChoiceSample,
    load_hateful_memes_dataset,
    load_jsonl_choice_dataset,
    load_mmau_image_parquet,
)
from .wrapper import F3AQwenVL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate F3A-style routing on Qwen-VL multiple-choice prompts.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--torch-dtype", type=str, default="auto")
    parser.add_argument("--device-map", type=str, default="")
    parser.add_argument(
        "--dataset-type",
        choices=["single", "jsonl_choice", "mmau_parquet", "hateful_memes"],
        default="single",
    )
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--image-root", type=str, default="")
    parser.add_argument("--include-meme-text", action="store_true", default=True)
    parser.add_argument("--no-include-meme-text", dest="include_meme_text", action="store_false")
    parser.add_argument("--image-path", type=str, default="")
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--choices", nargs="*", default=[])
    parser.add_argument("--answer-index", type=int, default=-1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--balanced-max-samples", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int, default=-1)
    parser.add_argument("--system-prompt", type=str, default="")
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--final-keep", type=int, default=0)
    parser.add_argument("--keep-ratio", type=float, default=DEFAULT_ROUTER_CONFIG["keep_ratio"])
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
    parser.add_argument("--save-json", type=str, default="")
    return parser.parse_args()


def load_samples(args: argparse.Namespace) -> list[ChoiceSample]:
    if args.dataset_type == "single":
        if not args.image_path or not args.question or not args.choices:
            raise ValueError("single mode requires --image-path, --question and --choices")
        answer_index = None if args.answer_index < 0 else args.answer_index
        return [
            ChoiceSample(
                sample_id="single",
                instruction=args.question,
                choices=args.choices,
                answer_index=answer_index,
                image_path=args.image_path,
            )
        ]
    if args.dataset_type == "jsonl_choice":
        if not args.dataset_path:
            raise ValueError("jsonl_choice mode requires --dataset-path")
        return load_jsonl_choice_dataset(
            jsonl_path=args.dataset_path,
            image_root=args.image_root or None,
        )
    if args.dataset_type == "hateful_memes":
        if not args.dataset_path or not args.image_root:
            raise ValueError("hateful_memes mode requires --dataset-path and --image-root")
        return load_hateful_memes_dataset(
            annotation_path=args.dataset_path,
            image_root=args.image_root,
            include_meme_text=args.include_meme_text,
        )
    if not args.dataset_path:
        raise ValueError("mmau_parquet mode requires --dataset-path")
    return load_mmau_image_parquet(
        parquet_path=args.dataset_path,
    )


def count_annotations(args: argparse.Namespace) -> int | None:
    if args.dataset_type != "hateful_memes":
        return None
    total = 0
    with Path(args.dataset_path).open("r", encoding="utf-8") as handle:
        for _ in handle:
            total += 1
    return total


def answer_distribution(samples: list[ChoiceSample]) -> dict[str, int]:
    counter = Counter()
    for sample in samples:
        if sample.answer_index is None:
            counter["unlabeled"] += 1
        else:
            counter[str(sample.answer_index)] += 1
    return dict(sorted(counter.items(), key=lambda item: item[0]))


def select_samples(
    samples: list[ChoiceSample],
    max_samples: int,
    balanced_max_samples: int,
    shuffle_seed: int,
) -> tuple[list[ChoiceSample], int | None]:
    if not samples:
        return samples, None

    effective_seed = shuffle_seed
    if effective_seed < 0 and (max_samples > 0 or balanced_max_samples > 0):
        effective_seed = 42

    selected = list(samples)
    rng = random.Random(effective_seed) if effective_seed >= 0 else None

    if balanced_max_samples > 0:
        grouped: dict[int, list[ChoiceSample]] = defaultdict(list)
        unlabeled = []
        for sample in selected:
            if sample.answer_index is None:
                unlabeled.append(sample)
            else:
                grouped[sample.answer_index].append(sample)
        if len(grouped) < 2:
            raise ValueError("balanced_max_samples requires at least two labeled classes")
        class_ids = sorted(grouped)
        if rng is not None:
            for class_id in class_ids:
                rng.shuffle(grouped[class_id])
        target_per_class = max(1, balanced_max_samples // len(class_ids))
        actual_per_class = min(min(len(grouped[class_id]) for class_id in class_ids), target_per_class)
        selected = []
        for class_id in class_ids:
            selected.extend(grouped[class_id][:actual_per_class])
        if rng is not None:
            rng.shuffle(selected)
        return selected, effective_seed

    if rng is not None:
        rng.shuffle(selected)
    if max_samples > 0:
        selected = selected[:max_samples]
    return selected, effective_seed


def build_data_summary(
    args: argparse.Namespace,
    loaded_samples: list[ChoiceSample],
    selected_samples: list[ChoiceSample],
    annotation_count: int | None,
    effective_seed: int | None,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "dataset_type": args.dataset_type,
        "loaded_samples": len(loaded_samples),
        "selected_samples": len(selected_samples),
        "loaded_answer_distribution": answer_distribution(loaded_samples),
        "selected_answer_distribution": answer_distribution(selected_samples),
    }
    if annotation_count is not None:
        summary["annotation_count"] = annotation_count
        summary["skipped_missing_or_invalid"] = max(annotation_count - len(loaded_samples), 0)
    if effective_seed is not None:
        summary["effective_shuffle_seed"] = effective_seed
    if args.max_samples > 0:
        summary["max_samples"] = args.max_samples
    if args.balanced_max_samples > 0:
        summary["balanced_max_samples"] = args.balanced_max_samples
    return summary


def summarize_results(results: list[dict]) -> dict:
    summary = {
        "num_samples": len(results),
    }
    valid = [item for item in results if item.get("correct") is not None]
    if valid:
        summary["accuracy"] = 100.0 * mean(item["correct"] for item in valid)
        binary_valid = [item for item in valid if isinstance(item.get("scores"), list) and len(item["scores"]) == 2]
        if binary_valid:
            auc = binary_auroc(
                labels=[int(item["answer_index"]) for item in binary_valid],
                scores=[float(item["scores"][1] - item["scores"][0]) for item in binary_valid],
            )
            if auc is not None:
                summary["auroc"] = 100.0 * auc
    selected = [item["selected_count"] for item in results if item.get("selected_count") is not None]
    if selected:
        summary["avg_selected_count"] = mean(selected)
    return summary


def binary_auroc(labels: list[int], scores: list[float]) -> float | None:
    if len(labels) != len(scores) or not labels:
        return None
    positives = sum(1 for label in labels if label == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    order = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum_positive = 0.0
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and math.isclose(order[end][0], order[index][0], rel_tol=0.0, abs_tol=1e-12):
            end += 1
        avg_rank = 0.5 * (index + 1 + end)
        positive_count = sum(1 for _, label in order[index:end] if label == 1)
        rank_sum_positive += positive_count * avg_rank
        index = end

    u_stat = rank_sum_positive - positives * (positives + 1) / 2.0
    return u_stat / float(positives * negatives)


def main() -> None:
    args = parse_args()
    loaded_samples = load_samples(args)
    annotation_count = count_annotations(args)
    samples, effective_seed = select_samples(
        loaded_samples,
        max_samples=args.max_samples,
        balanced_max_samples=args.balanced_max_samples,
        shuffle_seed=args.shuffle_seed,
    )
    if not samples:
        raise RuntimeError("No image-based samples were found for evaluation")
    data_summary = build_data_summary(
        args=args,
        loaded_samples=loaded_samples,
        selected_samples=samples,
        annotation_count=annotation_count,
        effective_seed=effective_seed,
    )
    print("data_summary", json.dumps(data_summary, ensure_ascii=False))

    model = F3AQwenVL.from_pretrained(
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

    routed_results = []
    baseline_results = []

    for sample in tqdm(samples, desc="evaluating", ncols=100):
        routed_output = model.predict_sample(
            sample,
            routed=True,
            final_keep=args.final_keep if args.final_keep > 0 else None,
            keep_ratio=args.keep_ratio,
            system_prompt=args.system_prompt or None,
        )
        routed_results.append(routed_output)
        if args.compare_baseline:
            baseline_output = model.predict_sample(
                sample,
                routed=False,
                system_prompt=args.system_prompt or None,
            )
            baseline_results.append(baseline_output)

    routed_summary = summarize_results(routed_results)
    print("routed_summary", json.dumps(routed_summary, ensure_ascii=False))
    if args.compare_baseline:
        baseline_summary = summarize_results(baseline_results)
        print("baseline_summary", json.dumps(baseline_summary, ensure_ascii=False))
    else:
        baseline_summary = None

    if args.save_json:
        payload = {
            "config": vars(args),
            "data_summary": data_summary,
            "routed_summary": routed_summary,
            "baseline_summary": baseline_summary,
            "routed_results": routed_results,
            "baseline_results": baseline_results if args.compare_baseline else None,
        }
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved_json={save_path}")


if __name__ == "__main__":
    main()
