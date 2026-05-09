#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

METRIC_PRIORITY = [
    "accuracy",
    "relaxed_accuracy",
    "vqa_accuracy",
    "f1",
    "score",
]


def pick_metric(summary: dict | None) -> tuple[str, float | None]:
    if not summary:
        return "", None
    for key in METRIC_PRIORITY:
        value = summary.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)
    for key, value in summary.items():
        if key.startswith("avg_") or key == "num_samples":
            continue
        if isinstance(value, (int, float)):
            return key, float(value)
    return "", None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize F3A scaling JSON files into a CSV table.")
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.result_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        routed_summary = payload.get("routed_summary")
        baseline_summary = payload.get("baseline_summary")
        metric_name, routed_score = pick_metric(routed_summary)
        _, baseline_score = pick_metric(baseline_summary)
        retention = None
        if routed_score is not None and baseline_score not in (None, 0):
            retention = routed_score / baseline_score
        rows.append(
            {
                "file": path.name,
                "dataset": config.get("dataset", ""),
                "model_path": config.get("model_path", ""),
                "routing_mode": config.get("routing_mode", ""),
                "keep_ratio_config": config.get("keep_ratio", ""),
                "metric": metric_name,
                "routed_score": routed_score,
                "baseline_score": baseline_score,
                "retention": retention,
                "num_samples": (routed_summary or {}).get("num_samples", ""),
                "avg_keep_ratio_actual": (routed_summary or {}).get("avg_keep_ratio_actual", ""),
                "avg_selected_visual_tokens": (routed_summary or {}).get("avg_selected_visual_tokens", ""),
                "avg_full_visual_tokens": (routed_summary or {}).get("avg_full_visual_tokens", ""),
                "avg_prefill_seq_len": (routed_summary or {}).get("avg_prefill_seq_len", ""),
                "avg_kv_cache_mb_estimate": (routed_summary or {}).get("avg_kv_cache_mb_estimate", ""),
            }
        )

    fieldnames = [
        "file",
        "dataset",
        "model_path",
        "routing_mode",
        "keep_ratio_config",
        "metric",
        "routed_score",
        "baseline_score",
        "retention",
        "num_samples",
        "avg_keep_ratio_actual",
        "avg_selected_visual_tokens",
        "avg_full_visual_tokens",
        "avg_prefill_seq_len",
        "avg_kv_cache_mb_estimate",
    ]
    output = args.output or args.result_dir / "scaling_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
