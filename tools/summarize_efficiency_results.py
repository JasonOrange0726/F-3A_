#!/usr/bin/env python3
"""Summarize F3A efficiency JSON files into CSV/LaTeX-friendly tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

METHOD_LABELS = {
    "full": "Full",
    "fastv": "FastV",
    "divprune": "DivPrune",
    "cdprune": "CDPrune",
    "visionzip": "VisionZip",
    "foraging": "F3A",
}
METHOD_ORDER = ["full", "fastv", "divprune", "cdprune", "visionzip", "foraging"]
RATIO_ORDER = ["k100", "k60", "k40", "k20"]


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def pick_score(summary: dict[str, Any] | None) -> tuple[str, float | None]:
    if not isinstance(summary, dict):
        return "", None
    for key in ["accuracy", "f1", "score", "relaxed_accuracy", "vqa_accuracy"]:
        value = summary.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)
    return "", None


def result_score(results_root: Path, dataset: str, model: str, method: str, ratio: str) -> tuple[str, float | None]:
    tag = f"models--Qwen--{model}"
    if method == "full":
        path = results_root / "qwen3_ordered" / model / f"{dataset}_{tag}_k100_baseline.json"
        payload = load_json(path)
        return pick_score(payload.get("baseline_summary") if payload else None)
    if method == "foraging":
        path = results_root / "qwen3_f3a_final" / model / f"{dataset}_{tag}_{ratio}.json"
    else:
        path = results_root / "qwen3_baselines" / method / model / f"{dataset}_{tag}_{ratio}.json"
    payload = load_json(path)
    return pick_score(payload.get("routed_summary") if payload else None)


def ratio_sort_key(ratio: str) -> int:
    return RATIO_ORDER.index(ratio) if ratio in RATIO_ORDER else 99


def method_sort_key(method: str) -> int:
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else 99


def fmt(value: Any, digits: int = 2) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--results-root", default="outputs")
    parser.add_argument("--dataset", default="pope")
    parser.add_argument("--model", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-tex", default="")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    results_root = Path(args.results_root)
    rows: list[dict[str, Any]] = []

    for path in sorted(input_dir.glob("*.json")):
        payload = load_json(path)
        if not payload:
            continue
        cfg = payload.get("config", {})
        eff = payload.get("efficiency", {})
        method = str(payload.get("method_name") or cfg.get("method_name") or cfg.get("routing_mode") or "")
        if method == "":
            continue
        ratio_value = cfg.get("keep_ratio")
        ratio = "k100" if method == "full" or payload.get("profile_mode") == "full" else f"k{int(round(float(ratio_value) * 100))}"
        score_metric, score = result_score(results_root, args.dataset, args.model, method, ratio)
        rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "ratio": ratio,
                "keep_percent": 100 if ratio == "k100" else int(ratio[1:]),
                "score_metric": score_metric,
                "score": score,
                "num_profiled_samples": eff.get("num_profiled_samples"),
                "avg_full_visual_tokens": eff.get("avg_full_visual_tokens"),
                "avg_selected_visual_tokens": eff.get("avg_selected_visual_tokens"),
                "avg_keep_ratio_actual": eff.get("avg_keep_ratio_actual"),
                "avg_prefill_seq_len": eff.get("avg_prefill_seq_len"),
                "latency_ms_mean": eff.get("latency_ms_mean"),
                "latency_ms_p50": eff.get("latency_ms_p50"),
                "latency_ms_p90": eff.get("latency_ms_p90"),
                "prefill_kv_cache_mb_mean": eff.get("prefill_kv_cache_mb_mean"),
                "kv_cache_mb_mean": eff.get("kv_cache_mb_mean"),
                "peak_extra_memory_mb_mean": eff.get("peak_extra_memory_mb_mean"),
                "path": str(path),
            }
        )

    rows.sort(key=lambda r: (ratio_sort_key(str(r["ratio"])), method_sort_key(str(r["method"]))))
    full_latency = next((r.get("latency_ms_mean") for r in rows if r.get("method") == "full"), None)
    full_kv = next((r.get("kv_cache_mb_mean") for r in rows if r.get("method") == "full"), None)
    full_mem = next((r.get("peak_extra_memory_mb_mean") for r in rows if r.get("method") == "full"), None)
    for row in rows:
        latency = row.get("latency_ms_mean")
        kv = row.get("kv_cache_mb_mean")
        mem = row.get("peak_extra_memory_mb_mean")
        row["speedup_vs_full"] = float(full_latency) / float(latency) if full_latency and latency else ""
        row["kv_reduction_percent"] = (1.0 - float(kv) / float(full_kv)) * 100.0 if full_kv and kv else ""
        row["peak_mem_reduction_percent"] = (1.0 - float(mem) / float(full_mem)) * 100.0 if full_mem and mem else ""

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method_label",
        "ratio",
        "keep_percent",
        "score_metric",
        "score",
        "num_profiled_samples",
        "avg_full_visual_tokens",
        "avg_selected_visual_tokens",
        "avg_keep_ratio_actual",
        "avg_prefill_seq_len",
        "latency_ms_mean",
        "latency_ms_p50",
        "latency_ms_p90",
        "speedup_vs_full",
        "prefill_kv_cache_mb_mean",
        "kv_cache_mb_mean",
        "kv_reduction_percent",
        "peak_extra_memory_mb_mean",
        "peak_mem_reduction_percent",
        "path",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    print(f"saved_csv={output_csv}")

    if args.output_tex:
        tex_path = Path(args.output_tex)
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            r"\begin{tabular}{llrrrrrr}",
            r"\toprule",
            r"Method & Keep & Score & Tokens & Lat. (ms) & Speedup & KV (MB) & Mem. (MB) \\",
            r"\midrule",
        ]
        for row in rows:
            lines.append(
                f"{row['method_label']} & {row['keep_percent']}\\% & "
                f"{fmt(row.get('score'))} & {fmt(row.get('avg_selected_visual_tokens'), 1)} & "
                f"{fmt(row.get('latency_ms_mean'), 1)} & {fmt(row.get('speedup_vs_full'), 2)}$\\times$ & "
                f"{fmt(row.get('kv_cache_mb_mean'), 1)} & {fmt(row.get('peak_extra_memory_mb_mean'), 1)} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"saved_tex={tex_path}")


if __name__ == "__main__":
    main()
