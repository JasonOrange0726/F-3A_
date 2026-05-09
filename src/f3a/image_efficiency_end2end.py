import json
import time
from pathlib import Path
from typing import Any

import torch

from .image_eval import build_arg_parser, build_model, load_local_dataset, run_eval


def parse_args():
    parser = build_arg_parser(description="Run image-benchmark end-to-end efficiency evaluation on image benchmarks.")
    return parser.parse_args()


def benchmark_mode(model, dataset, args, routed: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    results, summary = run_eval(model=model, dataset=dataset, args=args, routed=routed)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        summary["peak_memory_gb"] = peak_memory_gb
    elapsed = time.perf_counter() - start
    summary["e2e_latency_seconds"] = elapsed
    summary["e2e_latency_hms"] = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    return results, summary


def main() -> None:
    args = parse_args()
    dataset = load_local_dataset(args)
    model = build_model(args)

    routed_results, routed_summary = benchmark_mode(model=model, dataset=dataset, args=args, routed=True)
    print("routed_efficiency", json.dumps(routed_summary, ensure_ascii=False))

    baseline_results = None
    baseline_summary = None
    if args.compare_baseline:
        baseline_results, baseline_summary = benchmark_mode(model=model, dataset=dataset, args=args, routed=False)
        print("baseline_efficiency", json.dumps(baseline_summary, ensure_ascii=False))
        if baseline_summary and routed_summary:
            baseline_t = baseline_summary.get("e2e_latency_seconds")
            routed_t = routed_summary.get("e2e_latency_seconds")
            if baseline_t and routed_t:
                routed_summary["speedup_vs_baseline"] = baseline_t / routed_t
                print("speedup_vs_baseline", f"{baseline_t / routed_t:.4f}")

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
