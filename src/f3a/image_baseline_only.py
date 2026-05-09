import json
from pathlib import Path

from .image_eval import build_arg_parser, build_model, load_local_dataset, run_eval


def parse_args():
    parser = build_arg_parser(description="Run baseline-only evaluation for image-benchmark image benchmarks with Qwen2.5-VL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_local_dataset(args)
    model = build_model(args)

    baseline_results, baseline_summary = run_eval(model=model, dataset=dataset, args=args, routed=False)
    print("baseline_summary", json.dumps(baseline_summary, ensure_ascii=False))

    if args.save_json:
        save_path = Path(args.save_json)
        payload = {
            "config": vars(args),
            "dataset_size": len(dataset),
            "routed_summary": None,
            "baseline_summary": baseline_summary,
            "routed_results": None,
            "baseline_results": baseline_results,
        }
        if save_path.exists():
            existing = json.loads(save_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                payload.update(existing)
                payload["config"] = existing.get("config", vars(args))
                payload["dataset_size"] = existing.get("dataset_size", len(dataset))
                payload["baseline_summary"] = baseline_summary
                payload["baseline_results"] = baseline_results
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved_json={save_path}")


if __name__ == "__main__":
    main()
