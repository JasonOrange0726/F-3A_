#!/usr/bin/env python3
"""Run one efficiency profile config without changing the main eval scripts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f3a.image_eval import build_arg_parser, build_model, load_local_dataset
from f3a.image_efficiency import profile_mode


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser(description="Profile one F3A/baseline efficiency configuration.")
    parser.add_argument(
        "--profile-mode",
        choices=["routed", "full"],
        default="routed",
        help="Use routed=True for pruning methods; use full for the unpruned baseline.",
    )
    parser.add_argument("--method-name", default="", help="Name written to output config, e.g. f3a/full.")
    parser.add_argument("--warmup-samples", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_local_dataset(args)
    model = build_model(args)

    routed = args.profile_mode == "routed"
    summary = profile_mode(model=model, dataset=dataset, args=args, routed=routed)
    print("efficiency", json.dumps(summary, ensure_ascii=False))

    if args.save_json:
        payload: dict[str, Any] = {
            "config": vars(args),
            "dataset_size": len(dataset),
            "profile_mode": args.profile_mode,
            "method_name": args.method_name or args.routing_mode,
            "efficiency": summary,
        }
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved_json={save_path}")


if __name__ == "__main__":
    main()
