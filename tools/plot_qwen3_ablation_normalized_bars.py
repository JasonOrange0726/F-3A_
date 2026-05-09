#!/usr/bin/env python3
"""Plot normalized ablation bars similar to image-benchmark pruning-ratio figures."""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-f3a")

import matplotlib.pyplot as plt
import numpy as np

DATASETS = ["hallusionbench", "realworldqa", "ai2d"]
DS_LABEL = {"hallusionbench": "Hall", "realworldqa": "RWQA", "ai2d": "AI2D"}
PANELS = [
    ("k60", "(a) Retention Ratio 0.6"),
    ("k40", "(b) Retention Ratio 0.4"),
    ("k20", "(c) Retention Ratio 0.2"),
]
VARIANTS = ["full", "wo_odor_cue", "wo_multi_cue", "wo_local_lockon", "wo_rescue_jump"]
VARIANT_LABEL = {
    "full": r"$F^3A$",
    "wo_odor_cue": "w/o Odor Cue",
    "wo_multi_cue": "w/o Multi-Cue",
    "wo_local_lockon": "w/o Visual Lock-on",
    "wo_rescue_jump": "w/o Rescue Jump",
}
TEX_VARIANT_TO_ID = {
    "Full F3A": "full",
    "Full F^3A": "full",
    r"$F^3A$": "full",
    "w/o Odor Cue": "wo_odor_cue",
    "w/o Multi-Cue": "wo_multi_cue",
    "w/o Visual Lock-on": "wo_local_lockon",
    "w/o Rescue Jump": "wo_rescue_jump",
}
RATIO_TO_CONFIG = {"60%": "k60", "40%": "k40", "20%": "k20"}
COLORS = {
    "full": "#a8d8ea",
    "wo_odor_cue": "#ffb3b3",
    "wo_multi_cue": "#aaaaff",
    "wo_local_lockon": "#ffe18a",
    "wo_rescue_jump": "#c7e9b4",
}


def strip_tex(cell: str) -> str:
    cell = cell.strip()
    cell = cell.replace("\\%", "%")
    cell = re.sub(r"\\textbf\{([^{}]+)\}", r"\1", cell)
    return cell.strip()


def load_scores(path: Path) -> Dict[Tuple[str, str, str], float]:
    scores = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            scores[(row["variant"], row["dataset"], row["config"])] = float(row["score"])
    return scores


def load_scores_from_tex(path: Path) -> Dict[Tuple[str, str, str], float]:
    """Read the main ablation table from paper/experiments.tex."""
    scores: Dict[Tuple[str, str, str], float] = {}
    in_table = False
    current_config = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if "\\label{tab:main-ablation}" in line:
            in_table = False
            continue
        if "\\caption{Main ablation study" in line:
            in_table = True
            continue
        if not in_table or "&" not in line or not line.endswith("\\"):
            continue

        parts = [p.strip() for p in line[:-2].split("&")]
        if len(parts) < 6:
            continue
        ratio_cell = strip_tex(parts[0])
        variant_cell = strip_tex(parts[1])
        ratio_match = re.search(r"\{([0-9]+%)\}", ratio_cell)
        if ratio_match:
            current_config = RATIO_TO_CONFIG.get(ratio_match.group(1))
        if current_config is None:
            continue

        variant = TEX_VARIANT_TO_ID.get(variant_cell)
        if variant is None:
            continue
        for dataset, idx in zip(DATASETS, [2, 3, 4]):
            value = strip_tex(parts[idx])
            if value == "--":
                continue
            scores[(variant, dataset, current_config)] = float(value)
    return scores


def has_panel_scores(scores: Dict[Tuple[str, str, str], float], config: str) -> bool:
    return all((variant, ds, config) in scores for variant in VARIANTS for ds in DATASETS)


def save_fig(fig, out_dir: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 450
        fig.savefig(out_dir / f"{name}.{ext}", **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("outputs/ablation/main_ablation_summary.csv"))
    ap.add_argument("--tex", type=Path, default=Path("tables/experiments.tex"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/figures/qwen3_ablation"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scores = load_scores(args.csv)
    if args.tex.exists():
        scores.update(load_scores_from_tex(args.tex))

    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, len(PANELS), figsize=(10.6, 2.9), sharey=True)
    x = np.arange(len(DATASETS))
    width = 0.14
    offsets = (np.arange(len(VARIANTS)) - (len(VARIANTS) - 1) / 2) * width

    for ax, (config, caption) in zip(axes, PANELS):
        if has_panel_scores(scores, config):
            for off, variant in zip(offsets, VARIANTS):
                vals = []
                for ds in DATASETS:
                    full = scores[("full", ds, config)]
                    vals.append(scores[(variant, ds, config)] / full)
                ax.bar(x + off, vals, width=width, color=COLORS[variant], label=VARIANT_LABEL[variant], edgecolor="none")
        else:
            ax.text(
                0.5,
                0.52,
                "Pending\n0.6 retention",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color="#555555",
            )
        ax.axhline(1.0, color="#666666", lw=0.8, ls="--", alpha=0.65)
        ax.set_xticks(x)
        ax.set_xticklabels([DS_LABEL[d] for d in DATASETS])
        ax.set_ylim(0.80, 1.02)
        ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
        ax.set_ylabel("Normalized Score")
        ax.grid(True, axis="y", alpha=0.35, linestyle="--", linewidth=0.6)
        if has_panel_scores(scores, config):
            ax.legend(frameon=True, loc="lower right")
        # Put subfigure caption below the axes, matching the reference style.
        ax.text(0.5, -0.24, caption, transform=ax.transAxes, ha="center", va="top", fontsize=13)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    fig.tight_layout(w_pad=2.0, rect=(0, 0.08, 1, 1))
    save_fig(fig, args.out_dir, "main_ablation_normalized_bars")
    print(f"Saved normalized ablation bars to {args.out_dir}")


if __name__ == "__main__":
    main()
