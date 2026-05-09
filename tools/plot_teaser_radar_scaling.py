#!/usr/bin/env python3
"""Generate the front-page teaser figure: benchmark radar + scaling curves."""
from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-f3a")

import matplotlib.pyplot as plt
import numpy as np


MODELS = ["2B", "4B", "8B", "30B", "32B", "235B"]
MODEL_X = np.arange(len(MODELS), dtype=float)
RATIOS = ["60%", "40%", "20%"]
SCALING_RATIOS = ["60%", "40%"]
METHODS = ["DivPrune", "FastV", "CDPrune", "VisionZip", "F3A"]
RADAR_METHODS = ["FastV", "DivPrune", "CDPrune", "VisionZip", "F3A"]
METHOD_LABELS = {
    "DivPrune": "CDPruner",
    "FastV": "FastV",
    "CDPrune": "DivPrune",
    "VisionZip": "VisionZip",
    "F3A": r"$F^3A$",
}
BENCHMARKS = [
    ("Hall", "Hall"),
    ("MME", "MME"),
    ("AI2D", "AI2D"),
    ("RWQA", "RWQA"),
    ("SQA", "SQA"),
    ("POPE", "POPE"),
    ("MB$^{en}$", "MMB-en"),
    ("MB$^{zh}$", "MMB-zh"),
    ("CCB", "CCB"),
    ("VSR", "VSR"),
    ("V7W", "V7W"),
]
NON_MME_INDICES = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]

COLORS = {
    "Full": "#202020",
    "60%": "#1F77B4",
    "40%": "#2CA02C",
    "20%": "#9467BD",
    "FastV": "#72B7B2",
    "DivPrune": "#4C78A8",
    "CDPrune": "#F28E2B",
    "VisionZip": "#B279A2",
    "F3A": "#D62728",
}
MARKERS = {
    "Full": "o",
    "60%": "s",
    "40%": "^",
    "20%": "D",
    "DivPrune": "o",
    "FastV": "s",
    "CDPrune": "^",
    "VisionZip": "P",
    "F3A": "D",
}


def strip_tex(cell: str) -> str:
    cell = cell.strip().replace("\\%", "%")
    cell = cell.replace("\\textbf{F3A (Ours)}", "F3A")
    cell = re.sub(r"\\textbf\{([^{}]+)\}", r"\1", cell)
    cell = re.sub(r"[{}]", "", cell)
    cell = cell.replace("\\", "").strip()
    return cell


def to_float(cell: str) -> float | None:
    cell = strip_tex(cell)
    if not cell or cell == "--":
        return None
    return float(cell)


def parse_experiments(tex_path: Path) -> Tuple[Dict[Tuple[str, str, str], dict], Dict[str, dict]]:
    data: Dict[Tuple[str, str, str], dict] = {}
    full: Dict[str, dict] = {}

    text = tex_path.read_text()
    blocks = re.findall(r"\\begin\{table\*\}\[t\].*?\\end\{table\*\}", text, flags=re.S)
    for block in blocks:
        if "Qwen3-VL scaling results" not in block and "Qwen3-VL-235B-A22B" not in block:
            continue
        tabular = re.search(r"\\begin\{tabular\*\}.*?\n(.*?)\\end\{tabular\*\}", block, flags=re.S)
        if not tabular:
            continue

        current_model = None
        current_ratio = None
        table_text = re.sub(r"\\(toprule|midrule|bottomrule)\b", "", tabular.group(1))
        for raw_row in re.split(r"\\{2,}", table_text):
            row = " ".join(raw_row.split())
            if "&" not in row:
                continue
            parts = [p.strip() for p in row.split("&")]
            if len(parts) < 15:
                continue
            parts = parts[:15]

            first_raw = parts[0]
            first = strip_tex(first_raw)
            method = strip_tex(parts[1])
            if method == "Method":
                continue
            values = [to_float(parts[i]) for i in range(2, 13)]
            acc = to_float(parts[-2])
            rel = to_float(parts[-1])

            ratio_match = re.search(r"\\multirow\{5\}\{\*\}\{([0-9]+)\\%\}", first_raw)
            if ratio_match:
                current_ratio = f"{ratio_match.group(1)}%"
                ratio = current_ratio
            elif first.startswith("100"):
                ratio = "100%"
            else:
                ratio = current_ratio

            if ratio == "100%":
                model_match = re.search(r"Qwen3-VL-([0-9]+B)", method)
                if not model_match:
                    continue
                current_model = model_match.group(1)
                full[current_model] = {"bench": values, "acc": acc, "rel": rel}
                continue

            if current_model is None or ratio is None:
                continue
            data[(current_model, ratio, method)] = {"bench": values, "acc": acc, "rel": rel}

    missing = set(MODELS) - set(full)
    if missing:
        raise RuntimeError(f"Missing full-token rows for {sorted(missing)}")
    return data, full


def sem(values: list[float | None]) -> float:
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / math.sqrt(arr.size))


def rel_sem(values: list[float | None], full_values: list[float | None]) -> float:
    pairs = [(v, f) for v, f in zip(values, full_values) if v is not None and f not in (None, 0)]
    if len(pairs) <= 1:
        return 0.0
    arr = np.array([v / f * 100.0 for v, f in pairs], dtype=float)
    return float(arr.std(ddof=1) / math.sqrt(arr.size))


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.0,
            "legend.fontsize": 6.4,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.45,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_band(ax, y: list[float], err: list[float], color: str, alpha: float = 0.12) -> None:
    y_arr = np.array(y, dtype=float)
    err_arr = np.array(err, dtype=float)
    ax.fill_between(MODEL_X, y_arr - err_arr, y_arr + err_arr, color=color, alpha=alpha, linewidth=0, zorder=1)


def finish_scaling_axis(ax, ylabel: str, ylim: tuple[float, float]) -> None:
    ax.set_xticks(MODEL_X)
    ax.set_xticklabels(MODELS)
    # ax.set_xlabel("Qwen3-VL model size")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.35)
    ax.grid(True, axis="x", alpha=0.12)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def radar_values(data: dict, full: dict, model: str, ratio: str, method: str) -> list[float]:
    vals = data[(model, ratio, method)]["bench"]
    base = full[model]["bench"]
    return [100.0 * v / b for v, b in zip(vals, base) if v is not None and b not in (None, 0)]


def averaged_radar_values(data: dict, full: dict, ratio: str, method: str) -> list[float]:
    per_model = [radar_values(data, full, model, ratio, method) for model in MODELS]
    return [float(np.mean([values[i] for values in per_model])) for i in range(len(BENCHMARKS))]


def plot_radar(ax, data: dict, full: dict, model: str, ratio: str, average_models: bool = False) -> None:
    n = len(BENCHMARKS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    closed_angles = angles + angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels([short for _, short in BENCHMARKS])
    ax.tick_params(axis="x", pad=1, labelsize=7.1)
    ax.set_ylim(75, 103)
    ax.set_yticks([80, 90, 100])
    ax.set_yticklabels([])
    ax.set_rlabel_position(88)
    ax.grid(True, alpha=0.32)
    ax.spines["polar"].set_alpha(0.35)

    ax.plot(closed_angles, [100] * (n + 1), color="#202020", lw=1.0, ls="--", alpha=0.5, label="Full")
    for method in RADAR_METHODS:
        vals = averaged_radar_values(data, full, ratio, method) if average_models else radar_values(data, full, model, ratio, method)
        vals = vals + vals[:1]
        color = COLORS[method]
        lw = 1.8 if method == "F3A" else 0.9
        line_alpha = 1.0 if method == "F3A" else 0.55
        fill_alpha = 0.20 if method == "F3A" else 0.035
        ax.plot(closed_angles, vals, color=color, lw=lw, alpha=line_alpha, label=METHOD_LABELS[method])
        ax.fill(closed_angles, vals, color=color, alpha=fill_alpha)


def baseline_values(data: dict, model: str, ratio: str) -> list[float]:
    return [data[(model, ratio, method)]["acc"] for method in ["DivPrune", "FastV", "CDPrune", "VisionZip"]]


def method_average_acc(data: dict, model: str, method: str) -> float:
    return float(np.mean([data[(model, ratio, method)]["acc"] for ratio in RATIOS]))


def plot_method_scaling(ax, data: dict) -> None:
    method_values = {
        method: np.array([method_average_acc(data, model, method) for model in MODELS], dtype=float)
        for method in METHODS
    }
    for method in ["DivPrune", "FastV", "CDPrune", "VisionZip"]:
        ax.plot(
            MODEL_X,
            method_values[method],
            color=COLORS[method],
            lw=1.45,
            ls="--",
            marker=MARKERS[method],
            markersize=4.4,
            markerfacecolor="white",
            markeredgewidth=1.25,
            alpha=0.92,
            label=METHOD_LABELS[method],
            zorder=2,
        )

    ax.plot(
        MODEL_X,
        method_values["F3A"],
        color=COLORS["F3A"],
        lw=2.35,
        marker=MARKERS["F3A"],
        markersize=5.5,
        markerfacecolor=COLORS["F3A"],
        markeredgecolor=COLORS["F3A"],
        label=METHOD_LABELS["F3A"],
        zorder=5,
    )

    best_baseline = np.vstack([method_values[m] for m in ["DivPrune", "FastV", "CDPrune", "VisionZip"]]).max(axis=0)
    gap_best = method_values["F3A"] - best_baseline
    for idx, model in enumerate(MODELS):
        if model not in {"8B", "32B", "235B"}:
            continue
        ax.annotate(
            f"+{gap_best[idx]:.2f}",
            (MODEL_X[idx], method_values["F3A"][idx]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color=COLORS["F3A"],
            fontweight="bold",
        )

    ax.set_xticks(MODEL_X)
    ax.set_xticklabels(MODELS)
    ax.set_xlabel("Qwen3-VL model size")
    ax.set_ylabel("Average accuracy (%)")
    all_values = np.concatenate(list(method_values.values()))
    y_min = np.floor((all_values.min() - 1.0) / 2.0) * 2.0
    y_max = np.ceil((all_values.max() + 1.0) / 2.0) * 2.0
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(plt.MultipleLocator(2))
    ax.grid(True, axis="y", alpha=0.35)
    ax.grid(True, axis="x", alpha=0.12)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_advantage_pair(ax_env, ax_gain, data: dict, ratio: str) -> None:
    """Right-side teaser: show widening gap to training-free baselines."""
    ours = np.array([data[(m, ratio, "F3A")]["acc"] for m in MODELS], dtype=float)
    base_by_model = np.array([baseline_values(data, m, ratio) for m in MODELS], dtype=float)
    base_min = base_by_model.min(axis=1)
    base_max = base_by_model.max(axis=1)
    base_mean = base_by_model.mean(axis=1)

    ax_env.fill_between(MODEL_X, base_min, base_max, color="#9E9E9E", alpha=0.22, linewidth=0, label="Comparison Methods range")
    ax_env.plot(MODEL_X, base_mean, color="#777777", lw=1.8, ls="--", marker="o", markersize=3.8, label="Comparison Methods avg.")
    ax_env.plot(MODEL_X, ours, color=COLORS["F3A"], lw=2.0, marker="D", markersize=5.2, label="F3A")
    for x, y, gap in zip(MODEL_X, ours, ours - base_mean):
        if x in {MODEL_X[2], MODEL_X[-1]}:
            ax_env.annotate(f"+{gap:.1f}", (x, y), xytext=(0, 8), textcoords="offset points",
                            ha="center", va="bottom", fontsize=6.5, color=COLORS["F3A"])
    finish_scaling_axis(ax_env, "Average accuracy (%)", (70, 86.5))
    ax_env.legend(frameon=False, loc="lower right", handlelength=1.6)

    for ratio_i in SCALING_RATIOS:
        gains_avg = []
        for model in MODELS:
            base = baseline_values(data, model, ratio_i)
            ours_i = data[(model, ratio_i, "F3A")]["acc"]
            gains_avg.append(ours_i - float(np.mean(base)))
        ax_gain.plot(MODEL_X, gains_avg, color=COLORS[ratio_i], marker=MARKERS[ratio_i], lw=2.35, label=f"{ratio_i} tokens")
        ax_gain.annotate(
            f"+{gains_avg[-1]:.1f}",
            (MODEL_X[-1], gains_avg[-1]),
            xytext=(7, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=6.6,
            color=COLORS[ratio_i],
        )
    ax_gain.axhline(0, color="#202020", lw=0.9, ls="--", alpha=0.55)
    ax_gain.set_xticks(MODEL_X)
    ax_gain.set_xticklabels(MODELS)
    ax_gain.set_xlabel("Qwen3-VL model size")
    ax_gain.set_ylabel("Gain over baseline avg.")
    ax_gain.set_ylim(-0.2, 2.7)
    ax_gain.grid(True, axis="y", alpha=0.35)
    ax_gain.grid(True, axis="x", alpha=0.12)
    for spine in ["top", "right"]:
        ax_gain.spines[spine].set_visible(False)
    ax_gain.set_title("Advantage grows with scale", fontsize=8.8)
    ax_gain.legend(frameon=False, loc="upper left", ncol=1, handlelength=1.5)


def plot_teaser(data: dict, full: dict, out_dir: Path, model: str, ratio: str, name: str, average_models: bool = False) -> None:
    fig = plt.figure(figsize=(7.1, 2.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.38], wspace=0.48)
    ax_radar = fig.add_subplot(gs[0, 0], projection="polar")
    ax_scaling = fig.add_subplot(gs[0, 1])

    plot_radar(ax_radar, data, full, model, ratio, average_models=average_models)
    plot_method_scaling(ax_scaling, data)

    handles, labels = ax_radar.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=6,
        columnspacing=0.85,
        handlelength=1.35,
    )

    ax_radar.text(0.5, -0.26, "(a)", transform=ax_radar.transAxes, ha="center", va="top", fontsize=8.8, fontweight="normal")
    ax_scaling.text(0.5, -0.25, "(b)", transform=ax_scaling.transAxes, ha="center", va="top", fontsize=8.8, fontweight="normal")

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.03}
        if ext == "png":
            kwargs["dpi"] = 450
        fig.savefig(out_dir / f"{name}.{ext}", **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, default=Path("tables/main_results.tex"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/figures/teaser"))
    parser.add_argument("--model", default="8B", choices=MODELS)
    parser.add_argument("--ratio", default="40%", choices=RATIOS)
    parser.add_argument("--name", default="f3a_teaser_radar_scaling")
    parser.add_argument("--radar-average-models", action="store_true")
    args = parser.parse_args()

    setup_style()
    data, full = parse_experiments(args.tex)
    plot_teaser(data, full, args.out_dir, args.model, args.ratio, args.name, average_models=args.radar_average_models)
    print(f"Saved teaser figure to {args.out_dir}")


if __name__ == "__main__":
    main()
