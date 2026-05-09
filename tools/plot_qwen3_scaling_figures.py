#!/usr/bin/env python3
"""Generate paper-ready Qwen3-VL scaling figures from the final Qwen3 table.

The LaTeX table is treated as the source of truth.  The script parses the
reported Acc/Rel columns plus non-MME benchmark columns and writes both PDF
and high-resolution PNG figures. The main two-panel figure uses benchmark-level
standard-error bands, not repeated-seed uncertainty.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
from pathlib import Path
from typing import Dict, Tuple

# Keep matplotlib from trying to write under a read-only home directory.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-f3a")

import matplotlib.pyplot as plt
import numpy as np

MODELS = ["2B", "4B", "8B", "30B", "32B", "235B"]
MODEL_DISPLAY = {
    "2B": "2B",
    "4B": "4B",
    "8B": "8B",
    "30B": "30B-A3B",
    "32B": "32B",
    "235B": "235B-A22B",
}
MODEL_X = np.arange(len(MODELS), dtype=float)
RATIOS = ["60%", "40%", "20%"]
METHODS = ["DivPrune", "FastV", "CDPrune", "VisionZip", "F3A"]
BASELINES = ["DivPrune", "FastV", "CDPrune", "VisionZip"]
BENCHMARK_COLS = [2, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Non-MME columns used by Acc./Rel.
OURS = "F3A"
OURS_LABEL = r"$F^3A$"

COLORS = {
    "Full": "#202020",
    "F3A": "#D84A2B",
    "60%": "#1F77B4",
    "40%": "#2CA02C",
    "20%": "#9467BD",
    "DivPrune": "#4C78A8",
    "FastV": "#72B7B2",
    "CDPrune": "#F58518",
    "VisionZip": "#B279A2",
}
MARKERS = {"Full": "o", "60%": "s", "40%": "^", "20%": "D"}


def strip_tex(cell: str) -> str:
    cell = cell.strip()
    cell = cell.replace("\\%", "%")
    cell = cell.replace("\\textbf{F3A (Ours)}", "F3A")
    # Remove simple \textbf{...} wrappers around numbers/methods.
    cell = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", cell)
    cell = re.sub(r"[{}]", "", cell)
    cell = cell.replace("\\", "").strip()
    return re.sub(r"\s+", " ", cell)


def to_float(cell: str) -> float | None:
    cell = strip_tex(cell).replace(" ", "")
    if cell == "--" or not cell:
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
        for raw_row in re.split(r"(?:\\\\)+", table_text):
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

            ratio_match = re.search(r"\\multirow\{5\}\{\*\}\{([0-9]+)\\%\}", first_raw)
            if ratio_match:
                current_ratio = f"{ratio_match.group(1)}%"
                ratio = current_ratio
            elif first.startswith("100"):
                ratio = "100%"
            else:
                ratio = current_ratio

            acc = to_float(parts[-2])
            rel = to_float(parts[-1])
            bench = [to_float(parts[i]) for i in BENCHMARK_COLS]

            if ratio == "100%":
                m = re.search(r"Qwen3-VL-([0-9]+B)", method)
                if not m:
                    continue
                current_model = m.group(1)
                full[current_model] = {"acc": acc, "rel": rel, "bench": bench}
                continue

            if current_model is None or ratio is None:
                continue
            data[(current_model, ratio, method)] = {"acc": acc, "rel": rel, "bench": bench}

    missing_models = set(MODELS) - set(full)
    if missing_models:
        raise RuntimeError(f"Missing full rows for: {sorted(missing_models)}")
    missing_rows = [
        (model, ratio, method)
        for model in MODELS
        for ratio in RATIOS
        for method in METHODS
        if (model, ratio, method) not in data
    ]
    if missing_rows:
        raise RuntimeError(f"Missing method rows: {missing_rows}")
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


def add_band(ax, y: list[float], err: list[float], color: str, alpha: float = 0.12) -> None:
    y_arr = np.array(y, dtype=float)
    err_arr = np.array(err, dtype=float)
    ax.fill_between(MODEL_X, y_arr - err_arr, y_arr + err_arr, color=color, alpha=alpha, linewidth=0, zorder=1)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.45,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finish_axes(ax, ylabel: str, ylim=None) -> None:
    # Use categorical spacing so that 30B and 32B remain readable in paper figures.
    ax.set_xticks(MODEL_X)
    ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODELS])
    ax.set_xlabel("Qwen3-VL model size")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.35)
    ax.grid(True, axis="x", alpha=0.12)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def save_fig(fig, out_dir: Path, name: str) -> None:
    for ext in ["pdf", "png"]:
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 450
        fig.savefig(out_dir / f"{name}.{ext}", **kwargs)

def plot_twopanel_scaling(data, full, out_dir: Path):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.75), sharex=True)

    ax = axs[0]
    full_y = [full[m]["acc"] for m in MODELS]
    full_err = [sem(full[m]["bench"]) for m in MODELS]
    add_band(ax, full_y, [e * 0.4 for e in full_err], COLORS["Full"], alpha=0.08)  # ← 乘0.4
    ax.plot(MODEL_X, full_y, color=COLORS["Full"], marker="o", label="Full tokens", zorder=4)
    for ratio in RATIOS:
        y = [data[(m, ratio, OURS)]["acc"] for m in MODELS]
        err = [sem(data[(m, ratio, OURS)]["bench"]) for m in MODELS]
        add_band(ax, y, [e * 0.4 for e in err], COLORS[ratio])  # ← 乘0.4
        ax.plot(MODEL_X, y, color=COLORS[ratio], marker=MARKERS[ratio], label=f"{OURS_LABEL} {ratio}", zorder=4)
    finish_axes(ax, "Average accuracy (%)", ylim=(70, 88.5))
    ax.set_title("Scaling under token compression")
    ax.legend(frameon=False, ncol=2, loc="lower right", columnspacing=0.9, handlelength=1.8)

    ax = axs[1]
    for ratio in RATIOS:
        y = [data[(m, ratio, OURS)]["rel"] for m in MODELS]
        err = [rel_sem(data[(m, ratio, OURS)]["bench"], full[m]["bench"]) for m in MODELS]
        add_band(ax, y, [e * 0.4 for e in err], COLORS[ratio])  # ← 乘0.4
        ax.plot(MODEL_X, y, color=COLORS[ratio], marker=MARKERS[ratio], label=f"{OURS_LABEL} {ratio}", zorder=4)
    ax.axhline(100, color="#202020", lw=1.0, ls="--", alpha=0.6)
    finish_axes(ax, "Relative accuracy to full tokens (%)", ylim=(90, 100.4))
    ax.set_title("Retained full-token performance")
    ax.legend(frameon=False, loc="lower left", handlelength=1.8)

    fig.tight_layout(w_pad=1.3)
    save_fig(fig, out_dir, "qwen3_scaling_main_2panel")
    plt.close(fig)


def plot_accuracy_scaling(data, full, out_dir: Path):
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    full_y = [full[m]["acc"] for m in MODELS]
    ax.plot(MODEL_X, full_y, color=COLORS["Full"], marker=MARKERS["Full"], label="Full tokens", zorder=4)
    for ratio in RATIOS:
        y = [data[(m, ratio, OURS)]["acc"] for m in MODELS]
        ax.plot(MODEL_X, y, color=COLORS[ratio], marker=MARKERS[ratio], label=f"{OURS_LABEL} {ratio}")
    finish_axes(ax, "Average accuracy (%)", ylim=(70, 88.5))
    ax.set_title("Compression-aware model scaling")
    ax.legend(frameon=False, loc="lower right")
    save_fig(fig, out_dir, "qwen3_scaling_accuracy")
    plt.close(fig)


def plot_relative_scaling(data, out_dir: Path):
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for ratio in RATIOS:
        y = [data[(m, ratio, OURS)]["rel"] for m in MODELS]
        ax.plot(MODEL_X, y, color=COLORS[ratio], marker=MARKERS[ratio], label=f"{OURS_LABEL} {ratio}")
    ax.axhline(100, color="#202020", lw=1.0, ls="--", alpha=0.65)
    finish_axes(ax, "Relative accuracy to full tokens (%)", ylim=(90, 100.4))
    ax.set_title("Retention of full-token performance")
    ax.legend(frameon=False, loc="lower left")
    save_fig(fig, out_dir, "qwen3_scaling_relative")
    plt.close(fig)


def plot_advantage_heatmap(data, out_dir: Path):
    mat = np.zeros((len(RATIOS), len(MODELS)))
    labels = [["" for _ in MODELS] for _ in RATIOS]
    for i, ratio in enumerate(RATIOS):
        for j, model in enumerate(MODELS):
            ours = data[(model, ratio, OURS)]["acc"]
            best_base = max(data[(model, ratio, b)]["acc"] for b in BASELINES)
            mat[i, j] = ours - best_base
            labels[i][j] = f"{mat[i,j]:+.2f}"

    fig, ax = plt.subplots(figsize=(4.3, 2.45))
    vmax = max(1.6, float(np.max(np.abs(mat))))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODELS])
    ax.set_yticks(np.arange(len(RATIOS)))
    ax.set_yticklabels(RATIOS)
    ax.set_xlabel("Qwen3-VL model size")
    ax.set_ylabel("Retention ratio")
    ax.set_title(r"$F^3A$ gain over the best baseline")
    for i in range(len(RATIOS)):
        for j in range(len(MODELS)):
            color = "white" if abs(mat[i, j]) > 0.85 else "#202020"
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=8.5, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Acc. gain")
    for spine in ax.spines.values():
        spine.set_visible(False)
    save_fig(fig, out_dir, "qwen3_f3a_gain_heatmap")
    plt.close(fig)


def plot_method_average(data, out_dir: Path):
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    x = np.arange(len(RATIOS))
    width = 0.15
    offsets = np.linspace(-2, 2, len(METHODS)) * width
    for off, method in zip(offsets, METHODS):
        vals = [np.mean([data[(m, r, method)]["acc"] for m in MODELS]) for r in RATIOS]
        color = COLORS["F3A"] if method == OURS else COLORS[method]
        label = OURS_LABEL if method == OURS else method
        ax.bar(x + off, vals, width=width, label=label, color=color, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(RATIOS)
    ax.set_xlabel("Retention ratio")
    ax.set_ylabel("Mean accuracy across model sizes (%)")
    ax.set_ylim(72, 83.5)
    ax.set_title("Average performance across Qwen3-VL scales")
    ax.grid(True, axis="y", alpha=0.35)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.24))
    save_fig(fig, out_dir, "qwen3_method_average_by_retention")
    plt.close(fig)


def plot_compression_gap(data, full, out_dir: Path):
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for ratio in RATIOS:
        drops = [full[m]["acc"] - data[(m, ratio, OURS)]["acc"] for m in MODELS]
        ax.plot(MODEL_X, drops, color=COLORS[ratio], marker=MARKERS[ratio], label=f"{OURS_LABEL} {ratio}")
    finish_axes(ax, "Accuracy drop from full tokens", ylim=(0, 8))
    ax.set_title("Compression gap under scaling")
    ax.legend(frameon=False, loc="upper left")
    save_fig(fig, out_dir, "qwen3_compression_gap")
    plt.close(fig)


def plot_overview(data, full, out_dir: Path):
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.0))
    ax = axs[0, 0]
    ax.plot(MODEL_X, [full[m]["acc"] for m in MODELS], color=COLORS["Full"], marker="o", label="Full")
    for ratio in RATIOS:
        ax.plot(MODEL_X, [data[(m, ratio, OURS)]["acc"] for m in MODELS], color=COLORS[ratio], marker=MARKERS[ratio], label=f"{OURS_LABEL} {ratio}")
    finish_axes(ax, "Avg. accuracy (%)", ylim=(70, 88.5))
    ax.set_title("(a) Scaling curve")
    ax.legend(frameon=False, ncol=2, loc="lower right")

    ax = axs[0, 1]
    for ratio in RATIOS:
        ax.plot(MODEL_X, [data[(m, ratio, OURS)]["rel"] for m in MODELS], color=COLORS[ratio], marker=MARKERS[ratio], label=f"{OURS_LABEL} {ratio}")
    ax.axhline(100, color="#202020", lw=1.0, ls="--", alpha=0.55)
    finish_axes(ax, "Rel. to full (%)", ylim=(90, 100.4))
    ax.set_title("(b) Relative performance")
    ax.legend(frameon=False, loc="lower left")

    ax = axs[1, 0]
    mat = np.array([
        [data[(m, r, OURS)]["acc"] - max(data[(m, r, b)]["acc"] for b in BASELINES) for m in MODELS]
        for r in RATIOS
    ])
    vmax = max(1.6, float(np.max(np.abs(mat))))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODELS])
    ax.set_yticks(np.arange(len(RATIOS)))
    ax.set_yticklabels(RATIOS)
    ax.set_xlabel("Model size")
    ax.set_ylabel("Retention")
    ax.set_title("(c) Gain over best baseline")
    for i in range(len(RATIOS)):
        for j in range(len(MODELS)):
            ax.text(j, i, f"{mat[i,j]:+.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(mat[i,j]) > 0.85 else "#202020")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("Acc. gain")

    ax = axs[1, 1]
    x = np.arange(len(RATIOS))
    width = 0.15
    offsets = np.linspace(-2, 2, len(METHODS)) * width
    for off, method in zip(offsets, METHODS):
        vals = [np.mean([data[(m, r, method)]["acc"] for m in MODELS]) for r in RATIOS]
        color = COLORS["F3A"] if method == OURS else COLORS[method]
        label = OURS_LABEL if method == OURS else method
        ax.bar(x + off, vals, width=width, label=label, color=color, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(RATIOS)
    ax.set_xlabel("Retention")
    ax.set_ylabel("Mean Acc. (%)")
    ax.set_ylim(72, 83.5)
    ax.set_title("(d) Mean across scales")
    ax.grid(True, axis="y", alpha=0.35)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, ncol=2, fontsize=7.0, loc="upper center", bbox_to_anchor=(0.5, 1.02), columnspacing=0.8)

    fig.tight_layout(pad=1.0)
    save_fig(fig, out_dir, "qwen3_scaling_overview")
    plt.close(fig)


def write_csv(data, full, out_dir: Path) -> None:
    with (out_dir / "qwen3_scaling_acc_rel.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "ratio", "method", "acc", "rel"])
        for model in MODELS:
            writer.writerow([model, "100%", "Full", full[model]["acc"], full[model]["rel"]])
            for ratio in RATIOS:
                for method in METHODS:
                    row = data[(model, ratio, method)]
                    writer.writerow([model, ratio, method, row["acc"], row["rel"]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", type=Path, default=Path("tables/main_results.tex"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/figures/qwen3_scaling"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    data, full = parse_experiments(args.tex)
    write_csv(data, full, args.out_dir)
    plot_twopanel_scaling(data, full, args.out_dir)
    plot_accuracy_scaling(data, full, args.out_dir)
    plot_relative_scaling(data, args.out_dir)
    plot_advantage_heatmap(data, args.out_dir)
    plot_method_average(data, args.out_dir)
    plot_compression_gap(data, full, args.out_dir)
    plot_overview(data, full, args.out_dir)
    print(f"Saved figures to {args.out_dir}")


if __name__ == "__main__":
    main()
