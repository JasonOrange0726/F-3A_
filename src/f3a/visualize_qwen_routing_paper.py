from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import textwrap
from typing import Any, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageFilter
import torch

from .routing import _window_indices
from .image_eval import (
    load_local_dataset,
    materialize_image_source,
    parse_realworldqa_question,
)
from .wrapper import F3AQwenVL


@dataclass
class VisualizationCase:
    dataset: str
    sample_id: str
    image: Image.Image
    question: str
    task_type: str
    prompt_text: str
    choices: list[str]
    answer: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a paper-style F3A routing figure.")
    parser.add_argument(
        "--dataset",
        choices=[
            "custom",
            "pope",
            "chartqa",
            "ai2d",
            "hallusionbench",
            "mme",
            "scienceqa",
            "realworldqa",
            "textvqa",
        ],
        default="custom",
    )
    parser.add_argument("--dataset-root", type=str, default="")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--image-path", type=str, default="")
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--choices", nargs="*", default=[])
    parser.add_argument("--answer", type=str, default="")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--keep-ratio", type=float, default=0.6)
    parser.add_argument("--final-keep", type=int, default=0)
    parser.add_argument("--compare-full", action="store_true")
    parser.add_argument("--text-conditioning-mode", type=str, default="universal_three_cue")
    parser.add_argument(
        "--pope-category",
        choices=["all", "random", "popular", "adversarial"],
        default="all",
    )
    parser.add_argument(
        "--figure-style",
        choices=["soft_triptych", "process"],
        default="soft_triptych",
    )
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def _truncate(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text).strip())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "..."


def _choice_line(choices: Sequence[str], limit_per_choice: int = 24) -> str:
    parts = []
    for idx, choice in enumerate(choices):
        label = chr(ord("A") + idx)
        parts.append(f"{label}. {_truncate(choice, limit_per_choice)}")
    return "  |  ".join(parts)


def _short_cues(model: F3AQwenVL, question: str) -> list[str]:
    if model.text_conditioning_mode not in {"universal_three_cue", "universal_vqa", "multi_cue"}:
        return ["prompt-conditioned routing"]
    _, cue_bank = model._build_universal_three_cues(question)
    cleaned = []
    for cue in cue_bank:
        cue = cue.replace("Find the region relevant to:", "target:")
        cue = cue.replace("Focus on ", "focus: ")
        cue = cue.rstrip(".")
        cleaned.append(_truncate(cue, 52))
    return cleaned[:3]


def _answer_text(task_type: str, output: dict[str, Any], choices: Sequence[str]) -> str:
    if task_type == "choice":
        prediction = int(output["prediction"])
        label = chr(ord("A") + prediction)
        return f"({label}) {_truncate(choices[prediction], 48)}"
    return _truncate(str(output["text"]), 64)


def _load_custom_case(args: argparse.Namespace, model: F3AQwenVL) -> VisualizationCase:
    if not args.image_path or not args.question:
        raise ValueError("custom mode requires --image-path and --question")
    image = Image.open(args.image_path).convert("RGB")
    if args.choices:
        prompt_text = model.format_multiple_choice_prompt(args.question, args.choices)
        task_type = "choice"
    else:
        prompt_text = model.format_short_answer_prompt(args.question)
        task_type = "open"
    return VisualizationCase(
        dataset="custom",
        sample_id="custom",
        image=image,
        question=args.question,
        task_type=task_type,
        prompt_text=prompt_text,
        choices=list(args.choices),
        answer=args.answer,
    )


def _load_benchmark_case(args: argparse.Namespace, model: F3AQwenVL) -> VisualizationCase:
    dataset = load_local_dataset(args)
    row = dataset[int(args.sample_index)]
    image = materialize_image_source(row["image"])
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")

    if args.dataset in {"pope", "hallusionbench", "mme"}:
        question = str(row["question"])
        answer = str(row.get("answer", row.get("gt_answer", "")))
        return VisualizationCase(
            dataset=args.dataset,
            sample_id=str(row.get("question_id", row.get("id", args.sample_index))),
            image=image,
            question=question,
            task_type="open",
            prompt_text=model.format_yes_no_prompt(question),
            choices=[],
            answer=answer,
        )

    if args.dataset in {"chartqa", "textvqa"}:
        question = str(row["question"])
        answer = row.get("answer", row.get("answers", ""))
        if isinstance(answer, list):
            answer = "; ".join(str(item) for item in answer[:3])
        return VisualizationCase(
            dataset=args.dataset,
            sample_id=str(row.get("question_id", row.get("id", args.sample_index))),
            image=image,
            question=question,
            task_type="open",
            prompt_text=model.format_short_answer_prompt(question),
            choices=[],
            answer=str(answer),
        )

    if args.dataset == "ai2d":
        question = str(row["question"])
        choices = [str(option) for option in row["options"]]
        answer_index = int(row["answer"])
        return VisualizationCase(
            dataset=args.dataset,
            sample_id=str(row.get("question_id", row.get("id", args.sample_index))),
            image=image,
            question=question,
            task_type="choice",
            prompt_text=model.format_multiple_choice_prompt(question, choices),
            choices=choices,
            answer=f"({chr(ord('A') + answer_index)}) {choices[answer_index]}",
        )

    if args.dataset == "scienceqa":
        prompt_parts = []
        hint = str(row.get("hint", "")).strip()
        if hint:
            prompt_parts.append(f"Context: {hint}")
        prompt_parts.append(str(row["question"]))
        question = "\n".join(prompt_parts)
        choices = [str(choice) for choice in row["choices"]]
        answer_index = int(row["answer"])
        return VisualizationCase(
            dataset=args.dataset,
            sample_id=str(row.get("question_id", row.get("id", args.sample_index))),
            image=image,
            question=question,
            task_type="choice",
            prompt_text=model.format_multiple_choice_prompt(question, choices),
            choices=choices,
            answer=f"({chr(ord('A') + answer_index)}) {choices[answer_index]}",
        )

    if args.dataset == "realworldqa":
        question, choice_labels, choices = parse_realworldqa_question(str(row["question"]))
        answer = str(row["answer"]).strip()
        if choices:
            return VisualizationCase(
                dataset=args.dataset,
                sample_id=str(row.get("question_id", row.get("id", args.sample_index))),
                image=image,
                question=question,
                task_type="choice",
                prompt_text=model.format_multiple_choice_prompt(question, choices),
                choices=choices,
                answer=answer,
            )
        return VisualizationCase(
            dataset=args.dataset,
            sample_id=str(row.get("question_id", row.get("id", args.sample_index))),
            image=image,
            question=question,
            task_type="open",
            prompt_text=model.format_short_answer_prompt(question),
            choices=[],
            answer=answer,
        )

    raise ValueError(f"Unsupported dataset: {args.dataset}")


def load_case(args: argparse.Namespace, model: F3AQwenVL) -> VisualizationCase:
    if args.dataset == "custom":
        return _load_custom_case(args, model)
    return _load_benchmark_case(args, model)


def _normalize_scores(values: Optional[torch.Tensor], grid_size: tuple[int, int]) -> Optional[np.ndarray]:
    if values is None:
        return None
    grid_h, grid_w = grid_size
    tensor = values.detach().float().cpu().reshape(grid_h, grid_w)
    tensor = tensor - tensor.min()
    denom = tensor.max().clamp_min(1e-6)
    return (tensor / denom).numpy()


def _selection_overlay(
    grid_size: tuple[int, int],
    stage_sets: list[tuple[set[int], tuple[float, float, float, float]]],
    selected_union: Optional[set[int]] = None,
    dim_alpha: float = 0.34,
) -> np.ndarray:
    grid_h, grid_w = grid_size
    rgba = np.zeros((grid_h, grid_w, 4), dtype=np.float32)
    if selected_union is not None:
        all_idx = set(range(grid_h * grid_w))
        for idx in all_idx - selected_union:
            row = idx // grid_w
            col = idx % grid_w
            rgba[row, col] = np.array([0.0, 0.0, 0.0, dim_alpha], dtype=np.float32)
    for indices, color in stage_sets:
        for idx in indices:
            row = idx // grid_w
            col = idx % grid_w
            rgba[row, col] = np.array(color, dtype=np.float32)
    return rgba


def _show_image(ax, image: Image.Image, title: str) -> None:
    ax.imshow(image)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _show_heatmap(ax, image: Image.Image, score_map: Optional[np.ndarray], title: str, cmap: str) -> None:
    _show_image(ax, image, title)
    if score_map is None:
        return
    width, height = image.size
    ax.imshow(
        score_map,
        cmap=cmap,
        alpha=0.58,
        interpolation="nearest",
        extent=(0, width, height, 0),
        vmin=0.0,
        vmax=1.0,
    )


def _show_overlay(ax, image: Image.Image, overlay: np.ndarray, title: str) -> None:
    _show_image(ax, image, title)
    width, height = image.size
    ax.imshow(
        overlay,
        interpolation="nearest",
        extent=(0, width, height, 0),
    )


def _smooth_score_map(
    score_map: Optional[np.ndarray],
    image_size: tuple[int, int],
    blur_radius: float = 10.0,
) -> Optional[np.ndarray]:
    if score_map is None:
        return None
    width, height = image_size
    base = np.clip(score_map, 0.0, 1.0)
    heat = Image.fromarray((base * 255.0).astype(np.uint8), mode="L")
    heat = heat.resize((width, height), resample=Image.Resampling.BICUBIC)
    if blur_radius > 0:
        heat = heat.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    smooth = np.asarray(heat, dtype=np.float32) / 255.0
    smooth = smooth - smooth.min()
    denom = max(float(smooth.max()), 1e-6)
    return smooth / denom


def _draw_selected_patch_outlines(
    ax,
    image_size: tuple[int, int],
    grid_size: tuple[int, int],
    main_tokens: set[int],
    context_tokens: set[int],
    rescue_tokens: set[int],
) -> None:
    width, height = image_size
    grid_h, grid_w = grid_size
    cell_w = width / grid_w
    cell_h = height / grid_h

    color_map = {
        "main": ("#20bf55", (32 / 255.0, 191 / 255.0, 85 / 255.0, 0.12)),
        "context": ("#19b5d1", (25 / 255.0, 181 / 255.0, 209 / 255.0, 0.13)),
        "rescue": ("#f1c40f", (241 / 255.0, 196 / 255.0, 15 / 255.0, 0.18)),
    }
    token_groups = []
    for idx in sorted(main_tokens):
        token_groups.append((idx, "main"))
    for idx in sorted(context_tokens):
        token_groups.append((idx, "context"))
    for idx in sorted(rescue_tokens):
        token_groups.append((idx, "rescue"))

    for idx, group in token_groups:
        row = idx // grid_w
        col = idx % grid_w
        x0 = col * cell_w
        y0 = row * cell_h
        edge, fill = color_map[group]
        rect = Rectangle(
            (x0, y0),
            cell_w,
            cell_h,
            linewidth=2.0,
            edgecolor=edge,
            facecolor=fill,
        )
        ax.add_patch(rect)


def build_soft_triptych_figure(
    case: VisualizationCase,
    image: Image.Image,
    grid_size: tuple[int, int],
    query_map: Optional[np.ndarray],
    agreement_map: Optional[np.ndarray],
    main_tokens: set[int],
    context_tokens: set[int],
    rescue_tokens: set[int],
    cues: Sequence[str],
    answer_line: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 6.2))
    plt.subplots_adjust(left=0.03, right=0.985, top=0.73, bottom=0.10, wspace=0.035)

    smooth_query = _smooth_score_map(query_map, image.size, blur_radius=10.0)
    smooth_agreement = _smooth_score_map(agreement_map, image.size, blur_radius=10.0)

    _show_image(axes[0], image, "Text-guided coarse response")
    if smooth_query is not None:
        axes[0].imshow(
            smooth_query,
            cmap="magma",
            alpha=0.62,
            interpolation="bilinear",
            extent=(0, image.size[0], image.size[1], 0),
            vmin=0.0,
            vmax=1.0,
        )

    _show_image(axes[1], image, "Consensus / agreement")
    if smooth_agreement is not None:
        axes[1].imshow(
            smooth_agreement,
            cmap="viridis",
            alpha=0.60,
            interpolation="bilinear",
            extent=(0, image.size[0], image.size[1], 0),
            vmin=0.0,
            vmax=1.0,
        )

    _show_image(axes[2], image, "Final routed tokens")
    dim_mask = np.zeros((image.size[1], image.size[0], 4), dtype=np.float32)
    dim_mask[..., 3] = 0.36
    axes[2].imshow(dim_mask, extent=(0, image.size[0], image.size[1], 0))
    _draw_selected_patch_outlines(
        ax=axes[2],
        image_size=image.size,
        grid_size=grid_size,
        main_tokens=main_tokens,
        context_tokens=context_tokens,
        rescue_tokens=rescue_tokens,
    )

    question_text = textwrap.fill(case.question, width=110)
    fig.text(
        0.03,
        0.965,
        f"{case.dataset.upper()} | sample {case.sample_id}",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(0.03, 0.920, question_text, ha="left", va="top", fontsize=12)

    if case.choices:
        fig.text(
            0.03,
            0.865,
            textwrap.fill(_choice_line(case.choices), width=120),
            ha="left",
            va="top",
            fontsize=10,
            color="#333333",
        )
        cue_y = 0.807
    else:
        cue_y = 0.855

    x = 0.03
    for cue in cues:
        fig.text(
            x,
            cue_y,
            cue,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#eef3ff", "edgecolor": "#c9d6ff"},
        )
        x += min(0.24, 0.012 * len(cue) + 0.05)

    fig.text(0.03, 0.045, answer_line, ha="left", va="bottom", fontsize=11)
    fig.text(
        0.985,
        0.045,
        "green=target-driven tokens, cyan=context-preserving tokens, yellow=exploration-rescue tokens",
        ha="right",
        va="bottom",
        fontsize=10,
        color="#444444",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_routing_figure(
    case: VisualizationCase,
    image: Image.Image,
    grid_size: tuple[int, int],
    query_map: Optional[np.ndarray],
    agreement_map: Optional[np.ndarray],
    scaffold: set[int],
    expanded: set[int],
    jump_fill: set[int],
    final_selected: set[int],
    main_tokens: set[int],
    context_tokens: set[int],
    rescue_tokens: set[int],
    cues: Sequence[str],
    answer_line: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(22, 6.4))
    plt.subplots_adjust(left=0.02, right=0.985, top=0.70, bottom=0.08, wspace=0.03)

    scaffold_color = (0.14, 0.78, 0.93, 0.72)
    expand_color = (1.00, 0.62, 0.16, 0.68)
    jump_color = (0.98, 0.90, 0.18, 0.80)
    final_color = (0.17, 0.84, 0.45, 0.70)
    context_color = (0.10, 0.71, 0.82, 0.72)

    _show_image(axes[0], image, "Original image")
    _show_heatmap(axes[1], image, query_map, "Text-guided coarse response", cmap="magma")
    _show_overlay(
        axes[2],
        image,
        _selection_overlay(grid_size, [(scaffold, scaffold_color)], selected_union=scaffold),
        "Local scaffold",
    )
    _show_overlay(
        axes[3],
        image,
        _selection_overlay(
            grid_size,
            [(scaffold, scaffold_color), (expanded - scaffold, expand_color)],
            selected_union=expanded,
        ),
        "Consensus-guided expansion",
    )
    if agreement_map is not None:
        axes[3].imshow(
            agreement_map,
            cmap="viridis",
            alpha=0.34,
            interpolation="nearest",
            extent=(0, image.size[0], image.size[1], 0),
            vmin=0.0,
            vmax=1.0,
        )
    _show_overlay(
        axes[4],
        image,
        _selection_overlay(
            grid_size,
            [
                (main_tokens, final_color),
                (context_tokens, context_color),
                (rescue_tokens, jump_color),
            ],
            selected_union=final_selected,
        ),
        "Final routed tokens",
    )

    question_text = textwrap.fill(case.question, width=110)
    fig.text(
        0.02,
        0.955,
        f"{case.dataset.upper()} | sample {case.sample_id}",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(0.02, 0.912, question_text, ha="left", va="top", fontsize=12)

    if case.choices:
        fig.text(
            0.02,
            0.855,
            textwrap.fill(_choice_line(case.choices), width=140),
            ha="left",
            va="top",
            fontsize=10,
            color="#333333",
        )
        cue_y = 0.795
    else:
        cue_y = 0.84

    x = 0.02
    for cue in cues:
        fig.text(
            x,
            cue_y,
            cue,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#eef3ff", "edgecolor": "#c9d6ff"},
        )
        x += min(0.24, 0.012 * len(cue) + 0.05)

    fig.text(0.02, 0.04, answer_line, ha="left", va="bottom", fontsize=11)
    fig.text(
        0.98,
        0.04,
        "cyan=scaffold, orange=agreement expansion, green=target-driven final, blue=context final, yellow=exploration rescue",
        ha="right",
        va="bottom",
        fontsize=10,
        color="#444444",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    model = F3AQwenVL.from_pretrained(
        model_path=args.model_path,
        device=args.device,
        keep_ratio=args.keep_ratio,
        text_conditioning_mode=args.text_conditioning_mode,
    )
    case = load_case(args, model)

    inputs = model.prepare_inputs(image_source=case.image, prompt_text=case.prompt_text)
    visual_tokens, grid_size = model._get_single_image_tokens(
        pixel_values=inputs["pixel_values"],
        image_grid_thw=inputs["image_grid_thw"],
    )
    odor_cue, hypothesis_cues, contrast_cues = model._build_text_conditioning(
        instruction=case.question,
        choices=case.choices,
        inputs=inputs,
    )
    with torch.inference_mode():
        routing = model.router(
            visual_tokens=visual_tokens.unsqueeze(0),
            odor_cue=odor_cue,
            hypothesis_cues=hypothesis_cues,
            contrast_cues=contrast_cues,
            grid_size=grid_size,
            final_keep=args.final_keep if args.final_keep > 0 else None,
            keep_ratio=args.keep_ratio,
        )

    if case.task_type == "choice":
        routed_output = model.predict_choice(
            image_source=case.image,
            instruction=case.question,
            choices=case.choices,
            routed=True,
            final_keep=args.final_keep if args.final_keep > 0 else None,
            keep_ratio=args.keep_ratio,
        )
        full_output = None
        if args.compare_full:
            full_output = model.predict_choice(
                image_source=case.image,
                instruction=case.question,
                choices=case.choices,
                routed=False,
            )
    else:
        routed_output = model.generate_answer(
            image_source=case.image,
            question=case.question,
            prompt_text=case.prompt_text,
            routed=True,
            final_keep=args.final_keep if args.final_keep > 0 else None,
            keep_ratio=args.keep_ratio,
            max_new_tokens=16,
        )
        full_output = None
        if args.compare_full:
            full_output = model.generate_answer(
                image_source=case.image,
                question=case.question,
                prompt_text=case.prompt_text,
                routed=False,
                max_new_tokens=16,
            )

    query_map = _normalize_scores(routing.query_scores, grid_size)
    agreement_map = _normalize_scores(routing.agreement_scores, grid_size)
    scaffold = set(routing.scaffold_indices.detach().cpu().tolist()) if routing.scaffold_indices is not None else set()
    exploit = set(routing.exploit_indices.detach().cpu().tolist()) if routing.exploit_indices is not None else set()
    jump = set(routing.jump_indices.detach().cpu().tolist()) if routing.jump_indices is not None else set()
    fill = set(routing.fill_indices.detach().cpu().tolist()) if routing.fill_indices is not None else set()
    final_selected = set(routing.selected_indices.detach().cpu().tolist())
    expanded = scaffold | exploit
    jump_fill = jump | fill
    rescue_tokens = final_selected & jump_fill

    context_windows = set(routing.context_window_indices.detach().cpu().tolist()) if routing.context_window_indices is not None else set()
    anchor_windows = set(routing.anchor_window_indices.detach().cpu().tolist()) if routing.anchor_window_indices is not None else set()
    context_like_windows = context_windows | anchor_windows
    context_tokens: set[int] = set()
    if context_like_windows:
        window_size = int(model.router.local_window_size)
        windows = _window_indices(grid_size[0], grid_size[1], window_size, device=torch.device("cpu"))
        token_to_window: dict[int, int] = {}
        for window_idx in range(windows.size(0)):
            valid_tokens = windows[window_idx][windows[window_idx] >= 0].tolist()
            for token_idx in valid_tokens:
                token_to_window[int(token_idx)] = int(window_idx)
        context_tokens = {
            idx
            for idx in final_selected - rescue_tokens
            if token_to_window.get(idx) in context_like_windows
        }
    main_tokens = final_selected - context_tokens - rescue_tokens

    total_tokens = grid_size[0] * grid_size[1]
    keep_tokens = len(final_selected)
    answer_line = f"Gold: {_truncate(case.answer, 60)} | Routed: {_answer_text(case.task_type, routed_output, case.choices)}"
    if full_output is not None:
        answer_line += f" | Full: {_answer_text(case.task_type, full_output, case.choices)}"
    answer_line += f" | kept {keep_tokens}/{total_tokens} ({100.0 * keep_tokens / max(1, total_tokens):.1f}%)"

    cues = _short_cues(model, case.question)
    if args.figure_style == "process":
        build_routing_figure(
            case=case,
            image=case.image,
            grid_size=grid_size,
            query_map=query_map,
            agreement_map=agreement_map,
            scaffold=scaffold,
            expanded=expanded,
            jump_fill=jump_fill,
            final_selected=final_selected,
            main_tokens=main_tokens,
            context_tokens=context_tokens,
            rescue_tokens=rescue_tokens,
            cues=cues,
            answer_line=answer_line,
            output_path=Path(args.output),
        )
    else:
        build_soft_triptych_figure(
            case=case,
            image=case.image,
            grid_size=grid_size,
            query_map=query_map,
            agreement_map=agreement_map,
            main_tokens=main_tokens,
            context_tokens=context_tokens,
            rescue_tokens=rescue_tokens,
            cues=cues,
            answer_line=answer_line,
            output_path=Path(args.output),
        )
    print(f"saved_image={args.output}")


if __name__ == "__main__":
    main()
