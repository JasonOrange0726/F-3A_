import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from .datasets import load_hateful_memes_dataset
from .wrapper import F3AQwenVL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize routed Qwen visual tokens on top of an image.")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image-path", type=str, default="")
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--choices", nargs="*", default=[])
    parser.add_argument("--dataset", choices=["none", "hateful_memes"], default="none")
    parser.add_argument("--annotation-path", type=str, default="")
    parser.add_argument("--image-root", type=str, default="")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--final-keep", type=int, default=0)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def load_case(args: argparse.Namespace) -> tuple[Path, str, list[str], str]:
    if args.dataset == "hateful_memes":
        samples = load_hateful_memes_dataset(
            annotation_path=args.annotation_path,
            image_root=args.image_root,
            include_meme_text=True,
            max_samples=max(args.sample_index + 1, 1),
        )
        sample = samples[args.sample_index]
        return Path(sample.image_path), sample.instruction, sample.choices, sample.sample_id
    if not args.image_path or not args.question or not args.choices:
        raise ValueError("Provide --image-path, --question and --choices when dataset=none")
    return Path(args.image_path), args.question, args.choices, "custom"


def draw_overlay(image: Image.Image, selected: set[int], grid_size: tuple[int, int], title: str) -> Image.Image:
    grid_h, grid_w = grid_size
    base = image.convert("RGBA")
    width, height = base.size
    cell_w = width / grid_w
    cell_h = height / grid_h

    removed_overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    kept_overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    grid_overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    removed_draw = ImageDraw.Draw(removed_overlay)
    kept_draw = ImageDraw.Draw(kept_overlay)
    grid_draw = ImageDraw.Draw(grid_overlay)

    for idx in range(grid_h * grid_w):
        row = idx // grid_w
        col = idx % grid_w
        x0 = round(col * cell_w)
        y0 = round(row * cell_h)
        x1 = round((col + 1) * cell_w)
        y1 = round((row + 1) * cell_h)
        if idx in selected:
            kept_draw.rectangle((x0, y0, x1, y1), fill=(50, 205, 50, 70), outline=(20, 120, 20, 220), width=2)
        else:
            removed_draw.rectangle((x0, y0, x1, y1), fill=(220, 20, 60, 88))

    for row in range(grid_h + 1):
        y = round(row * cell_h)
        grid_draw.line((0, y, width, y), fill=(255, 255, 255, 120), width=1)
    for col in range(grid_w + 1):
        x = round(col * cell_w)
        grid_draw.line((x, 0, x, height), fill=(255, 255, 255, 120), width=1)

    composed = Image.alpha_composite(base, removed_overlay)
    composed = Image.alpha_composite(composed, kept_overlay)
    composed = Image.alpha_composite(composed, grid_overlay)

    canvas = Image.new("RGBA", (width, height + 44), (255, 255, 255, 255))
    canvas.paste(composed, (0, 44))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 12), title, fill=(0, 0, 0, 255))
    return canvas.convert("RGB")


def main() -> None:
    args = parse_args()
    image_path, question, choices, case_id = load_case(args)
    model = F3AQwenVL.from_pretrained(
        model_path=args.model_path,
        device=args.device,
        keep_ratio=args.keep_ratio,
    )
    result = model.predict_choice(
        image_source=image_path,
        instruction=question,
        choices=choices,
        routed=True,
        final_keep=args.final_keep if args.final_keep > 0 else None,
        keep_ratio=args.keep_ratio,
    )

    image = Image.open(image_path).convert("RGB")
    selected = set(result["selected_indices"])
    grid_size = tuple(result["grid_size"])
    title = f"{case_id}: kept={len(selected)}/{grid_size[0] * grid_size[1]}"
    output = draw_overlay(image=image, selected=selected, grid_size=grid_size, title=title)

    save_path = Path(args.output)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(save_path)
    print(f"saved_image={save_path}")


if __name__ == "__main__":
    main()
