from dataclasses import dataclass, field
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Any, Optional

from PIL import Image
import pandas as pd


@dataclass
class ChoiceSample:
    sample_id: str
    instruction: str
    choices: list[str]
    answer_index: Optional[int] = None
    image_path: Optional[str] = None
    image_bytes: Optional[bytes] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def load_image(self) -> Image.Image:
        if self.image_bytes is not None:
            return Image.open(BytesIO(self.image_bytes)).convert("RGB")
        if self.image_path is None:
            raise ValueError("Sample does not contain an image")
        return Image.open(self.image_path).convert("RGB")


def choice_labels(num_choices: int) -> list[str]:
    if num_choices <= 0 or num_choices > 26:
        raise ValueError("num_choices must be in [1, 26]")
    return [chr(ord("A") + idx) for idx in range(num_choices)]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def _parse_answer_index(answer: Any, choices: list[str]) -> Optional[int]:
    if answer is None:
        return None
    if isinstance(answer, int):
        if 0 <= answer < len(choices):
            return answer
        return None
    answer_text = str(answer).strip()
    letter_match = re.search(r"\(([A-Z])\)", answer_text)
    if letter_match is None:
        letter_match = re.match(r"^\s*([A-Z])\b", answer_text)
    if letter_match is not None:
        idx = ord(letter_match.group(1)) - ord("A")
        if 0 <= idx < len(choices):
            return idx

    normalized_answer = _normalize_text(answer_text)
    normalized_choices = [_normalize_text(choice) for choice in choices]
    for idx, choice in enumerate(normalized_choices):
        if normalized_answer == choice:
            return idx
        if normalized_answer.endswith(choice):
            return idx
    return None


def _context_bytes_to_image_bytes(value: Any) -> Optional[bytes]:
    if not isinstance(value, dict):
        return None
    blob = value.get("bytes")
    if not isinstance(blob, (bytes, bytearray)):
        return None
    try:
        Image.open(BytesIO(blob)).verify()
    except Exception:
        return None
    return bytes(blob)


def load_jsonl_choice_dataset(
    jsonl_path: str,
    image_root: Optional[str] = None,
    image_key: str = "image",
    instruction_key: str = "question",
    choices_key: str = "choices",
    answer_key: str = "answer",
    answer_index_key: str = "answer_index",
    id_key: str = "id",
    max_samples: int = 0,
) -> list[ChoiceSample]:
    path = Path(jsonl_path)
    root = Path(image_root) if image_root else path.parent
    samples: list[ChoiceSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            choices = list(row[choices_key])
            image_value = row.get(image_key)
            image_path = None if image_value is None else str((root / image_value).resolve())
            answer_index = row.get(answer_index_key)
            if answer_index is None:
                answer_index = _parse_answer_index(row.get(answer_key), choices)
            samples.append(
                ChoiceSample(
                    sample_id=str(row.get(id_key, len(samples))),
                    instruction=str(row[instruction_key]),
                    choices=choices,
                    answer_index=answer_index,
                    image_path=image_path,
                    metadata={k: v for k, v in row.items() if k not in {image_key, instruction_key, choices_key, answer_key, answer_index_key, id_key}},
                )
            )
            if max_samples > 0 and len(samples) >= max_samples:
                break
    return samples


def load_mmau_image_parquet(
    parquet_path: str,
    max_samples: int = 0,
) -> list[ChoiceSample]:
    table = pd.read_parquet(parquet_path)
    samples: list[ChoiceSample] = []
    for _, row in table.iterrows():
        image_bytes = None
        for key in ("context", "image", "image_data", "media"):
            image_bytes = _context_bytes_to_image_bytes(row.get(key))
            if image_bytes is not None:
                break
        if image_bytes is None:
            continue

        choices_raw = row.get("choices")
        if choices_raw is None:
            choices_raw = row.get("options")
        choices = [str(choice) for choice in list(choices_raw)]
        instruction = row.get("instruction")
        if instruction is None:
            instruction = row.get("question")
        answer_index = row.get("answer_index")
        if answer_index is None:
            answer_index = _parse_answer_index(row.get("answer"), choices)

        metadata = {}
        for key in ("other_attributes", "dataset", "split", "category", "version", "key"):
            if key in row and row[key] is not None:
                metadata[key] = row[key]

        sample_id = str(metadata.get("key", len(samples)))
        samples.append(
            ChoiceSample(
                sample_id=sample_id,
                instruction=str(instruction),
                choices=choices,
                answer_index=answer_index,
                image_bytes=image_bytes,
                metadata=metadata,
            )
        )
        if max_samples > 0 and len(samples) >= max_samples:
            break
    return samples


def load_hateful_memes_dataset(
    annotation_path: str,
    image_root: str,
    include_meme_text: bool = True,
    max_samples: int = 0,
    skip_missing_images: bool = True,
) -> list[ChoiceSample]:
    annotation_file = Path(annotation_path)
    root = Path(image_root)
    samples: list[ChoiceSample] = []
    with annotation_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            answer_index = None
            hate_value = row.get("gold_hate")
            if hate_value is not None:
                if isinstance(hate_value, list):
                    hate_value = hate_value[0]
                answer_index = 1 if str(hate_value).strip() == "hateful" else 0

            meme_text = str(row.get("text", "")).strip()
            if include_meme_text and meme_text:
                instruction = (
                    "Determine whether this meme is hateful.\n"
                    f"Meme text: {meme_text}\n"
                    "Choose the best answer."
                )
            else:
                instruction = "Determine whether this meme is hateful. Choose the best answer."

            image_path = (root / Path(row["img"]).name).resolve()
            if skip_missing_images and not image_path.exists():
                continue

            samples.append(
                ChoiceSample(
                    sample_id=str(row.get("id", len(samples))),
                    instruction=instruction,
                    choices=["not hateful", "hateful"],
                    answer_index=answer_index,
                    image_path=str(image_path),
                    metadata={
                        "set_name": row.get("set_name"),
                        "meme_text": meme_text,
                        "gold_hate": hate_value,
                        "pc": row.get("pc"),
                        "attack": row.get("attack"),
                    },
                )
            )
            if max_samples > 0 and len(samples) >= max_samples:
                break
    return samples
