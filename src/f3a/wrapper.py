from dataclasses import dataclass
import os
from pathlib import Path
import re
import time
from typing import Any, Optional, Sequence

from PIL import Image
import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
)
try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )
except ImportError:
    create_causal_mask = None
    create_sliding_window_causal_mask = None

from .defaults import DEFAULT_MODEL_PATH, DEFAULT_ROUTER_CONFIG, DEFAULT_TEXT_CONDITIONING_MODE
from .datasets import ChoiceSample, choice_labels
from .paths import resolve_model_path
from .routing import OdorConditionedF3ARouter, RoutingResult
from .visionzip import VisionZipConfig, VisionZipResult, qwen3_visionzip_projected_tokens, zip_projected_tokens


@dataclass
class RoutedPrefill:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    mm_token_type_ids: torch.Tensor
    position_ids: torch.Tensor
    inputs_embeds: torch.Tensor
    routing: RoutingResult
    full_visual_token_count: int
    prefill_seq_len: int


def _resolve_qwen_vl_model_class(model_path: str):
    config = AutoConfig.from_pretrained(model_path)
    model_type = getattr(config, "model_type", "")
    import transformers

    if model_type == "qwen2_5_vl":
        model_cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
        if model_cls is None:
            raise ImportError("Installed transformers does not expose Qwen2_5_VLForConditionalGeneration")
        return model_cls, model_type
    if model_type == "qwen3_vl":
        model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
        if model_cls is None:
            raise ImportError("Installed transformers does not expose Qwen3VLForConditionalGeneration")
        return model_cls, model_type
    if model_type == "qwen3_vl_moe":
        model_cls = getattr(transformers, "Qwen3VLMoeForConditionalGeneration", None)
        if model_cls is None:
            raise ImportError("Installed transformers does not expose Qwen3VLMoeForConditionalGeneration")
        return model_cls, model_type
    raise ValueError(f"Unsupported vision-language model_type: {model_type}")


class F3AQwenVL:
    _READING_KEYWORDS = (
        "read",
        "text",
        "word",
        "written",
        "say",
        "says",
        "letter",
        "letters",
        "number",
        "numbers",
        "caption",
    )
    _CHART_KEYWORDS = (
        "chart",
        "graph",
        "plot",
        "axis",
        "axes",
        "legend",
        "diagram",
        "bar",
        "bars",
        "line chart",
        "pie chart",
        "table",
        "flowchart",
    )
    _SPATIAL_KEYWORDS = (
        "left",
        "right",
        "above",
        "below",
        "under",
        "over",
        "next to",
        "between",
        "behind",
        "in front of",
        "closest",
        "farthest",
    )
    _COUNT_COMPARE_KEYWORDS = (
        "how many",
        "more than",
        "less than",
        "highest",
        "lowest",
        "largest",
        "smallest",
        "most",
        "least",
        "compare",
    )
    _TARGET_STOPWORDS = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "with",
        "and",
        "or",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
        "there",
        "image",
        "picture",
        "photo",
        "shown",
        "showing",
    }

    def __init__(
        self,
        model,
        processor: AutoProcessor,
        device: torch.device,
        router: OdorConditionedF3ARouter,
        text_conditioning_mode: str = "multi_cue",
        fastv_prune_layer: int = 2,
        model_type: str = "qwen2_5_vl",
    ) -> None:
        self.model = model.eval()
        self.processor = processor
        self.device = device
        self.router = router.to(device).eval()
        self.text_conditioning_mode = text_conditioning_mode
        self.fastv_prune_layer = max(1, fastv_prune_layer)
        self.model_type = model_type

    @staticmethod
    def _trace(message: str) -> None:
        if os.environ.get("F3A_TRACE", "0") == "1":
            print(f"[trace] {time.strftime('%H:%M:%S')} {message}", flush=True)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str = "cuda:0",
        torch_dtype: str = "auto",
        device_map: Optional[str] = None,
        router_heads: int = DEFAULT_ROUTER_CONFIG["router_heads"],
        token_nonzero: int = DEFAULT_ROUTER_CONFIG["token_nonzero"],
        odor_nonzero: int = DEFAULT_ROUTER_CONFIG["odor_nonzero"],
        head_topk: int = DEFAULT_ROUTER_CONFIG["head_topk"],
        odor_topk: int = DEFAULT_ROUTER_CONFIG["odor_topk"],
        local_window_size: int = DEFAULT_ROUTER_CONFIG["local_window_size"],
        scaffold_keep: int = DEFAULT_ROUTER_CONFIG["scaffold_keep"],
        keep_ratio: float = DEFAULT_ROUTER_CONFIG["keep_ratio"],
        smell_weight: float = DEFAULT_ROUTER_CONFIG["smell_weight"],
        odor_gate_scale: float = DEFAULT_ROUTER_CONFIG["odor_gate_scale"],
        odor_temperature: float = DEFAULT_ROUTER_CONFIG["odor_temperature"],
        hypothesis_weight: float = DEFAULT_ROUTER_CONFIG["hypothesis_weight"],
        contrast_weight: float = DEFAULT_ROUTER_CONFIG["contrast_weight"],
        agreement_weight: float = DEFAULT_ROUTER_CONFIG["agreement_weight"],
        repulsion_weight: float = DEFAULT_ROUTER_CONFIG["repulsion_weight"],
        routing_mode: str = "foraging",
        region_agreement_weight: float = DEFAULT_ROUTER_CONFIG["region_agreement_weight"],
        lockon_weight: float = DEFAULT_ROUTER_CONFIG["lockon_weight"],
        visit_weight: float = DEFAULT_ROUTER_CONFIG["visit_weight"],
        uncertainty_weight: float = DEFAULT_ROUTER_CONFIG["uncertainty_weight"],
        jump_ratio: float = DEFAULT_ROUTER_CONFIG["jump_ratio"],
        ivc_keep_ratio: float = DEFAULT_ROUTER_CONFIG["ivc_keep_ratio"],
        ivc_window_bonus: float = DEFAULT_ROUTER_CONFIG["ivc_window_bonus"],
        coarse_pos_weight: float = DEFAULT_ROUTER_CONFIG["coarse_pos_weight"],
        local_pos_weight: float = DEFAULT_ROUTER_CONFIG["local_pos_weight"],
        anchor_pos_weight: float = DEFAULT_ROUTER_CONFIG["anchor_pos_weight"],
        text_conditioning_mode: str = DEFAULT_TEXT_CONDITIONING_MODE,
        fastv_prune_layer: int = DEFAULT_ROUTER_CONFIG["fastv_prune_layer"],
        router_seed: int = DEFAULT_ROUTER_CONFIG["router_seed"],
    ) -> "F3AQwenVL":
        resolved_device = torch.device(device)
        resolved_model_path = resolve_model_path(model_path)
        if torch_dtype == "auto":
            if resolved_device.type == "cuda" and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif resolved_device.type == "cuda":
                dtype = torch.float16
            else:
                dtype = torch.float32
        else:
            dtype = getattr(torch, torch_dtype)

        print(f"[load] resolved_model_path={resolved_model_path}", flush=True)
        print("[load] loading processor", flush=True)
        processor = AutoProcessor.from_pretrained(resolved_model_path)
        print("[load] processor loaded", flush=True)
        model_cls, model_type = _resolve_qwen_vl_model_class(resolved_model_path)
        print(f"[load] model_class={model_cls.__name__} model_type={model_type} dtype={dtype} device_map={device_map or 'none'}", flush=True)
        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if device_map:
            load_kwargs["device_map"] = device_map
        print("[load] loading weights", flush=True)
        model = model_cls.from_pretrained(resolved_model_path, **load_kwargs)
        print("[load] weights loaded", flush=True)
        if not device_map:
            print(f"[load] moving model to {resolved_device}", flush=True)
            model.to(resolved_device)
            print(f"[load] model moved to {resolved_device}", flush=True)
        vision_config = model.config.vision_config
        vision_token_dim = getattr(vision_config, "out_hidden_size", getattr(vision_config, "hidden_size"))
        odor_dim = model.config.text_config.hidden_size
        text_config = model.config.text_config
        num_attention_heads = getattr(text_config, "num_attention_heads", 32)
        ivc_rope_dim = getattr(text_config, "head_dim", None)
        if ivc_rope_dim is None:
            ivc_rope_dim = max(2, odor_dim // max(1, num_attention_heads))
        router = OdorConditionedF3ARouter(
            token_dim=vision_token_dim,
            odor_dim=odor_dim,
            num_heads=router_heads,
            token_nonzero=token_nonzero,
            odor_nonzero=odor_nonzero,
            head_topk=head_topk,
            odor_topk=odor_topk,
            local_window_size=local_window_size,
            scaffold_keep=scaffold_keep,
            default_keep_ratio=keep_ratio,
            smell_weight=smell_weight,
            odor_gate_scale=odor_gate_scale,
            odor_temperature=odor_temperature,
            hypothesis_weight=hypothesis_weight,
            contrast_weight=contrast_weight,
            agreement_weight=agreement_weight,
            repulsion_weight=repulsion_weight,
            routing_mode=routing_mode,
            region_agreement_weight=region_agreement_weight,
            lockon_weight=lockon_weight,
            visit_weight=visit_weight,
            uncertainty_weight=uncertainty_weight,
            jump_ratio=jump_ratio,
            ivc_keep_ratio=ivc_keep_ratio,
            ivc_rope_dim=ivc_rope_dim,
            ivc_window_bonus=ivc_window_bonus,
            coarse_pos_weight=coarse_pos_weight,
            local_pos_weight=local_pos_weight,
            anchor_pos_weight=anchor_pos_weight,
            seed=router_seed,
        )
        return cls(
            model=model,
            processor=processor,
            device=resolved_device,
            router=router,
            text_conditioning_mode=text_conditioning_mode,
            fastv_prune_layer=fastv_prune_layer,
            model_type=model_type,
        )

    @staticmethod
    def format_multiple_choice_prompt(instruction: str, choices: Sequence[str]) -> str:
        lines = [
            "You are given an image and a multiple-choice question.",
            instruction.strip(),
            "Choices:",
        ]
        for label, choice in zip(choice_labels(len(choices)), choices):
            lines.append(f"({label}) {choice}")
        lines.append("Reply with only the letter of the correct choice.")
        return "\n".join(lines)

    @staticmethod
    def format_open_vqa_prompt(question: str, answer_instruction: str = "Answer briefly.") -> str:
        return "\n".join(
            [
                "You are given an image and a question.",
                question.strip(),
                answer_instruction.strip(),
            ]
        )

    @classmethod
    def format_yes_no_prompt(cls, question: str) -> str:
        return cls.format_open_vqa_prompt(
            question=question,
            answer_instruction="Answer with a single word: yes or no.",
        )

    @classmethod
    def format_short_answer_prompt(cls, question: str) -> str:
        return cls.format_open_vqa_prompt(
            question=question,
            answer_instruction="Answer with a short phrase or number only.",
        )

    @staticmethod
    def _normalize_question_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    @staticmethod
    def _contains_any_keyword(text: str, keywords: Sequence[str]) -> bool:
        for keyword in keywords:
            if " " in keyword:
                if keyword in text:
                    return True
            else:
                if re.search(rf"\b{re.escape(keyword)}\b", text):
                    return True
        return False

    def _infer_special_cue(self, question: str) -> Optional[str]:
        lowered = question.lower()
        if self._contains_any_keyword(lowered, self._READING_KEYWORDS):
            return "Read text relevant to the question."
        if self._contains_any_keyword(lowered, self._CHART_KEYWORDS):
            return "Focus on chart or diagram structures relevant to the question."
        if self._contains_any_keyword(lowered, self._SPATIAL_KEYWORDS):
            return "Focus on regions needed for spatial or relational judgment."
        if self._contains_any_keyword(lowered, self._COUNT_COMPARE_KEYWORDS):
            return "Preserve regions needed for counting or comparison."
        return None

    @staticmethod
    def _looks_like_verification_question(question: str) -> bool:
        lowered = question.lower().strip()
        lowered = re.sub(r"\s+", " ", lowered)
        if re.match(r"^(is|are|was|were|do|does|did|can|could|has|have|had|should|would)\\b", lowered):
            return True
        verification_patterns = (
            "is there",
            "are there",
            "whether",
            "true or false",
            "correct or incorrect",
            "yes or no",
            "does the image",
            "in the image?",
        )
        return any(pattern in lowered for pattern in verification_patterns)

    def _build_verification_cues(self, question: str) -> list[str]:
        if not self._looks_like_verification_question(question):
            return []
        return [
            f"Find visual evidence supporting this statement: {question}",
            f"Find visual evidence contradicting this statement: {question}",
            "Preserve regions needed to verify existence, attributes, and spatial relations.",
        ]

    def _extract_target_phrase(self, question: str) -> str:
        lowered = question.lower().strip().rstrip("?.!")
        patterns = (
            r"what does (.+?) say$",
            r"what is written on (.+?)$",
            r"what is written in (.+?)$",
            r"is there (.+?)$",
            r"are there (.+?)$",
            r"where is (.+?)$",
            r"what color is (.+?)$",
            r"how many (.+?) (?:are|is|can|do|does|in|on|at|with)\b",
            r"how many (.+?)$",
            r"which (.+)$",
            r"is (.+)$",
            r"are (.+)$",
        )
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if not match:
                continue
            phrase = match.group(1).strip()
            phrase = re.sub(r"^(the|a|an)\s+", "", phrase)
            phrase = re.sub(r"\s+", " ", phrase).strip()
            if phrase:
                return phrase

        tokens = re.findall(r"[a-z0-9']+", lowered)
        filtered = [token for token in tokens if token not in self._TARGET_STOPWORDS]
        if filtered:
            return " ".join(filtered[:8])
        return "key visual evidence"

    def _build_universal_three_cues(self, instruction: str) -> tuple[str, list[str]]:
        question = self._normalize_question_text(instruction)
        global_cue_text = f"Answer the question about the image: {question}"
        target_phrase = self._extract_target_phrase(question)
        target_cue_text = f"Find the region relevant to: {target_phrase}."
        cue_bank = [target_cue_text]
        special_cue_text = self._infer_special_cue(question)
        if special_cue_text is not None:
            cue_bank.append(special_cue_text)
        cue_bank.extend(self._build_verification_cues(question))
        return global_cue_text, cue_bank

    def _build_choice_aware_universal_cues(
        self,
        instruction: str,
        choices: Sequence[str],
    ) -> tuple[str, list[str], list[str]]:
        question = self._normalize_question_text(instruction)
        labels = choice_labels(len(choices))
        choice_lines = [f"({label}) {choice.strip()}" for label, choice in zip(labels, choices)]
        global_cue_text = (
            "Answer the multiple-choice question about the image.\n"
            f"Question: {question}\n"
            "Candidate answers:\n"
            + "\n".join(choice_lines)
        )

        target_phrase = self._extract_target_phrase(question)
        hypothesis_texts = [f"Find the region relevant to: {target_phrase}."]
        special_cue_text = self._infer_special_cue(question)
        if special_cue_text is not None:
            hypothesis_texts.append(special_cue_text)

        option_focus_texts = [
            (
                f"Check visual evidence for option ({label}). "
                f"Keep details that confirm or reject: {choice.strip()}."
            )
            for label, choice in zip(labels, choices)
        ]
        hypothesis_texts.extend(option_focus_texts)

        contrast_texts = [
            (
                f"Distinguish option ({label}) {choice.strip()} "
                "from the other candidate answers using local visual evidence."
            )
            for label, choice in zip(labels, choices)
        ]
        return global_cue_text, hypothesis_texts, contrast_texts

    def _load_image(self, image_source: Any) -> Image.Image:
        if isinstance(image_source, Image.Image):
            return image_source.convert("RGB")
        if isinstance(image_source, (str, Path)):
            return Image.open(image_source).convert("RGB")
        raise TypeError(f"Unsupported image source type: {type(image_source)}")

    def prepare_inputs(
        self,
        image_source: Any,
        prompt_text: str,
        system_prompt: Optional[str] = None,
    ) -> dict[str, torch.Tensor]:
        image = self._load_image(image_source)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        )
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        return {key: value.to(self.device) for key, value in inputs.items()}

    def _compute_prompt_odor_cue(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        with torch.inference_mode():
            text_embeddings = self.model.get_input_embeddings()(input_ids)
        text_mask = (mm_token_type_ids == 0) & attention_mask.bool()
        text_mask = text_mask.unsqueeze(-1)
        pooled = (text_embeddings * text_mask).sum(dim=1)
        denom = text_mask.sum(dim=1).clamp_min(1)
        return pooled / denom

    def _pool_text_embeddings(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("texts must not be empty")
        encoded = self.processor.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        with torch.inference_mode():
            text_embeddings = self.model.get_input_embeddings()(input_ids)
        text_mask = attention_mask.unsqueeze(-1).to(text_embeddings.dtype)
        pooled = (text_embeddings * text_mask).sum(dim=1)
        denom = text_mask.sum(dim=1).clamp_min(1.0)
        return pooled / denom

    def _build_text_conditioning(
        self,
        instruction: str,
        choices: Sequence[str],
        inputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        prompt_cue = self._compute_prompt_odor_cue(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            mm_token_type_ids=inputs["mm_token_type_ids"],
        )
        if self.text_conditioning_mode == "pooled_prompt":
            return prompt_cue, None, None
        if self.text_conditioning_mode in {"universal_three_cue", "universal_vqa"}:
            if len(choices) > 0:
                global_text, cue_bank_texts, contrast_texts = self._build_choice_aware_universal_cues(
                    instruction=instruction,
                    choices=choices,
                )
            else:
                global_text, cue_bank_texts = self._build_universal_three_cues(instruction=instruction)
                contrast_texts = []
            global_cue = self._pool_text_embeddings([global_text])
            query_cue = 0.5 * (prompt_cue + global_cue)
            hypothesis_cues = self._pool_text_embeddings(cue_bank_texts).unsqueeze(0)
            contrast_cues = None
            if contrast_texts:
                contrast_cues = self._pool_text_embeddings(contrast_texts).unsqueeze(0)
            elif hypothesis_cues.size(1) > 1:
                cue_sum = hypothesis_cues.sum(dim=1, keepdim=True)
                others_mean = (cue_sum - hypothesis_cues) / float(hypothesis_cues.size(1) - 1)
                contrast_cues = hypothesis_cues - others_mean
            return query_cue, hypothesis_cues, contrast_cues
        if self.text_conditioning_mode != "multi_cue":
            raise ValueError(f"Unsupported text_conditioning_mode: {self.text_conditioning_mode}")

        instruction_text = f"Task: {instruction.strip()}"
        instruction_cue = self._pool_text_embeddings([instruction_text])
        query_cue = 0.5 * (prompt_cue + instruction_cue)
        if len(choices) == 0:
            return query_cue, None, None

        labels = choice_labels(len(choices))
        hypothesis_texts = [
            (
                f"Task: {instruction.strip()}\n"
                f"Hypothesis: the correct answer is ({label}) {choice.strip()}."
            )
            for label, choice in zip(labels, choices)
        ]
        hypothesis_cues = self._pool_text_embeddings(hypothesis_texts).unsqueeze(0)

        contrast_cues = None
        if hypothesis_cues.size(1) > 1:
            cue_sum = hypothesis_cues.sum(dim=1, keepdim=True)
            others_mean = (cue_sum - hypothesis_cues) / float(hypothesis_cues.size(1) - 1)
            contrast_cues = hypothesis_cues - others_mean

        return query_cue, hypothesis_cues, contrast_cues

    def _get_single_image_tokens(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        with torch.inference_mode():
            image_features = self.model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        if len(image_features.pooler_output) != 1:
            raise NotImplementedError("The current wrapper supports one image per sample")
        visual_tokens = image_features.pooler_output[0]
        return visual_tokens, self._grid_size_from_image_grid(image_grid_thw)

    def _grid_size_from_image_grid(self, image_grid_thw: torch.Tensor) -> tuple[int, int]:
        spatial_merge = self.model.model.visual.spatial_merge_size
        grid_h = int(image_grid_thw[0, 1].item() // spatial_merge)
        grid_w = int(image_grid_thw[0, 2].item() // spatial_merge)
        return grid_h, grid_w

    def _count_visual_tokens(self, image_grid_thw: torch.Tensor) -> int:
        grid_h, grid_w = self._grid_size_from_image_grid(image_grid_thw)
        return int(grid_h * grid_w)

    def _kv_cache_bytes_for_sequence_length(self, seq_len: int) -> int:
        text_config = self.model.config.text_config
        num_layers = int(text_config.num_hidden_layers)
        num_kv_heads = int(text_config.num_key_value_heads)
        head_dim = getattr(text_config, "head_dim", None)
        if head_dim is None:
            head_dim = int(text_config.hidden_size // text_config.num_attention_heads)
        dtype_size = torch.tensor([], dtype=self.model.dtype).element_size()
        return int(2 * num_layers * num_kv_heads * head_dim * seq_len * dtype_size)

    def _build_efficiency_stats(
        self,
        *,
        full_visual_tokens: int,
        selected_visual_tokens: int,
        prefill_seq_len: int,
        generated_tokens: int,
    ) -> dict[str, Any]:
        final_seq_len = int(prefill_seq_len + generated_tokens)
        return {
            "full_visual_tokens": int(full_visual_tokens),
            "selected_visual_tokens": int(selected_visual_tokens),
            "prefill_seq_len": int(prefill_seq_len),
            "generated_tokens": int(generated_tokens),
            "final_seq_len": final_seq_len,
            "prefill_kv_cache_mb_estimate": self._kv_cache_bytes_for_sequence_length(prefill_seq_len) / (1024**2),
            "kv_cache_mb_estimate": self._kv_cache_bytes_for_sequence_length(final_seq_len) / (1024**2),
        }

    def _build_position_ids(
        self,
        prefix_len: int,
        selected_indices: torch.Tensor,
        grid_size: tuple[int, int],
        suffix_len: int,
    ) -> torch.Tensor:
        device = selected_indices.device
        grid_h, grid_w = grid_size
        rows = selected_indices // grid_w
        cols = selected_indices % grid_w

        prefix_positions = torch.arange(prefix_len, device=device, dtype=torch.long).view(1, -1).expand(3, -1)
        image_start = prefix_len
        image_positions = torch.stack(
            [
                torch.full_like(rows, image_start),
                image_start + rows,
                image_start + cols,
            ],
            dim=0,
        )
        suffix_start = prefix_len + max(grid_h, grid_w)
        suffix_positions = (
            torch.arange(suffix_len, device=device, dtype=torch.long).view(1, -1).expand(3, -1) + suffix_start
        )
        return torch.cat([prefix_positions, image_positions, suffix_positions], dim=1).unsqueeze(1)

    def _locate_image_block(self, mm_token_type_ids: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        image_positions = torch.nonzero(mm_token_type_ids[0] == 1, as_tuple=False).squeeze(-1)
        if image_positions.numel() == 0:
            raise ValueError("No image placeholder tokens were found in the prompt")
        if int(image_positions[-1] - image_positions[0] + 1) != image_positions.numel():
            raise NotImplementedError("The current wrapper expects the image tokens to form a single contiguous block")
        prefix_len = int(image_positions[0].item())
        suffix_start = int(image_positions[-1].item()) + 1
        return image_positions, prefix_len, suffix_start

    def _assemble_prefill_sequence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        selected_indices: torch.Tensor,
        selected_tokens: torch.Tensor,
        grid_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        image_positions, prefix_len, suffix_start = self._locate_image_block(mm_token_type_ids)
        suffix_len = int(input_ids.size(1) - suffix_start)
        kept_count = int(selected_indices.numel())

        image_token_id = self.model.config.image_token_id
        routed_input_ids = torch.cat(
            [
                input_ids[:, :prefix_len],
                torch.full((1, kept_count), image_token_id, device=self.device, dtype=input_ids.dtype),
                input_ids[:, suffix_start:],
            ],
            dim=1,
        )
        routed_mm_types = torch.cat(
            [
                mm_token_type_ids[:, :prefix_len],
                torch.ones((1, kept_count), device=self.device, dtype=mm_token_type_ids.dtype),
                mm_token_type_ids[:, suffix_start:],
            ],
            dim=1,
        )
        routed_attention = torch.cat(
            [
                attention_mask[:, :prefix_len],
                torch.ones((1, kept_count), device=self.device, dtype=attention_mask.dtype),
                attention_mask[:, suffix_start:],
            ],
            dim=1,
        )
        routed_embeds = self.model.get_input_embeddings()(routed_input_ids)
        routed_embeds[:, prefix_len : prefix_len + kept_count, :] = selected_tokens.unsqueeze(0).to(routed_embeds.dtype)
        position_ids = self._build_position_ids(
            prefix_len=prefix_len,
            selected_indices=selected_indices,
            grid_size=grid_size,
            suffix_len=suffix_len,
        )
        return {
            "input_ids": routed_input_ids,
            "attention_mask": routed_attention,
            "mm_token_type_ids": routed_mm_types,
            "position_ids": position_ids,
            "inputs_embeds": routed_embeds,
            "image_positions": torch.arange(prefix_len, prefix_len + kept_count, device=self.device, dtype=torch.long),
            "prefix_len": torch.tensor(prefix_len, device=self.device, dtype=torch.long),
            "suffix_start": torch.tensor(suffix_start, device=self.device, dtype=torch.long),
        }

    def _build_full_visual_prefill(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, tuple[int, int]]:
        visual_tokens, grid_size = self._get_single_image_tokens(
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
        )
        all_indices = torch.arange(visual_tokens.size(0), device=self.device, dtype=torch.long)
        sequence = self._assemble_prefill_sequence(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            mm_token_type_ids=inputs["mm_token_type_ids"],
            selected_indices=all_indices,
            selected_tokens=visual_tokens.to(self.model.dtype),
            grid_size=grid_size,
        )
        return sequence, visual_tokens, grid_size

    def _visionzip_to_routing(self, result: VisionZipResult) -> RoutingResult:
        return RoutingResult(
            selected_indices=result.selected_indices,
            selected_scores=result.selected_scores,
            base_scores=result.base_scores,
            route_scores=result.base_scores,
            query_scores=result.base_scores,
            hypothesis_scores=None,
            contrast_scores=None,
            agreement_scores=None,
            repulsion_scores=None,
            uncertainty_scores=None,
            region_agreement_scores=None,
            odor_head_gate=torch.zeros(1, device=result.base_scores.device, dtype=torch.float32),
            head_logits=torch.zeros((result.full_visual_tokens, 1), device=result.base_scores.device, dtype=torch.float32),
            kept_count=result.selected_visual_tokens,
            grid_size=result.grid_size,
            scaffold_indices=result.dominant_indices,
            exploit_indices=result.contextual_indices,
            jump_indices=torch.empty(0, device=result.base_scores.device, dtype=torch.long),
            fill_indices=torch.empty(0, device=result.base_scores.device, dtype=torch.long),
            active_window_indices=torch.empty(0, device=result.base_scores.device, dtype=torch.long),
            context_window_indices=result.contextual_indices,
            anchor_window_indices=torch.empty(0, device=result.base_scores.device, dtype=torch.long),
            ivc_indices=torch.empty(0, device=result.base_scores.device, dtype=torch.long),
        )

    def _build_visionzip_prefill(
        self,
        inputs: dict[str, torch.Tensor],
        final_keep: Optional[int] = None,
        keep_ratio: Optional[float] = None,
    ) -> RoutedPrefill:
        if final_keep is not None and final_keep > 0:
            visual_tokens, grid_size = self._get_single_image_tokens(
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
            )
            resolved_keep_ratio = float(final_keep) / max(1, int(visual_tokens.size(0)))
            zip_result = zip_projected_tokens(
                visual_tokens=visual_tokens,
                grid_size=grid_size,
                keep_ratio=resolved_keep_ratio,
                config=VisionZipConfig(keep_ratio=resolved_keep_ratio),
            )
        elif self.model_type in {"qwen3_vl", "qwen3_vl_moe"}:
            resolved_keep_ratio = self.router.default_keep_ratio if keep_ratio is None else keep_ratio
            zip_result = qwen3_visionzip_projected_tokens(
                model=self.model,
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
                keep_ratio=resolved_keep_ratio,
                config=VisionZipConfig(keep_ratio=resolved_keep_ratio),
            )
        else:
            visual_tokens, grid_size = self._get_single_image_tokens(
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
            )
            zip_result = zip_projected_tokens(
                visual_tokens=visual_tokens,
                grid_size=grid_size,
                keep_ratio=keep_ratio,
                config=VisionZipConfig(keep_ratio=self.router.default_keep_ratio),
            )

        routing = self._visionzip_to_routing(zip_result)
        sequence = self._assemble_prefill_sequence(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            mm_token_type_ids=inputs["mm_token_type_ids"],
            selected_indices=routing.selected_indices.to(self.device),
            selected_tokens=zip_result.zipped_tokens.to(self.model.dtype),
            grid_size=zip_result.grid_size,
        )
        return RoutedPrefill(
            input_ids=sequence["input_ids"],
            attention_mask=sequence["attention_mask"],
            mm_token_type_ids=sequence["mm_token_type_ids"],
            position_ids=sequence["position_ids"],
            inputs_embeds=sequence["inputs_embeds"],
            routing=routing,
            full_visual_token_count=zip_result.full_visual_tokens,
            prefill_seq_len=int(sequence["attention_mask"].size(1)),
        )

    def _build_causal_mask_mapping(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], Optional[torch.Tensor]]:
        causal_mask_fn = create_causal_mask
        sliding_mask_fn = create_sliding_window_causal_mask
        if self.model_type == "qwen3_vl":
            from transformers.models.qwen3_vl.modeling_qwen3_vl import create_causal_mask as causal_mask_fn

            sliding_mask_fn = None
        elif self.model_type == "qwen3_vl_moe":
            from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import create_causal_mask as causal_mask_fn

            sliding_mask_fn = None

        if causal_mask_fn is None:
            raise ImportError(
                "FastV routing requires a transformers version that exposes Qwen-VL causal mask helpers."
            )
        text_model = self.model.model.language_model
        text_position_ids = None
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
        mask_kwargs = {
            "config": text_model.config,
            "inputs_embeds": hidden_states,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "position_ids": text_position_ids,
        }
        causal_mask_mapping = {
            "full_attention": causal_mask_fn(**mask_kwargs),
        }
        if getattr(text_model, "has_sliding_layers", False) and sliding_mask_fn is not None:
            causal_mask_mapping["sliding_attention"] = sliding_mask_fn(**mask_kwargs)
        if "sliding_attention" not in causal_mask_mapping:
            causal_mask_mapping["sliding_attention"] = causal_mask_mapping["full_attention"]
        return causal_mask_mapping, text_position_ids

    def _fastv_image_attention_scores(
        self,
        layer_attention: torch.Tensor,
        image_positions: torch.Tensor,
    ) -> torch.Tensor:
        # FastV ranks image tokens by the average attention score they receive at layer K.
        if layer_attention.dim() != 4:
            raise ValueError("Expected attention tensor with shape [B, H, Q, K]")
        mean_attention = layer_attention.float().mean(dim=1)[0]
        return mean_attention.index_select(1, image_positions).mean(dim=0)

    @staticmethod
    def _repeat_kv_for_fastv(hidden_states: torch.Tensor, num_repeats: int) -> torch.Tensor:
        if num_repeats == 1:
            return hidden_states
        batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch,
            num_key_value_heads,
            num_repeats,
            seq_len,
            head_dim,
        )
        return hidden_states.reshape(batch, num_key_value_heads * num_repeats, seq_len, head_dim)

    def _apply_fastv_rotary(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model_type == "qwen2_5_vl":
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb

            return apply_multimodal_rotary_pos_emb(
                query_states,
                key_states,
                position_embeddings[0],
                position_embeddings[1],
                attn_module.config.rope_parameters["mrope_section"],
            )
        if self.model_type == "qwen3_vl":
            from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
        elif self.model_type == "qwen3_vl_moe":
            from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import apply_rotary_pos_emb
        else:
            raise NotImplementedError(f"FastV rotary is not wired for model_type={self.model_type}")
        return apply_rotary_pos_emb(query_states, key_states, position_embeddings[0], position_embeddings[1])

    def _compute_fastv_received_attention_scores(
        self,
        decoder_layer,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        image_positions: torch.Tensor,
        query_chunk_size: int = 128,
    ) -> torch.Tensor:
        """Compute FastV image-token received attention without materializing QxK."""
        attn_module = decoder_layer.self_attn
        normed_states = decoder_layer.input_layernorm(hidden_states)
        batch_size, seq_len, _ = normed_states.shape
        hidden_shape = (batch_size, seq_len, -1, attn_module.head_dim)

        query_states = attn_module.q_proj(normed_states).view(hidden_shape).transpose(1, 2)
        key_states = attn_module.k_proj(normed_states).view(hidden_shape).transpose(1, 2)
        if hasattr(attn_module, "q_norm"):
            query_states = attn_module.q_norm(query_states)
        if hasattr(attn_module, "k_norm"):
            key_states = attn_module.k_norm(key_states)
        query_states, key_states = self._apply_fastv_rotary(
            query_states=query_states,
            key_states=key_states,
            position_embeddings=position_embeddings,
            attn_module=attn_module,
        )
        key_states = self._repeat_kv_for_fastv(key_states, attn_module.num_key_value_groups)
        key_states_t = key_states.transpose(2, 3)
        scaling = getattr(attn_module, "scaling", attn_module.head_dim**-0.5)

        received = torch.zeros(image_positions.numel(), device=hidden_states.device, dtype=torch.float32)
        denom = 0
        for start in range(0, seq_len, query_chunk_size):
            end = min(seq_len, start + query_chunk_size)
            attn_weights = torch.matmul(query_states[:, :, start:end, :], key_states_t) * scaling
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask[:, :, start:end, :]
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
            received += attn_weights.index_select(3, image_positions).sum(dim=(0, 1, 2))
            denom += batch_size * query_states.size(1) * (end - start)
        return received / max(1, denom)

    def _build_fastv_routing(
        self,
        selected_indices: torch.Tensor,
        attention_scores: torch.Tensor,
        grid_size: tuple[int, int],
    ) -> RoutingResult:
        selected_indices = selected_indices.sort().values
        selected_scores = attention_scores.index_select(0, selected_indices)
        return RoutingResult(
            selected_indices=selected_indices,
            selected_scores=selected_scores,
            base_scores=attention_scores,
            route_scores=attention_scores,
            query_scores=attention_scores,
            hypothesis_scores=None,
            contrast_scores=None,
            agreement_scores=None,
            repulsion_scores=None,
            uncertainty_scores=None,
            region_agreement_scores=None,
            odor_head_gate=torch.zeros(1, device=attention_scores.device, dtype=torch.float32),
            head_logits=torch.zeros((attention_scores.numel(), 1), device=attention_scores.device, dtype=torch.float32),
            kept_count=int(selected_indices.numel()),
            grid_size=grid_size,
        )

    @staticmethod
    def _layer_attention_type(text_model, layer_idx: int) -> str:
        layer_types = getattr(text_model.config, "layer_types", None)
        if layer_types is None:
            return "full_attention"
        return layer_types[layer_idx - 1]

    def _forward_text_layer_with_optional_attention(
        self,
        decoder_layer,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        text_position_ids: Optional[torch.Tensor],
        need_attention: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not need_attention:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
            )
            if isinstance(layer_outputs, tuple):
                return layer_outputs[0], layer_outputs[1] if len(layer_outputs) > 1 else None
            return layer_outputs, None

        residual = hidden_states
        hidden_states = decoder_layer.input_layernorm(hidden_states)
        attn_output, attn_weights = decoder_layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            position_ids=text_position_ids,
            past_key_values=None,
            use_cache=False,
            output_attentions=need_attention,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        mlp_output = decoder_layer.mlp(hidden_states)
        if isinstance(mlp_output, tuple):
            mlp_output = mlp_output[0]
        hidden_states = residual + mlp_output
        return hidden_states, attn_weights

    def _build_routed_prefill(
        self,
        inputs: dict[str, torch.Tensor],
        instruction: str,
        choices: Sequence[str],
        final_keep: Optional[int] = None,
        keep_ratio: Optional[float] = None,
    ) -> RoutedPrefill:
        if self.router.routing_mode == "visionzip":
            return self._build_visionzip_prefill(
                inputs=inputs,
                final_keep=final_keep,
                keep_ratio=keep_ratio,
            )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        mm_token_type_ids = inputs["mm_token_type_ids"]
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]
        if input_ids.size(0) != 1:
            raise NotImplementedError("The current wrapper supports batch size 1")

        self._trace("build_routed_prefill: text_conditioning start")
        odor_cue, hypothesis_cues, contrast_cues = self._build_text_conditioning(
            instruction=instruction,
            choices=choices,
            inputs=inputs,
        )
        self._trace("build_routed_prefill: image_features start")
        visual_tokens, grid_size = self._get_single_image_tokens(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        self._trace(f"build_routed_prefill: router start full_visual={visual_tokens.size(0)} grid={grid_size}")
        routing = self.router(
            visual_tokens.unsqueeze(0),
            odor_cue=odor_cue,
            grid_size=grid_size,
            hypothesis_cues=hypothesis_cues,
            contrast_cues=contrast_cues,
            final_keep=final_keep,
            keep_ratio=keep_ratio,
        )

        selected_indices = routing.selected_indices.to(self.device)
        selected_tokens = visual_tokens.index_select(0, selected_indices).to(self.model.dtype)
        self._trace(f"build_routed_prefill: assemble start selected={selected_indices.numel()}")
        sequence = self._assemble_prefill_sequence(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            selected_indices=selected_indices,
            selected_tokens=selected_tokens,
            grid_size=grid_size,
        )
        self._trace(f"build_routed_prefill: done prefill_seq_len={sequence['attention_mask'].size(1)}")
        return RoutedPrefill(
            input_ids=sequence["input_ids"],
            attention_mask=sequence["attention_mask"],
            mm_token_type_ids=sequence["mm_token_type_ids"],
            position_ids=sequence["position_ids"],
            inputs_embeds=sequence["inputs_embeds"],
            routing=routing,
            full_visual_token_count=int(grid_size[0] * grid_size[1]),
            prefill_seq_len=int(sequence["attention_mask"].size(1)),
        )

    @torch.inference_mode()
    def _forward_choice_logits_fastv(
        self,
        inputs: dict[str, torch.Tensor],
        final_keep: Optional[int] = None,
        keep_ratio: Optional[float] = None,
    ) -> dict[str, Any]:
        return self._forward_fastv_prefill_logits(
            inputs=inputs,
            final_keep=final_keep,
            keep_ratio=keep_ratio,
        )

    @torch.inference_mode()
    def _forward_fastv_prefill_logits(
        self,
        inputs: dict[str, torch.Tensor],
        final_keep: Optional[int] = None,
        keep_ratio: Optional[float] = None,
    ) -> dict[str, Any]:
        if self.model_type not in {"qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe"}:
            raise NotImplementedError(f"FastV routing is not wired for model_type={self.model_type}")
        full_sequence, visual_tokens, grid_size = self._build_full_visual_prefill(inputs)
        hidden_states = full_sequence["inputs_embeds"]
        attention_mask = full_sequence["attention_mask"]
        position_ids = full_sequence["position_ids"]
        image_positions = full_sequence["image_positions"]
        prefix_len = int(full_sequence["prefix_len"].item())
        suffix_start = int(full_sequence["suffix_start"].item())

        text_model = self.model.model.language_model
        prune_layer = min(self.fastv_prune_layer, len(text_model.layers))
        num_windows_h = (grid_size[0] + self.router.local_window_size - 1) // self.router.local_window_size
        num_windows_w = (grid_size[1] + self.router.local_window_size - 1) // self.router.local_window_size
        keep_count = self.router._resolve_keep_count(
            num_tokens=visual_tokens.size(0),
            num_windows=num_windows_h * num_windows_w,
            final_keep=final_keep,
            keep_ratio=keep_ratio,
        )

        causal_mask_mapping, text_position_ids = self._build_causal_mask_mapping(hidden_states, attention_mask, position_ids)
        position_embeddings = text_model.rotary_emb(hidden_states, position_ids)
        routing = None

        for layer_idx, decoder_layer in enumerate(text_model.layers, start=1):
            layer_type = self._layer_attention_type(text_model, layer_idx)
            layer_mask = causal_mask_mapping.get(layer_type, causal_mask_mapping["full_attention"])
            need_attention = routing is None and layer_idx == prune_layer
            image_attention = None
            if need_attention:
                image_attention = self._compute_fastv_received_attention_scores(
                    decoder_layer=decoder_layer,
                    hidden_states=hidden_states,
                    attention_mask=layer_mask,
                    position_embeddings=position_embeddings,
                    image_positions=image_positions,
                )
            hidden_states, _ = self._forward_text_layer_with_optional_attention(
                decoder_layer=decoder_layer,
                hidden_states=hidden_states,
                attention_mask=layer_mask,
                position_embeddings=position_embeddings,
                text_position_ids=text_position_ids,
                need_attention=False,
            )

            if image_attention is not None:
                topk = min(keep_count, int(image_attention.numel()))
                _, top_pos = image_attention.topk(k=topk, dim=0)
                selected_indices = top_pos.sort().values
                routing = self._build_fastv_routing(
                    selected_indices=selected_indices,
                    attention_scores=image_attention,
                    grid_size=grid_size,
                )
                prefix_positions = torch.arange(prefix_len, device=self.device, dtype=torch.long)
                selected_seq_positions = image_positions.index_select(0, routing.selected_indices)
                suffix_positions = torch.arange(suffix_start, hidden_states.size(1), device=self.device, dtype=torch.long)
                kept_positions = torch.cat([prefix_positions, selected_seq_positions, suffix_positions], dim=0)
                hidden_states = hidden_states.index_select(1, kept_positions)
                attention_mask = attention_mask.index_select(1, kept_positions)
                position_ids = position_ids.index_select(2, kept_positions)
                image_positions = selected_seq_positions
                causal_mask_mapping, text_position_ids = self._build_causal_mask_mapping(
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
                position_embeddings = text_model.rotary_emb(hidden_states, position_ids)

        hidden_states = text_model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states[:, -1, :])
        return {
            "logits": logits,
            "selected_count": None if routing is None else routing.kept_count,
            "selected_visual_tokens": int(visual_tokens.size(0)) if routing is None else int(routing.kept_count),
            "routing": routing,
            "full_visual_tokens": int(visual_tokens.size(0)),
            "prefill_seq_len": int(full_sequence["attention_mask"].size(1)),
            "final_seq_len": int(hidden_states.size(1)),
            "generated_tokens": 0,
            "prefill_kv_cache_mb_estimate": self._kv_cache_bytes_for_sequence_length(
                int(full_sequence["attention_mask"].size(1))
            )
            / (1024**2),
            "kv_cache_mb_estimate": self._kv_cache_bytes_for_sequence_length(int(hidden_states.size(1))) / (1024**2),
        }

    @torch.inference_mode()
    def forward_choice_logits(
        self,
        inputs: dict[str, torch.Tensor],
        instruction: str,
        choices: Sequence[str],
        routed: bool,
        final_keep: Optional[int] = None,
        keep_ratio: Optional[float] = None,
    ) -> dict[str, Any]:
        if not routed:
            full_visual_tokens = self._count_visual_tokens(inputs["image_grid_thw"])
            prefill_seq_len = int(inputs["attention_mask"].size(1))
            outputs = self.model(
                **inputs,
                use_cache=False,
                logits_to_keep=1,
            )
            return {
                "logits": outputs.logits[:, -1, :],
                "selected_count": None,
                "selected_visual_tokens": full_visual_tokens,
                "routing": None,
                "full_visual_tokens": full_visual_tokens,
                "prefill_seq_len": prefill_seq_len,
                "final_seq_len": prefill_seq_len,
                "generated_tokens": 0,
                "prefill_kv_cache_mb_estimate": self._kv_cache_bytes_for_sequence_length(prefill_seq_len) / (1024**2),
                "kv_cache_mb_estimate": self._kv_cache_bytes_for_sequence_length(prefill_seq_len) / (1024**2),
            }

        if self.router.routing_mode == "fastv":
            return self._forward_choice_logits_fastv(
                inputs=inputs,
                final_keep=final_keep,
                keep_ratio=keep_ratio,
            )

        routed_prefill = self._build_routed_prefill(
            inputs=inputs,
            instruction=instruction,
            choices=choices,
            final_keep=final_keep,
            keep_ratio=keep_ratio,
        )
        outputs = self.model.model.language_model(
            input_ids=None,
            attention_mask=routed_prefill.attention_mask,
            position_ids=routed_prefill.position_ids,
            inputs_embeds=routed_prefill.inputs_embeds,
            use_cache=False,
        )
        logits = self.model.lm_head(outputs.last_hidden_state[:, -1, :])
        return {
            "logits": logits,
            "selected_count": routed_prefill.routing.kept_count,
            "selected_visual_tokens": routed_prefill.routing.kept_count,
            "routing": routed_prefill.routing,
            **self._build_efficiency_stats(
                full_visual_tokens=routed_prefill.full_visual_token_count,
                selected_visual_tokens=routed_prefill.routing.kept_count,
                prefill_seq_len=routed_prefill.prefill_seq_len,
                generated_tokens=0,
            ),
        }

    def _label_token_ids(self, label: str) -> list[int]:
        variants = [label, f" {label}", f"({label})", f" ({label})"]
        token_ids = []
        for variant in variants:
            ids = self.processor.tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.append(ids[0])
        token_ids = sorted(set(token_ids))
        if not token_ids:
            raise ValueError(f"Could not find a single-token representation for choice label {label}")
        return token_ids

    @torch.inference_mode()
    def _generate_from_prefill(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        max_new_tokens: int = 16,
    ) -> tuple[list[int], str]:
        text_model = self.model.model.language_model
        self._trace(f"generate: llm_prefill start seq_len={attention_mask.size(1)} max_new_tokens={max_new_tokens}")
        outputs = text_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=True,
        )
        self._trace("generate: llm_prefill done")
        past_key_values = outputs.past_key_values
        hidden_states = outputs.last_hidden_state
        generated_ids: list[int] = []
        eos_token_id = self.processor.tokenizer.eos_token_id

        current_attention = attention_mask
        current_position_ids = position_ids
        for _ in range(max_new_tokens):
            self._trace(f"generate: decode step {len(generated_ids) + 1} start")
            logits = self.model.lm_head(hidden_states[:, -1, :])
            next_token = logits.argmax(dim=-1)
            token_id = int(next_token.item())
            if eos_token_id is not None and token_id == eos_token_id:
                break
            generated_ids.append(token_id)

            current_attention = torch.cat(
                [
                    current_attention,
                    torch.ones((1, 1), device=self.device, dtype=current_attention.dtype),
                ],
                dim=1,
            )
            next_position_ids = current_position_ids[:, :, -1:] + 1
            outputs = text_model(
                input_ids=next_token.unsqueeze(1),
                attention_mask=current_attention,
                position_ids=next_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            hidden_states = outputs.last_hidden_state
            current_position_ids = torch.cat([current_position_ids, next_position_ids], dim=-1)
            self._trace(f"generate: decode step {len(generated_ids)} done token_id={token_id}")

        generated_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        self._trace(f"generate: done text={generated_text!r}")
        return generated_ids, generated_text

    def predict_choice(
        self,
        image_source: Any,
        instruction: str,
        choices: Sequence[str],
        routed: bool = True,
        final_keep: Optional[int] = None,
        keep_ratio: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        prompt = self.format_multiple_choice_prompt(instruction=instruction, choices=choices)
        inputs = self.prepare_inputs(
            image_source=image_source,
            prompt_text=prompt,
            system_prompt=system_prompt,
        )
        outputs = self.forward_choice_logits(
            inputs=inputs,
            instruction=instruction,
            choices=choices,
            routed=routed,
            final_keep=final_keep,
            keep_ratio=keep_ratio,
        )
        logits = outputs["logits"][0]
        scores = []
        for label in choice_labels(len(choices)):
            token_ids = self._label_token_ids(label)
            scores.append(float(logits[token_ids].max().item()))
        prediction = int(max(range(len(scores)), key=lambda idx: scores[idx]))
        result = {
            "prediction": prediction,
            "scores": scores,
            "selected_count": outputs["selected_count"],
            "full_visual_tokens": outputs.get("full_visual_tokens"),
            "selected_visual_tokens": outputs.get("selected_visual_tokens", outputs["selected_count"]),
            "prefill_seq_len": outputs.get("prefill_seq_len"),
            "generated_tokens": outputs.get("generated_tokens", 0),
            "final_seq_len": outputs.get("final_seq_len"),
            "prefill_kv_cache_mb_estimate": outputs.get("prefill_kv_cache_mb_estimate"),
            "kv_cache_mb_estimate": outputs.get("kv_cache_mb_estimate"),
        }
        if outputs["routing"] is not None:
            routing = outputs["routing"]
            result["selected_indices"] = routing.selected_indices.detach().cpu().tolist()
            result["selected_scores"] = routing.selected_scores.detach().cpu().tolist()
            result["query_scores"] = routing.query_scores.detach().cpu().tolist()
            if routing.hypothesis_scores is not None:
                result["hypothesis_scores"] = routing.hypothesis_scores.detach().cpu().tolist()
            if routing.contrast_scores is not None:
                result["contrast_scores"] = routing.contrast_scores.detach().cpu().tolist()
            if routing.agreement_scores is not None:
                result["agreement_scores"] = routing.agreement_scores.detach().cpu().tolist()
            if routing.repulsion_scores is not None:
                result["repulsion_scores"] = routing.repulsion_scores.detach().cpu().tolist()
            if routing.uncertainty_scores is not None:
                result["uncertainty_scores"] = routing.uncertainty_scores.detach().cpu().tolist()
            if routing.region_agreement_scores is not None:
                result["region_agreement_scores"] = routing.region_agreement_scores.detach().cpu().tolist()
            result["grid_size"] = list(routing.grid_size)
        return result

    def generate_answer(
        self,
        image_source: Any,
        question: str,
        prompt_text: Optional[str] = None,
        routed: bool = True,
        final_keep: Optional[int] = None,
        keep_ratio: Optional[float] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 16,
    ) -> dict[str, Any]:
        prompt = prompt_text or self.format_short_answer_prompt(question)
        self._trace(f"generate_answer: prepare_inputs start question={question[:80]!r}")
        inputs = self.prepare_inputs(
            image_source=image_source,
            prompt_text=prompt,
            system_prompt=system_prompt,
        )
        self._trace("generate_answer: prepare_inputs done")

        routing = None
        selected_count = None
        if routed:
            if self.router.routing_mode == "fastv":
                fastv_outputs = self._forward_fastv_prefill_logits(
                    inputs=inputs,
                    final_keep=final_keep,
                    keep_ratio=keep_ratio,
                )
                token_ids = []
                text = ""
                if max_new_tokens > 0:
                    next_token = fastv_outputs["logits"].argmax(dim=-1)
                    token_id = int(next_token.item())
                    eos_token_id = self.processor.tokenizer.eos_token_id
                    if eos_token_id is None or token_id != eos_token_id:
                        token_ids.append(token_id)
                        text = self.processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                routing = fastv_outputs["routing"]
                selected_count = fastv_outputs["selected_visual_tokens"]
                efficiency = self._build_efficiency_stats(
                    full_visual_tokens=fastv_outputs["full_visual_tokens"],
                    selected_visual_tokens=fastv_outputs["selected_visual_tokens"],
                    prefill_seq_len=fastv_outputs["final_seq_len"],
                    generated_tokens=len(token_ids),
                )
            else:
                routed_prefill = self._build_routed_prefill(
                    inputs=inputs,
                    instruction=question,
                    choices=[],
                    final_keep=final_keep,
                    keep_ratio=keep_ratio,
                )
                token_ids, text = self._generate_from_prefill(
                    inputs_embeds=routed_prefill.inputs_embeds,
                    attention_mask=routed_prefill.attention_mask,
                    position_ids=routed_prefill.position_ids,
                    max_new_tokens=max_new_tokens,
                )
                routing = routed_prefill.routing
                selected_count = routed_prefill.routing.kept_count
                efficiency = self._build_efficiency_stats(
                    full_visual_tokens=routed_prefill.full_visual_token_count,
                    selected_visual_tokens=routed_prefill.routing.kept_count,
                    prefill_seq_len=routed_prefill.prefill_seq_len,
                    generated_tokens=len(token_ids),
                )
        else:
            full_sequence, _, _ = self._build_full_visual_prefill(inputs)
            token_ids, text = self._generate_from_prefill(
                inputs_embeds=full_sequence["inputs_embeds"],
                attention_mask=full_sequence["attention_mask"],
                position_ids=full_sequence["position_ids"],
                max_new_tokens=max_new_tokens,
            )
            full_visual_tokens = self._count_visual_tokens(inputs["image_grid_thw"])
            prefill_seq_len = int(full_sequence["attention_mask"].size(1))
            efficiency = self._build_efficiency_stats(
                full_visual_tokens=full_visual_tokens,
                selected_visual_tokens=full_visual_tokens,
                prefill_seq_len=prefill_seq_len,
                generated_tokens=len(token_ids),
            )

        result = {
            "text": text,
            "generated_token_ids": token_ids,
            "selected_count": selected_count,
            **efficiency,
        }
        if routing is not None:
            result["selected_indices"] = routing.selected_indices.detach().cpu().tolist()
            result["selected_scores"] = routing.selected_scores.detach().cpu().tolist()
            result["query_scores"] = routing.query_scores.detach().cpu().tolist()
            result["grid_size"] = list(routing.grid_size)
            if routing.hypothesis_scores is not None:
                result["hypothesis_scores"] = routing.hypothesis_scores.detach().cpu().tolist()
            if routing.contrast_scores is not None:
                result["contrast_scores"] = routing.contrast_scores.detach().cpu().tolist()
            if routing.agreement_scores is not None:
                result["agreement_scores"] = routing.agreement_scores.detach().cpu().tolist()
            if routing.repulsion_scores is not None:
                result["repulsion_scores"] = routing.repulsion_scores.detach().cpu().tolist()
            if routing.uncertainty_scores is not None:
                result["uncertainty_scores"] = routing.uncertainty_scores.detach().cpu().tolist()
            if routing.region_agreement_scores is not None:
                result["region_agreement_scores"] = routing.region_agreement_scores.detach().cpu().tolist()
        return result

    def predict_sample(
        self,
        sample: ChoiceSample,
        routed: bool = True,
        final_keep: Optional[int] = None,
        keep_ratio: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        image = sample.load_image()
        output = self.predict_choice(
            image_source=image,
            instruction=sample.instruction,
            choices=sample.choices,
            routed=routed,
            final_keep=final_keep,
            keep_ratio=keep_ratio,
            system_prompt=system_prompt,
        )
        output["sample_id"] = sample.sample_id
        output["answer_index"] = sample.answer_index
        output["correct"] = None if sample.answer_index is None else int(output["prediction"] == sample.answer_index)
        return output
