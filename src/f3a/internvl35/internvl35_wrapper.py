from __future__ import annotations

import math
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from f3a.datasets import ChoiceSample, choice_labels
from f3a.defaults import DEFAULT_ROUTER_CONFIG, DEFAULT_TEXT_CONDITIONING_MODE
from f3a.routing import OdorConditionedF3ARouter
from f3a.wrapper import F3AQwenVL

DEFAULT_INTERNVL35_8B_MODEL_PATH = os.environ.get("F3A_INTERNVL35_8B_MODEL", "OpenGVLab/InternVL3_5-8B")
DEFAULT_INTERNVL35_38B_MODEL_PATH = os.environ.get("F3A_INTERNVL35_38B_MODEL", "OpenGVLab/InternVL3_5-38B")
DEFAULT_INTERNVL35_8B_INSTRUCT_MODEL_PATH = os.environ.get(
    "F3A_INTERNVL35_8B_INSTRUCT_MODEL",
    "OpenGVLab/InternVL3_5-8B-Instruct",
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def _resolve_snapshot_path(model_path: str) -> str:
    path = Path(model_path).expanduser()
    if (path / "config.json").is_file():
        if (path / "model.safetensors.index.json").is_file() and not any(path.glob("model-*.safetensors")):
            sibling = path.with_name(path.name.replace("models--", "model--", 1))
            if (sibling / "config.json").is_file() and any(sibling.glob("model-*.safetensors")):
                return str(sibling)
        return str(path)
    ref_path = path / "refs" / "main"
    if ref_path.is_file():
        ref = ref_path.read_text(encoding="utf-8").strip()
        candidate = path / "snapshots" / ref
        if (candidate / "config.json").is_file():
            return str(candidate)
    return str(path)


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda item: item[0] * item[1],
    )
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        orig_width,
        orig_height,
        image_size,
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized = image.resize((target_width, target_height))
    processed: list[Image.Image] = []
    tiles_per_row = target_width // image_size
    for idx in range(blocks):
        box = (
            (idx % tiles_per_row) * image_size,
            (idx // tiles_per_row) * image_size,
            ((idx % tiles_per_row) + 1) * image_size,
            ((idx // tiles_per_row) + 1) * image_size,
        )
        processed.append(resized.crop(box))
    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))
    return processed


def _build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _require_internvl_dependencies() -> None:
    missing = [name for name in ("timm", "einops") if importlib.util.find_spec(name) is None]
    if missing:
        names = ", ".join(missing)
        raise ModuleNotFoundError(
            "InternVL3.5 remote code requires extra Python packages: "
            f"{names}. Install them in the `f3a` environment first, for example: "
            "`pip install timm einops`"
        )


def _load_patched_internvl_model_class(
    resolved_model_path: str,
    local_files_only: bool,
) -> type:
    model_cls = get_class_from_dynamic_module(
        "modeling_internvl_chat.InternVLChatModel",
        resolved_model_path,
        local_files_only=local_files_only,
    )
    if getattr(model_cls, "_f3a_post_init_patch", False):
        return model_cls

    original_init = model_cls.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # InternVL3.5 remote code does not call `post_init()`, but newer
        # transformers expects the attributes registered there during load.
        if not hasattr(self, "all_tied_weights_keys"):
            self.post_init()

    model_cls.__init__ = patched_init
    model_cls._f3a_post_init_patch = True
    return model_cls


class F3AInternVL35(F3AQwenVL):
    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        router: OdorConditionedF3ARouter,
        text_conditioning_mode: str = DEFAULT_TEXT_CONDITIONING_MODE,
        max_num_tiles: int = 12,
        use_thumbnail: bool = True,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.processor = SimpleNamespace(tokenizer=tokenizer)
        self.device = device
        self.router = router.to(device).eval()
        self.text_conditioning_mode = text_conditioning_mode
        self.max_num_tiles = max_num_tiles
        self.use_thumbnail = use_thumbnail
        self.input_size = int(getattr(self.model.config, "force_image_size", 448))
        self.tile_token_count = int(getattr(self.model, "num_image_token", 256))
        self.tile_side = int(round(math.sqrt(self.tile_token_count)))
        self.model_type = "internvl35"
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.transform = _build_transform(self.input_size)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cuda:0",
        torch_dtype: str = "auto",
        device_map: Optional[str] = None,
        use_flash_attn: bool = True,
        trust_remote_code: bool = True,
        local_files_only: bool = True,
        max_num_tiles: int = 12,
        use_thumbnail: bool = True,
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
        router_seed: int = DEFAULT_ROUTER_CONFIG["router_seed"],
    ) -> "F3AInternVL35":
        resolved_device = torch.device(device)
        resolved_model_path = _resolve_snapshot_path(model_path)
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
        _require_internvl_dependencies()
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_model_path,
            trust_remote_code=trust_remote_code,
            use_fast=False,
            local_files_only=local_files_only,
        )
        model_cls = _load_patched_internvl_model_class(
            resolved_model_path=resolved_model_path,
            local_files_only=local_files_only,
        )
        model = model_cls.from_pretrained(
            resolved_model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=use_flash_attn,
            local_files_only=local_files_only,
            device_map=device_map or None,
        ).eval()
        if not device_map:
            model = model.to(resolved_device)

        vision_dim = int(model.config.llm_config.hidden_size)
        odor_dim = int(model.config.llm_config.hidden_size)
        llm_heads = int(model.config.llm_config.num_attention_heads)
        ivc_rope_dim = int(getattr(model.config.llm_config, "head_dim", odor_dim // max(1, llm_heads)))
        router = OdorConditionedF3ARouter(
            token_dim=vision_dim,
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
            tokenizer=tokenizer,
            device=resolved_device,
            router=router,
            text_conditioning_mode=text_conditioning_mode,
            max_num_tiles=max_num_tiles,
            use_thumbnail=use_thumbnail,
        )

    def _load_image(self, image_source: Any) -> Image.Image:
        if isinstance(image_source, Image.Image):
            return image_source.convert("RGB")
        if isinstance(image_source, (str, Path)):
            return Image.open(image_source).convert("RGB")
        raise TypeError(f"Unsupported image source type: {type(image_source)}")

    def _prepare_pixel_values(self, image_source: Any) -> tuple[torch.Tensor, int]:
        image = self._load_image(image_source)
        tiles = _dynamic_preprocess(
            image,
            image_size=self.input_size,
            max_num=self.max_num_tiles,
            use_thumbnail=self.use_thumbnail,
        )
        pixel_values = torch.stack([self.transform(tile) for tile in tiles]).to(torch.bfloat16 if self.device.type == "cuda" else torch.float32)
        return pixel_values.to(self.device), len(tiles)

    def _pool_text_embeddings(self, texts: Sequence[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        with torch.inference_mode():
            embeds = self.model.language_model.get_input_embeddings()(input_ids)
        mask = attention_mask.unsqueeze(-1).to(embeds.dtype)
        pooled = (embeds * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return pooled / denom

    def _build_text_conditioning(
        self,
        instruction: str,
        choices: Sequence[str],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.text_conditioning_mode in {"universal_three_cue", "universal_vqa"}:
            if choices:
                global_text, cue_bank_texts, contrast_texts = self._build_choice_aware_universal_cues(
                    instruction=instruction,
                    choices=choices,
                )
            else:
                global_text, cue_bank_texts = self._build_universal_three_cues(instruction=instruction)
                contrast_texts = []
            query_cue = self._pool_text_embeddings([global_text])
            hypothesis_cues = self._pool_text_embeddings(cue_bank_texts).unsqueeze(0)
            contrast_cues = self._pool_text_embeddings(contrast_texts).unsqueeze(0) if contrast_texts else None
            if contrast_cues is None and hypothesis_cues.size(1) > 1:
                cue_sum = hypothesis_cues.sum(dim=1, keepdim=True)
                contrast_cues = hypothesis_cues - (cue_sum - hypothesis_cues) / float(hypothesis_cues.size(1) - 1)
            return query_cue, hypothesis_cues, contrast_cues

        task_text = f"Task: {instruction.strip()}"
        query_cue = self._pool_text_embeddings([task_text])
        if not choices:
            return query_cue, None, None
        labels = choice_labels(len(choices))
        hypothesis_texts = [
            f"Task: {instruction.strip()}\nHypothesis: the correct answer is ({label}) {choice.strip()}."
            for label, choice in zip(labels, choices)
        ]
        hypothesis_cues = self._pool_text_embeddings(hypothesis_texts).unsqueeze(0)
        contrast_cues = None
        if hypothesis_cues.size(1) > 1:
            cue_sum = hypothesis_cues.sum(dim=1, keepdim=True)
            contrast_cues = hypothesis_cues - (cue_sum - hypothesis_cues) / float(hypothesis_cues.size(1) - 1)
        return query_cue, hypothesis_cues, contrast_cues

    def _build_chat_query(self, question: str, image_token_count: int) -> tuple[str, int]:
        if "<image>" not in question:
            question = "<image>\n" + question
        template = self.model.conv_template.copy()
        template.system_message = self.model.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        image_tokens = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * image_token_count) + IMG_END_TOKEN
        query = query.replace("<image>", image_tokens, 1)
        eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep.strip())
        return query, eos_token_id

    def _tokenize_query(self, query: str) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(query, return_tensors="pt")
        return encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device)

    def _extract_visual_tokens(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        with torch.inference_mode():
            visual_tokens = self.model.extract_feature(pixel_values)
        num_tiles = int(visual_tokens.shape[0])
        grid_size = (self.tile_side * num_tiles, self.tile_side)
        return visual_tokens.reshape(-1, visual_tokens.shape[-1]).to(self.device), grid_size

    def _assemble_input_embeds(
        self,
        input_ids: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        with torch.inference_mode():
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
            batch, seq_len, hidden = input_embeds.shape
            flat_embeds = input_embeds.reshape(batch * seq_len, hidden)
            flat_ids = input_ids.reshape(batch * seq_len)
            selected = flat_ids == self.model.img_context_token_id
            if int(selected.sum().item()) != int(visual_features.shape[0]):
                raise ValueError(
                    f"IMG_CONTEXT token count ({int(selected.sum().item())}) does not match visual feature count "
                    f"({int(visual_features.shape[0])})"
                )
            flat_embeds[selected] = visual_features.to(flat_embeds.dtype)
            return flat_embeds.reshape(batch, seq_len, hidden)

    def _kv_cache_bytes_for_sequence_length(self, seq_len: int) -> int:
        cfg = self.model.config.llm_config
        num_layers = int(cfg.num_hidden_layers)
        num_kv_heads = int(cfg.num_key_value_heads)
        head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
        dtype_size = torch.tensor([], dtype=self.model.dtype).element_size()
        return int(2 * num_layers * num_kv_heads * head_dim * seq_len * dtype_size)

    def _build_efficiency_stats(
        self,
        full_visual_tokens: int,
        selected_visual_tokens: int,
        prefill_seq_len: int,
        generated_tokens: int,
    ) -> dict[str, Any]:
        final_seq_len = prefill_seq_len + generated_tokens
        return {
            "full_visual_tokens": int(full_visual_tokens),
            "selected_visual_tokens": int(selected_visual_tokens),
            "prefill_seq_len": int(prefill_seq_len),
            "generated_tokens": int(generated_tokens),
            "final_seq_len": int(final_seq_len),
            "prefill_kv_cache_mb_estimate": self._kv_cache_bytes_for_sequence_length(prefill_seq_len) / (1024**2),
            "kv_cache_mb_estimate": self._kv_cache_bytes_for_sequence_length(final_seq_len) / (1024**2),
        }

    def _select_visual_features(
        self,
        visual_tokens: torch.Tensor,
        grid_size: tuple[int, int],
        instruction: str,
        choices: Sequence[str],
        routed: bool,
        final_keep: Optional[int],
        keep_ratio: Optional[float],
    ) -> tuple[torch.Tensor, Optional[Any]]:
        if not routed:
            return visual_tokens, None
        odor_cue, hypothesis_cues, contrast_cues = self._build_text_conditioning(instruction, choices)
        routing = self.router(
            visual_tokens.unsqueeze(0),
            odor_cue=odor_cue,
            grid_size=grid_size,
            hypothesis_cues=hypothesis_cues,
            contrast_cues=contrast_cues,
            final_keep=final_keep,
            keep_ratio=keep_ratio,
        )
        selected = visual_tokens.index_select(0, routing.selected_indices.to(self.device))
        return selected, routing

    def forward_choice_logits(
        self,
        image_source: Any,
        instruction: str,
        choices: Sequence[str],
        routed: bool,
        final_keep: Optional[int] = None,
        keep_ratio: Optional[float] = None,
    ) -> dict[str, Any]:
        pixel_values, _ = self._prepare_pixel_values(image_source)
        visual_tokens, grid_size = self._extract_visual_tokens(pixel_values)
        full_visual_tokens = int(visual_tokens.shape[0])
        selected_visual_tokens, routing = self._select_visual_features(
            visual_tokens=visual_tokens,
            grid_size=grid_size,
            instruction=instruction,
            choices=choices,
            routed=routed,
            final_keep=final_keep,
            keep_ratio=keep_ratio,
        )
        prompt = self.format_multiple_choice_prompt(instruction=instruction, choices=choices)
        query, _ = self._build_chat_query(prompt, int(selected_visual_tokens.shape[0]))
        input_ids, attention_mask = self._tokenize_query(query)
        input_embeds = self._assemble_input_embeds(input_ids, selected_visual_tokens)
        with torch.inference_mode():
            outputs = self.model.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            if getattr(outputs, "logits", None) is not None:
                logits = outputs.logits[:, -1, :]
            else:
                logits = self.model.lm_head(outputs.last_hidden_state[:, -1, :])
        return {
            "logits": logits,
            "routing": routing,
            "selected_count": None if routing is None else routing.kept_count,
            **self._build_efficiency_stats(
                full_visual_tokens=full_visual_tokens,
                selected_visual_tokens=int(selected_visual_tokens.shape[0]),
                prefill_seq_len=int(input_ids.shape[1]),
                generated_tokens=0,
            ),
        }

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
        del system_prompt
        outputs = self.forward_choice_logits(
            image_source=image_source,
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
            "full_visual_tokens": outputs["full_visual_tokens"],
            "selected_visual_tokens": outputs["selected_visual_tokens"],
            "prefill_seq_len": outputs["prefill_seq_len"],
            "generated_tokens": outputs["generated_tokens"],
            "final_seq_len": outputs["final_seq_len"],
            "prefill_kv_cache_mb_estimate": outputs["prefill_kv_cache_mb_estimate"],
            "kv_cache_mb_estimate": outputs["kv_cache_mb_estimate"],
        }
        if outputs["routing"] is not None:
            routing = outputs["routing"]
            result["selected_indices"] = routing.selected_indices.detach().cpu().tolist()
            result["selected_scores"] = routing.selected_scores.detach().cpu().tolist()
            result["query_scores"] = routing.query_scores.detach().cpu().tolist()
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
        del system_prompt
        pixel_values, _ = self._prepare_pixel_values(image_source)
        visual_tokens, grid_size = self._extract_visual_tokens(pixel_values)
        full_visual_tokens = int(visual_tokens.shape[0])
        selected_visual_tokens, routing = self._select_visual_features(
            visual_tokens=visual_tokens,
            grid_size=grid_size,
            instruction=question,
            choices=[],
            routed=routed,
            final_keep=final_keep,
            keep_ratio=keep_ratio,
        )
        query, eos_token_id = self._build_chat_query(prompt_text or self.format_short_answer_prompt(question), int(selected_visual_tokens.shape[0]))
        input_ids, attention_mask = self._tokenize_query(query)
        generation_output = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=selected_visual_tokens,
            generation_config=None,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_token_id,
        )
        token_ids = generation_output[0].detach().cpu().tolist()
        text = self.tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        text = text.split(self.model.conv_template.sep.strip())[0].strip()
        generated_tokens = max(0, int(generation_output.shape[1] - input_ids.shape[1]))
        result = {
            "text": text,
            "generated_token_ids": token_ids,
            "selected_count": None if routing is None else routing.kept_count,
            **self._build_efficiency_stats(
                full_visual_tokens=full_visual_tokens,
                selected_visual_tokens=int(selected_visual_tokens.shape[0]),
                prefill_seq_len=int(input_ids.shape[1]),
                generated_tokens=generated_tokens,
            ),
        }
        if routing is not None:
            result["selected_indices"] = routing.selected_indices.detach().cpu().tolist()
            result["selected_scores"] = routing.selected_scores.detach().cpu().tolist()
            result["query_scores"] = routing.query_scores.detach().cpu().tolist()
            result["grid_size"] = list(routing.grid_size)
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


def build_internvl35_model_from_args(args) -> F3AInternVL35:
    return F3AInternVL35.from_pretrained(
        model_path=args.model_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        device_map=getattr(args, "device_map", "") or None,
        use_flash_attn=not getattr(args, "disable_flash_attn", False),
        max_num_tiles=getattr(args, "max_num_tiles", 12),
        use_thumbnail=not getattr(args, "disable_thumbnail", False),
        routing_mode=args.routing_mode,
        text_conditioning_mode=args.text_conditioning_mode,
        keep_ratio=args.keep_ratio,
        router_heads=args.router_heads,
        token_nonzero=args.token_nonzero,
        odor_nonzero=args.odor_nonzero,
        head_topk=args.head_topk,
        odor_topk=args.odor_topk,
        local_window_size=args.local_window_size,
        scaffold_keep=args.scaffold_keep,
        smell_weight=args.smell_weight,
        odor_gate_scale=args.odor_gate_scale,
        odor_temperature=args.odor_temperature,
        hypothesis_weight=args.hypothesis_weight,
        contrast_weight=args.contrast_weight,
        agreement_weight=args.agreement_weight,
        repulsion_weight=args.repulsion_weight,
        region_agreement_weight=args.region_agreement_weight,
        lockon_weight=args.lockon_weight,
        visit_weight=args.visit_weight,
        uncertainty_weight=args.uncertainty_weight,
        jump_ratio=args.jump_ratio,
        ivc_keep_ratio=args.ivc_keep_ratio,
        ivc_window_bonus=args.ivc_window_bonus,
        coarse_pos_weight=args.coarse_pos_weight,
        local_pos_weight=args.local_pos_weight,
        anchor_pos_weight=args.anchor_pos_weight,
        router_seed=args.router_seed,
    )
