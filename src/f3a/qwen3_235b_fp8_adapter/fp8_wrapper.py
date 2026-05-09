from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoProcessor

from f3a.defaults import DEFAULT_ROUTER_CONFIG, DEFAULT_TEXT_CONDITIONING_MODE
from f3a.routing import OdorConditionedF3ARouter
from f3a.wrapper import F3AQwenVL, _resolve_qwen_vl_model_class

DEFAULT_FP8_MODEL_PATH = os.environ.get(
    "F3A_QWEN3_235B_FP8_MODEL",
    "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
)


def resolve_snapshot_path(model_path: str | os.PathLike[str]) -> str:
    """Resolve a Hugging Face cache directory to its concrete snapshot path."""
    path = Path(model_path).expanduser()
    if (path / "config.json").is_file():
        return str(path)
    snapshots = path / "snapshots"
    if not snapshots.is_dir():
        return str(path)

    ref_path = path / "refs" / "main"
    if ref_path.is_file():
        ref = ref_path.read_text(encoding="utf-8").strip()
        candidate = snapshots / ref
        if (candidate / "config.json").is_file():
            return str(candidate)

    candidates = [item for item in snapshots.iterdir() if (item / "config.json").is_file()]
    if not candidates:
        return str(path)
    return str(max(candidates, key=lambda item: item.stat().st_mtime))


def _parse_dtype(torch_dtype: str, device: torch.device):
    if torch_dtype == "auto":
        # Qwen3-VL-235B-FP8 keeps unquantized modules in bf16 in its config.
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return getattr(torch, torch_dtype)


def _first_parameter_device(module, fallback: torch.device) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return fallback
    except Exception:
        return fallback


def _parse_max_memory(raw: str) -> Optional[dict[Any, str]]:
    """Parse strings like '0:135GiB,1:135GiB,cpu:512GiB' for from_pretrained."""
    raw = raw.strip()
    if not raw:
        return None
    parsed: dict[Any, str] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        key, sep, value = item.partition(":")
        if not sep or not value:
            raise ValueError(f"Invalid max_memory item: {item!r}")
        key = key.strip()
        parsed[int(key) if key.isdigit() else key] = value.strip()
    return parsed


class F3AQwenVL235BFP8(F3AQwenVL):
    """F3A wrapper tuned for local Qwen3-VL-235B-A22B-Instruct-FP8.

    This subclass leaves the original F3A code untouched.  It only changes
    model loading defaults and makes image/text tensor placement explicit so the
    235B FP8 checkpoint can be sharded with device_map=auto.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = DEFAULT_FP8_MODEL_PATH,
        device: str = "cuda:0",
        torch_dtype: str = "auto",
        device_map: Optional[str] = "auto",
        max_memory: Optional[dict[Any, str]] = None,
        max_memory_spec: str = "",
        offload_folder: str = "",
        attn_implementation: str = "",
        trust_remote_code: bool = False,
        local_files_only: bool = True,
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
    ) -> "F3AQwenVL235BFP8":
        requested_device = torch.device(device)
        resolved_model_path = resolve_snapshot_path(model_path)
        dtype = _parse_dtype(torch_dtype, requested_device)
        max_memory = max_memory or _parse_max_memory(max_memory_spec)

        print(f"[fp8-load] resolved_model_path={resolved_model_path}", flush=True)
        print("[fp8-load] loading processor", flush=True)
        processor = AutoProcessor.from_pretrained(
            resolved_model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        print("[fp8-load] processor loaded", flush=True)

        model_cls, model_type = _resolve_qwen_vl_model_class(resolved_model_path)
        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        if device_map:
            load_kwargs["device_map"] = device_map
        if max_memory:
            load_kwargs["max_memory"] = max_memory
        if offload_folder:
            load_kwargs["offload_folder"] = offload_folder
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation

        print(
            "[fp8-load] "
            f"model_class={model_cls.__name__} model_type={model_type} "
            f"dtype={dtype} device_map={device_map or 'none'} max_memory={max_memory or 'auto'}",
            flush=True,
        )
        model = model_cls.from_pretrained(resolved_model_path, **load_kwargs).eval()
        print("[fp8-load] weights loaded", flush=True)
        if not device_map:
            print(f"[fp8-load] moving model to {requested_device}", flush=True)
            model.to(requested_device)

        input_device = _first_parameter_device(model.get_input_embeddings(), requested_device)
        visual_device = _first_parameter_device(model.model.visual, input_device)
        router_device = input_device if input_device.type != "meta" else requested_device
        print(
            f"[fp8-load] input_device={input_device} visual_device={visual_device} router_device={router_device}",
            flush=True,
        )

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
        instance = cls(
            model=model,
            processor=processor,
            device=router_device,
            router=router,
            text_conditioning_mode=text_conditioning_mode,
            fastv_prune_layer=fastv_prune_layer,
            model_type=model_type,
        )
        instance.input_device = input_device
        instance.visual_device = visual_device
        return instance

    def prepare_inputs(self, image_source: Any, prompt_text: str, system_prompt: Optional[str] = None) -> dict[str, torch.Tensor]:
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

        input_device = getattr(self, "input_device", self.device)
        visual_device = getattr(self, "visual_device", self.device)
        placed: dict[str, torch.Tensor] = {}
        for key, value in inputs.items():
            if not torch.is_tensor(value):
                placed[key] = value
                continue
            target = visual_device if key.startswith("pixel_values") or key.endswith("grid_thw") else input_device
            placed[key] = value.to(target)
        return placed

    def _get_single_image_tokens(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor):
        with torch.inference_mode():
            image_features = self.model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        if len(image_features.pooler_output) != 1:
            raise NotImplementedError("The current wrapper supports one image per sample")
        visual_tokens = image_features.pooler_output[0].to(self.device)
        return visual_tokens, self._grid_size_from_image_grid(image_grid_thw)


def build_fp8_model_from_args(args) -> F3AQwenVL235BFP8:
    return F3AQwenVL235BFP8.from_pretrained(
        model_path=getattr(args, "model_path", DEFAULT_FP8_MODEL_PATH),
        device=getattr(args, "device", "cuda:0"),
        torch_dtype=getattr(args, "torch_dtype", "auto"),
        device_map=getattr(args, "device_map", "auto") or None,
        max_memory_spec=getattr(args, "max_memory", ""),
        offload_folder=getattr(args, "offload_folder", ""),
        attn_implementation=getattr(args, "attn_implementation", ""),
        routing_mode=getattr(args, "routing_mode", "foraging"),
        text_conditioning_mode=getattr(args, "text_conditioning_mode", DEFAULT_TEXT_CONDITIONING_MODE),
        keep_ratio=getattr(args, "keep_ratio", DEFAULT_ROUTER_CONFIG["keep_ratio"]),
        router_heads=getattr(args, "router_heads", DEFAULT_ROUTER_CONFIG["router_heads"]),
        token_nonzero=getattr(args, "token_nonzero", DEFAULT_ROUTER_CONFIG["token_nonzero"]),
        odor_nonzero=getattr(args, "odor_nonzero", DEFAULT_ROUTER_CONFIG["odor_nonzero"]),
        head_topk=getattr(args, "head_topk", DEFAULT_ROUTER_CONFIG["head_topk"]),
        odor_topk=getattr(args, "odor_topk", DEFAULT_ROUTER_CONFIG["odor_topk"]),
        local_window_size=getattr(args, "local_window_size", DEFAULT_ROUTER_CONFIG["local_window_size"]),
        scaffold_keep=getattr(args, "scaffold_keep", DEFAULT_ROUTER_CONFIG["scaffold_keep"]),
        smell_weight=getattr(args, "smell_weight", DEFAULT_ROUTER_CONFIG["smell_weight"]),
        odor_gate_scale=getattr(args, "odor_gate_scale", DEFAULT_ROUTER_CONFIG["odor_gate_scale"]),
        odor_temperature=getattr(args, "odor_temperature", DEFAULT_ROUTER_CONFIG["odor_temperature"]),
        hypothesis_weight=getattr(args, "hypothesis_weight", DEFAULT_ROUTER_CONFIG["hypothesis_weight"]),
        contrast_weight=getattr(args, "contrast_weight", DEFAULT_ROUTER_CONFIG["contrast_weight"]),
        agreement_weight=getattr(args, "agreement_weight", DEFAULT_ROUTER_CONFIG["agreement_weight"]),
        repulsion_weight=getattr(args, "repulsion_weight", DEFAULT_ROUTER_CONFIG["repulsion_weight"]),
        region_agreement_weight=getattr(args, "region_agreement_weight", DEFAULT_ROUTER_CONFIG["region_agreement_weight"]),
        lockon_weight=getattr(args, "lockon_weight", DEFAULT_ROUTER_CONFIG["lockon_weight"]),
        visit_weight=getattr(args, "visit_weight", DEFAULT_ROUTER_CONFIG["visit_weight"]),
        uncertainty_weight=getattr(args, "uncertainty_weight", DEFAULT_ROUTER_CONFIG["uncertainty_weight"]),
        jump_ratio=getattr(args, "jump_ratio", DEFAULT_ROUTER_CONFIG["jump_ratio"]),
        ivc_keep_ratio=getattr(args, "ivc_keep_ratio", DEFAULT_ROUTER_CONFIG["ivc_keep_ratio"]),
        ivc_window_bonus=getattr(args, "ivc_window_bonus", DEFAULT_ROUTER_CONFIG["ivc_window_bonus"]),
        coarse_pos_weight=getattr(args, "coarse_pos_weight", DEFAULT_ROUTER_CONFIG["coarse_pos_weight"]),
        local_pos_weight=getattr(args, "local_pos_weight", DEFAULT_ROUTER_CONFIG["local_pos_weight"]),
        anchor_pos_weight=getattr(args, "anchor_pos_weight", DEFAULT_ROUTER_CONFIG["anchor_pos_weight"]),
        fastv_prune_layer=getattr(args, "fastv_prune_layer", DEFAULT_ROUTER_CONFIG["fastv_prune_layer"]),
        router_seed=getattr(args, "router_seed", DEFAULT_ROUTER_CONFIG["router_seed"]),
    )
