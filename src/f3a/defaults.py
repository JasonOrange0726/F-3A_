from pathlib import Path
import os


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_HUB = os.environ.get("F3A_MODEL_HUB", "")
DEFAULT_DATASET_ROOT = os.environ.get("F3A_DATASET_ROOT", str(DEFAULT_PROJECT_ROOT / "data"))
DEFAULT_OUTPUT_ROOT = os.environ.get(
    "F3A_OUTPUT_ROOT",
    str(DEFAULT_PROJECT_ROOT / "outputs"),
)
DEFAULT_MODEL_PATH = os.environ.get("F3A_DEFAULT_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
DEFAULT_TEXT_CONDITIONING_MODE = "universal_three_cue"

# Current main configuration after the IVC-style positional refinement.
DEFAULT_ROUTER_CONFIG = {
    "router_heads": 16,
    "token_nonzero": 32,
    "odor_nonzero": 8,
    "head_topk": 4,
    "odor_topk": 4,
    "local_window_size": 2,
    "scaffold_keep": 1,
    "keep_ratio": 0.4,
    "smell_weight": 0.15,
    "odor_gate_scale": 1.0,
    "odor_temperature": 0.5,
    "hypothesis_weight": 0.35,
    "contrast_weight": 0.25,
    "agreement_weight": 0.45,
    "repulsion_weight": 0.35,
    "region_agreement_weight": 1.0,
    "lockon_weight": 0.35,
    "visit_weight": 0.25,
    "uncertainty_weight": 0.25,
    "jump_ratio": 0.15,
    "ivc_keep_ratio": 0.10,
    "ivc_window_bonus": 0.20,
    "coarse_pos_weight": 0.18,
    "local_pos_weight": 0.22,
    "anchor_pos_weight": 0.16,
    "fastv_prune_layer": 2,
    "router_seed": 42,
}
