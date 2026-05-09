from pathlib import Path
from typing import Iterable

from .defaults import DEFAULT_DATASET_ROOT, DEFAULT_MODEL_HUB


DATASET_DIR_NAMES = {
    "pope": "POPE",
    "chartqa": "ChartQA",
    "ai2d": "ai2d",
    "hallusionbench": "HallusionBench",
    "mme": "MME",
    "scienceqa": "ScienceQA",
    "realworldqa": "RealWorldQA",
    "mmbench": "MMBench-en",
    "mmbench_v11": "MMBench-en-V11",
    "vsr": "VSR",
    "visual7w": "Visual7W",
    "textvqa": "TextVQA",
    "hateful_memes": "Multimodal_Hatefule_MeMe_Data",
}


def _snapshot_from_cache_dir(cache_dir: Path) -> Path | None:
    if (cache_dir / "config.json").is_file():
        return cache_dir
    snapshots = cache_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    ref_path = cache_dir / "refs" / "main"
    if ref_path.is_file():
        ref = ref_path.read_text(encoding="utf-8").strip()
        snapshot = snapshots / ref
        if (snapshot / "config.json").is_file():
            return snapshot
    candidates = [path for path in snapshots.iterdir() if (path / "config.json").is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_model_path(model_path: str, model_hub: str | Path = DEFAULT_MODEL_HUB) -> str:
    """Resolve model ids or HF cache dirs to a concrete local snapshot when possible."""
    raw = str(model_path)
    direct = Path(raw).expanduser()
    if direct.exists():
        snapshot = _snapshot_from_cache_dir(direct)
        return str(snapshot or direct)

    hub = Path(model_hub).expanduser()
    candidates: list[Path] = []
    if "/" in raw:
        org, name = raw.split("/", 1)
        candidates.append(hub / f"models--{org}--{name}")
    else:
        candidates.extend(
            [
                hub / f"models--Qwen--{raw}",
                hub / raw,
            ]
        )
    for candidate in candidates:
        snapshot = _snapshot_from_cache_dir(candidate)
        if snapshot is not None:
            return str(snapshot)
    return raw


def candidate_dataset_roots(dataset: str, explicit_root: str = "") -> Iterable[Path]:
    if explicit_root:
        yield Path(explicit_root).expanduser()
        return

    dirname = DATASET_DIR_NAMES.get(dataset, dataset)
    roots = [
        Path(DEFAULT_DATASET_ROOT).expanduser() / dirname,
        Path(DEFAULT_DATASET_ROOT).expanduser() / "datasets" / dirname,
    ]
    if dataset == "mmbench":
        roots.append(Path(DEFAULT_DATASET_ROOT).expanduser() / DATASET_DIR_NAMES["mmbench_v11"])
        roots.append(Path(DEFAULT_DATASET_ROOT).expanduser() / "datasets" / DATASET_DIR_NAMES["mmbench_v11"])
    for root in roots:
        yield root


def resolve_dataset_root(dataset: str, explicit_root: str = "") -> Path:
    candidates = list(candidate_dataset_roots(dataset, explicit_root))
    for root in candidates:
        if root.exists():
            return root
    return candidates[0]


def require_files(paths: list[str], description: str) -> list[str]:
    if not paths:
        raise FileNotFoundError(
            f"No files found for {description}. Set --dataset-root or F3A_DATASET_ROOT to the downloaded dataset."
        )
    return paths
