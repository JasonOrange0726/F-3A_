"""Training-free VisionZip utilities for F3A.

This module is intentionally standalone: it does not modify the existing
F3A wrapper or router.  The code follows the training-free VisionZip idea:
1) score visual tokens by vision-side attention/centrality,
2) keep dominant tokens,
3) merge part of the remaining tokens into contextual tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class VisionZipConfig:
    """Configuration for the training-free VisionZip baseline."""

    keep_ratio: float = 0.4
    contextual_ratio: float = 0.05
    min_contextual_tokens: int = 1
    score_layer: int = -2
    score_chunk_size: int = 128
    merge_chunk_size: int = 512
    eps: float = 1e-6


@dataclass
class VisionZipResult:
    """Compressed visual tokens and bookkeeping for evaluation output."""

    zipped_tokens: torch.Tensor
    selected_indices: torch.Tensor
    selected_scores: torch.Tensor
    base_scores: torch.Tensor
    dominant_indices: torch.Tensor
    contextual_indices: torch.Tensor
    dominant_count: int
    contextual_count: int
    full_visual_tokens: int
    selected_visual_tokens: int
    grid_size: tuple[int, int]


def _resolve_keep_count(num_tokens: int, keep_ratio: float) -> int:
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    keep = int(round(float(keep_ratio) * num_tokens))
    return max(1, min(num_tokens, keep))


def _split_keep_budget(num_tokens: int, keep_ratio: float, config: VisionZipConfig) -> tuple[int, int]:
    total_keep = _resolve_keep_count(num_tokens, keep_ratio)
    contextual = int(round(config.contextual_ratio * num_tokens))
    if config.contextual_ratio > 0 and total_keep > 1:
        contextual = max(config.min_contextual_tokens, contextual)
    contextual = min(contextual, max(0, total_keep - 1))
    dominant = total_keep - contextual
    return dominant, contextual


def _token_centrality_scores(tokens: torch.Tensor, chunk_size: int = 512, eps: float = 1e-6) -> torch.Tensor:
    """Fallback score: average cosine similarity received by each token."""
    normed = F.normalize(tokens.float(), dim=-1, eps=eps)
    scores = torch.zeros(tokens.size(0), device=tokens.device, dtype=torch.float32)
    denom = 0
    for start in range(0, tokens.size(0), chunk_size):
        end = min(tokens.size(0), start + chunk_size)
        sims = torch.matmul(normed[start:end], normed.T)
        scores += sims.sum(dim=0)
        denom += end - start
    return scores / max(1, denom)


def _uniform_targets(indices: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 0 or indices.numel() == 0:
        return indices.new_empty(0)
    count = min(count, indices.numel())
    if count == indices.numel():
        return indices
    positions = torch.linspace(0, indices.numel() - 1, steps=count, device=indices.device).round().long()
    return indices.index_select(0, positions).unique(sorted=True)


def _merge_contextual_tokens(
    tokens: torch.Tensor,
    remaining_indices: torch.Tensor,
    contextual_count: int,
    *,
    chunk_size: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge non-dominant tokens into uniformly sampled contextual targets."""
    if contextual_count <= 0 or remaining_indices.numel() == 0:
        return tokens.new_empty((0, tokens.size(-1))), remaining_indices.new_empty(0)

    target_indices = _uniform_targets(remaining_indices, contextual_count)
    if target_indices.numel() == 0:
        return tokens.new_empty((0, tokens.size(-1))), remaining_indices.new_empty(0)

    target_tokens = tokens.index_select(0, target_indices)
    target_normed = F.normalize(target_tokens.float(), dim=-1, eps=eps)
    remaining_tokens = tokens.index_select(0, remaining_indices)
    remaining_normed = F.normalize(remaining_tokens.float(), dim=-1, eps=eps)

    sums = target_tokens.float().clone()
    counts = torch.ones(target_indices.numel(), device=tokens.device, dtype=torch.float32)
    is_target = torch.isin(remaining_indices, target_indices)
    merge_positions = torch.nonzero(~is_target, as_tuple=False).flatten()

    for start in range(0, merge_positions.numel(), chunk_size):
        end = min(merge_positions.numel(), start + chunk_size)
        pos = merge_positions[start:end]
        source_tokens = remaining_tokens.index_select(0, pos).float()
        source_normed = remaining_normed.index_select(0, pos)
        assignment = torch.matmul(source_normed, target_normed.T).argmax(dim=1)
        sums.index_add_(0, assignment, source_tokens)
        counts.index_add_(0, assignment, torch.ones_like(assignment, dtype=torch.float32))

    context_tokens = (sums / counts.clamp_min(1.0).unsqueeze(-1)).to(tokens.dtype)
    return context_tokens, target_indices


def zip_projected_tokens(
    visual_tokens: torch.Tensor,
    grid_size: tuple[int, int],
    *,
    keep_ratio: Optional[float] = None,
    attention_scores: Optional[torch.Tensor] = None,
    config: Optional[VisionZipConfig] = None,
) -> VisionZipResult:
    """Apply training-free VisionZip to already projected visual tokens.

    If ``attention_scores`` is provided, it is used for dominant token selection.
    Otherwise a text-agnostic centrality score is used as a safe fallback.
    """
    config = config or VisionZipConfig()
    ratio = config.keep_ratio if keep_ratio is None else keep_ratio
    if visual_tokens.ndim != 2:
        raise ValueError("visual_tokens must have shape [num_tokens, hidden_dim]")

    num_tokens = int(visual_tokens.size(0))
    dominant_count, contextual_count = _split_keep_budget(num_tokens, ratio, config)
    if attention_scores is None:
        base_scores = _token_centrality_scores(
            visual_tokens,
            chunk_size=config.merge_chunk_size,
            eps=config.eps,
        )
    else:
        base_scores = attention_scores.to(device=visual_tokens.device, dtype=torch.float32).flatten()
        if base_scores.numel() != num_tokens:
            raise ValueError(
                f"attention_scores length ({base_scores.numel()}) must match visual token count ({num_tokens})"
            )

    _, dominant_indices = base_scores.topk(k=dominant_count, dim=0)
    dominant_indices = dominant_indices.sort().values
    all_indices = torch.arange(num_tokens, device=visual_tokens.device, dtype=torch.long)
    dominant_mask = torch.zeros(num_tokens, device=visual_tokens.device, dtype=torch.bool)
    dominant_mask[dominant_indices] = True
    remaining_indices = all_indices[~dominant_mask]

    context_tokens, contextual_indices = _merge_contextual_tokens(
        tokens=visual_tokens,
        remaining_indices=remaining_indices,
        contextual_count=contextual_count,
        chunk_size=config.merge_chunk_size,
        eps=config.eps,
    )

    dominant_tokens = visual_tokens.index_select(0, dominant_indices)
    selected_indices = torch.cat([dominant_indices, contextual_indices], dim=0)
    selected_tokens = torch.cat([dominant_tokens, context_tokens], dim=0)
    selected_scores = base_scores.index_select(0, selected_indices)

    order = selected_indices.argsort()
    selected_indices = selected_indices.index_select(0, order)
    selected_tokens = selected_tokens.index_select(0, order)
    selected_scores = selected_scores.index_select(0, order)

    return VisionZipResult(
        zipped_tokens=selected_tokens,
        selected_indices=selected_indices,
        selected_scores=selected_scores,
        base_scores=base_scores,
        dominant_indices=dominant_indices,
        contextual_indices=contextual_indices,
        dominant_count=int(dominant_indices.numel()),
        contextual_count=int(contextual_indices.numel()),
        full_visual_tokens=num_tokens,
        selected_visual_tokens=int(selected_indices.numel()),
        grid_size=grid_size,
    )


def _vision_apply_rotary(model_type: str, query: torch.Tensor, key: torch.Tensor, position_embeddings):
    if model_type == "qwen3_vl_moe":
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import apply_rotary_pos_emb_vision
    else:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision

    return apply_rotary_pos_emb_vision(query, key, position_embeddings[0], position_embeddings[1])


def _qwen3_vision_received_attention_scores(
    attn_module,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    *,
    model_type: str,
    chunk_size: int,
) -> torch.Tensor:
    """Compute vision-token received attention in chunks to avoid SxS OOM."""
    seq_length = hidden_states.shape[0]
    query, key, _ = (
        attn_module.qkv(hidden_states).reshape(seq_length, 3, attn_module.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    query, key = _vision_apply_rotary(model_type, query, key, position_embeddings)
    query = query.transpose(0, 1).unsqueeze(0)
    key = key.transpose(0, 1).unsqueeze(0)
    key_t = key.transpose(2, 3)

    scores = torch.zeros(seq_length, device=hidden_states.device, dtype=torch.float32)
    scaling = getattr(attn_module, "scaling", query.size(-1) ** -0.5)
    for seg_start, seg_end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        seg_start = int(seg_start)
        seg_end = int(seg_end)
        denom = 0
        for start in range(seg_start, seg_end, chunk_size):
            end = min(seg_end, start + chunk_size)
            weights = torch.matmul(query[:, :, start:end, :], key_t[:, :, :, seg_start:seg_end]) * scaling
            weights = torch.softmax(weights, dim=-1, dtype=torch.float32)
            scores[seg_start:seg_end] += weights.sum(dim=(0, 1, 2))
            denom += query.size(0) * query.size(1) * (end - start)
        scores[seg_start:seg_end] /= max(1, denom)
    return scores


@torch.inference_mode()
def qwen3_visionzip_projected_tokens(
    model,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    keep_ratio: Optional[float] = None,
    config: Optional[VisionZipConfig] = None,
) -> VisionZipResult:
    """Extract Qwen3-VL visual tokens and apply VisionZip without changing model code.

    This mirrors the Qwen3 vision forward pass, computes a no-CLS VisionZip score
    at ``config.score_layer``, aggregates scores to post-merge visual tokens, then
    applies dominant selection + contextual merging.
    """
    config = config or VisionZipConfig()
    visual = model.model.visual
    model_type = getattr(model.config, "model_type", "qwen3_vl")

    hidden_states = visual.patch_embed(pixel_values)
    hidden_states = hidden_states + visual.fast_pos_embed_interpolate(image_grid_thw)
    rotary_pos_emb = visual.rot_pos_emb(image_grid_thw)
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    layer_count = len(visual.blocks)
    score_layer = config.score_layer if config.score_layer >= 0 else layer_count + config.score_layer
    score_layer = max(0, min(layer_count - 1, score_layer))
    premerge_scores = None

    for layer_num, block in enumerate(visual.blocks):
        if layer_num == score_layer:
            premerge_scores = _qwen3_vision_received_attention_scores(
                block.attn,
                block.norm1(hidden_states),
                cu_seqlens,
                position_embeddings,
                model_type=model_type,
                chunk_size=config.score_chunk_size,
            )
        hidden_states = block(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )

    projected_tokens = visual.merger(hidden_states)
    spatial_merge = int(visual.spatial_merge_size)
    group_size = spatial_merge * spatial_merge
    if premerge_scores is None:
        merged_scores = None
    else:
        merged_scores = premerge_scores.reshape(-1, group_size).mean(dim=1)

    grid_h = int(image_grid_thw[0, 1].item() // spatial_merge)
    grid_w = int(image_grid_thw[0, 2].item() // spatial_merge)
    return zip_projected_tokens(
        projected_tokens,
        grid_size=(grid_h, grid_w),
        keep_ratio=keep_ratio,
        attention_scores=merged_scores,
        config=config,
    )
