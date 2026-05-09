from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_sparse_random_matrix(
    out_dim: int,
    in_dim: int,
    non_zero_per_row: int,
    seed: int,
) -> torch.Tensor:
    if out_dim <= 0 or in_dim <= 0:
        raise ValueError("out_dim and in_dim must be positive")
    if non_zero_per_row <= 0 or non_zero_per_row > in_dim:
        raise ValueError("non_zero_per_row must be in [1, in_dim]")
    generator = torch.Generator()
    generator.manual_seed(seed)
    matrix = torch.zeros(out_dim, in_dim, dtype=torch.float32)
    scale = 1.0 / math.sqrt(max(out_dim, 1))
    for row in range(out_dim):
        perm = torch.randperm(in_dim, generator=generator)
        idx = perm[:non_zero_per_row]
        matrix[row, idx] = torch.randn(non_zero_per_row, generator=generator) * scale
    return matrix


def _window_indices(grid_h: int, grid_w: int, window_size: int, device: torch.device) -> torch.Tensor:
    windows = []
    for top in range(0, grid_h, window_size):
        for left in range(0, grid_w, window_size):
            current = []
            for row in range(top, min(top + window_size, grid_h)):
                for col in range(left, min(left + window_size, grid_w)):
                    current.append(row * grid_w + col)
            windows.append(current)
    max_len = max(len(window) for window in windows)
    padded = torch.full((len(windows), max_len), -1, dtype=torch.long, device=device)
    for i, current in enumerate(windows):
        padded[i, : len(current)] = torch.tensor(current, dtype=torch.long, device=device)
    return padded


@dataclass
class RoutingResult:
    selected_indices: torch.Tensor
    selected_scores: torch.Tensor
    base_scores: torch.Tensor
    route_scores: torch.Tensor
    query_scores: torch.Tensor
    hypothesis_scores: Optional[torch.Tensor]
    contrast_scores: Optional[torch.Tensor]
    agreement_scores: Optional[torch.Tensor]
    repulsion_scores: Optional[torch.Tensor]
    uncertainty_scores: Optional[torch.Tensor]
    region_agreement_scores: Optional[torch.Tensor]
    odor_head_gate: torch.Tensor
    head_logits: torch.Tensor
    kept_count: int
    grid_size: tuple[int, int]
    scaffold_indices: Optional[torch.Tensor] = None
    exploit_indices: Optional[torch.Tensor] = None
    jump_indices: Optional[torch.Tensor] = None
    fill_indices: Optional[torch.Tensor] = None
    active_window_indices: Optional[torch.Tensor] = None
    context_window_indices: Optional[torch.Tensor] = None
    anchor_window_indices: Optional[torch.Tensor] = None
    ivc_indices: Optional[torch.Tensor] = None


class OdorConditionedF3ARouter(nn.Module):
    def __init__(
        self,
        token_dim: int,
        odor_dim: int,
        num_heads: int = 16,
        token_nonzero: int = 32,
        odor_nonzero: int = 8,
        head_topk: int = 4,
        odor_topk: int = 4,
        shared_head_dim: int = 128,
        head_proto_nonzero: int = 16,
        local_window_size: int = 2,
        scaffold_keep: int = 1,
        default_keep_ratio: float = 0.5,
        smell_weight: float = 0.15,
        odor_gate_scale: float = 1.0,
        odor_temperature: float = 0.5,
        visual_temperature: float = 0.7,
        hypothesis_weight: float = 0.35,
        contrast_weight: float = 0.25,
        agreement_weight: float = 0.45,
        repulsion_weight: float = 0.35,
        routing_mode: str = "foraging",
        region_agreement_weight: float = 1.0,
        lockon_weight: float = 0.35,
        visit_weight: float = 0.25,
        uncertainty_weight: float = 0.25,
        jump_ratio: float = 0.15,
        cross_similarity_dim: int = 96,
        cross_token_nonzero: int = 24,
        cross_odor_nonzero: int = 24,
        cross_similarity_weight: float = 0.35,
        ivc_keep_ratio: float = 0.10,
        ivc_rope_dim: int = 128,
        ivc_window_bonus: float = 0.20,
        coarse_pos_weight: float = 0.18,
        local_pos_weight: float = 0.22,
        anchor_pos_weight: float = 0.16,
        seed: int = 42,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if scaffold_keep <= 0:
            raise ValueError("scaffold_keep must be positive")
        self.token_dim = token_dim
        self.odor_dim = odor_dim
        self.num_heads = num_heads
        self.head_topk = max(1, min(head_topk, num_heads))
        self.odor_topk = max(1, min(odor_topk, num_heads))
        self.shared_head_dim = shared_head_dim
        self.local_window_size = max(1, local_window_size)
        self.scaffold_keep = scaffold_keep
        self.default_keep_ratio = default_keep_ratio
        self.smell_weight = smell_weight
        self.odor_gate_scale = odor_gate_scale
        self.odor_temperature = max(odor_temperature, eps)
        self.visual_temperature = max(visual_temperature, eps)
        self.hypothesis_weight = hypothesis_weight
        self.contrast_weight = contrast_weight
        self.agreement_weight = agreement_weight
        self.repulsion_weight = repulsion_weight
        self.routing_mode = routing_mode
        self.region_agreement_weight = region_agreement_weight
        self.lockon_weight = lockon_weight
        self.visit_weight = visit_weight
        self.uncertainty_weight = uncertainty_weight
        self.jump_ratio = jump_ratio
        self.cross_similarity_weight = cross_similarity_weight
        self.ivc_keep_ratio = max(0.0, ivc_keep_ratio)
        self.ivc_rope_dim = max(2, ivc_rope_dim if ivc_rope_dim % 2 == 0 else ivc_rope_dim - 1)
        self.ivc_window_bonus = max(0.0, ivc_window_bonus)
        self.coarse_pos_weight = max(0.0, coarse_pos_weight)
        self.local_pos_weight = max(0.0, local_pos_weight)
        self.anchor_pos_weight = max(0.0, anchor_pos_weight)
        self.eps = eps

        vision_head_proj = build_sparse_random_matrix(
            out_dim=shared_head_dim,
            in_dim=token_dim,
            non_zero_per_row=min(token_nonzero, token_dim),
            seed=seed,
        )
        odor_head_proj = build_sparse_random_matrix(
            out_dim=shared_head_dim,
            in_dim=odor_dim,
            non_zero_per_row=min(odor_nonzero, odor_dim),
            seed=seed + 1,
        )
        shared_head_matrix = build_sparse_random_matrix(
            out_dim=num_heads,
            in_dim=shared_head_dim,
            non_zero_per_row=min(head_proto_nonzero, shared_head_dim),
            seed=seed + 2,
        )
        vision_align_matrix = build_sparse_random_matrix(
            out_dim=cross_similarity_dim,
            in_dim=token_dim,
            non_zero_per_row=min(cross_token_nonzero, token_dim),
            seed=seed + 3,
        )
        odor_align_matrix = build_sparse_random_matrix(
            out_dim=cross_similarity_dim,
            in_dim=odor_dim,
            non_zero_per_row=min(cross_odor_nonzero, odor_dim),
            seed=seed + 4,
        )
        self.register_buffer("vision_head_proj", vision_head_proj)
        self.register_buffer("odor_head_proj", odor_head_proj)
        self.register_buffer("shared_head_matrix", shared_head_matrix)
        self.register_buffer("vision_align_matrix", vision_align_matrix)
        self.register_buffer("odor_align_matrix", odor_align_matrix)

    def _build_simple_result(
        self,
        selected_indices: torch.Tensor,
        scores: torch.Tensor,
        grid_size: tuple[int, int],
    ) -> RoutingResult:
        device = scores.device
        selected_indices = selected_indices.sort().values
        selected_scores = scores.index_select(0, selected_indices)
        return RoutingResult(
            selected_indices=selected_indices,
            selected_scores=selected_scores,
            base_scores=scores,
            route_scores=scores,
            query_scores=scores,
            hypothesis_scores=None,
            contrast_scores=None,
            agreement_scores=None,
            repulsion_scores=None,
            uncertainty_scores=None,
            region_agreement_scores=None,
            odor_head_gate=torch.zeros(self.num_heads, device=device, dtype=torch.float32),
            head_logits=torch.zeros((scores.numel(), self.num_heads), device=device, dtype=torch.float32),
            kept_count=int(selected_indices.numel()),
            grid_size=grid_size,
            active_window_indices=None,
            context_window_indices=None,
            anchor_window_indices=None,
        )

    def _normalize_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        centered = tokens - tokens.mean(dim=1, keepdim=True)
        scale = centered.std(dim=1, keepdim=True, unbiased=False) + self.eps
        return centered / scale

    def _compute_visual_head_logits(self, token_features: torch.Tensor) -> torch.Tensor:
        vision_latent = token_features.float() @ self.vision_head_proj.t()
        vision_latent = F.normalize(vision_latent, dim=-1)
        head_basis = F.normalize(self.shared_head_matrix.float(), dim=-1)
        return vision_latent @ head_basis.t()

    def _compute_visual_head_probs(self, visual_head_logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(visual_head_logits / self.visual_temperature, dim=-1)

    def _compute_odor_gate(self, odor_cue: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        odor_centered = odor_cue - odor_cue.mean(dim=-1, keepdim=True)
        odor_norm = odor_centered / (odor_cue.std(dim=-1, keepdim=True, unbiased=False) + self.eps)
        odor_latent = odor_norm @ self.odor_head_proj.t()
        odor_latent = F.normalize(odor_latent, dim=-1)
        head_basis = F.normalize(self.shared_head_matrix.float(), dim=-1)
        odor_logits = odor_latent @ head_basis.t()
        odor_energy = 0.5 * (odor_logits + 1.0)
        if self.odor_topk < self.num_heads:
            top_vals, top_idx = odor_energy.topk(k=self.odor_topk, dim=-1)
            sparse_energy = torch.full_like(odor_energy, float("-inf"))
            sparse_energy.scatter_(dim=-1, index=top_idx, src=top_vals)
            odor_energy = sparse_energy
        odor_gate = F.softmax(odor_energy / self.odor_temperature, dim=-1)
        if abs(self.odor_gate_scale - 1.0) > self.eps:
            odor_gate = odor_gate.pow(self.odor_gate_scale)
            odor_gate = odor_gate / odor_gate.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return odor_logits, odor_gate

    def _compute_multi_odor_gate(self, odor_cues: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if odor_cues.dim() != 3:
            raise ValueError("odor_cues must have shape [B, C, D]")
        batch, num_cues, odor_dim = odor_cues.shape
        flat_logits, flat_gate = self._compute_odor_gate(odor_cues.reshape(batch * num_cues, odor_dim))
        return (
            flat_logits.view(batch, num_cues, self.num_heads),
            flat_gate.view(batch, num_cues, self.num_heads),
        )

    def _compute_cue_scores(
        self,
        visual_head_probs: torch.Tensor,
        odor_cues: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        odor_head_logits, odor_head_gate = self._compute_multi_odor_gate(odor_cues.float())
        cue_scores = (visual_head_probs.unsqueeze(1) * odor_head_gate.unsqueeze(2)).sum(dim=-1)
        return cue_scores, odor_head_logits, odor_head_gate

    def _compute_signed_cue_scores(
        self,
        visual_head_logits: torch.Tensor,
        odor_cues: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        odor_head_logits, odor_head_gate = self._compute_multi_odor_gate(odor_cues.float())
        token_basis = visual_head_logits / (visual_head_logits.norm(dim=-1, keepdim=True) + self.eps)
        odor_basis = odor_head_logits * odor_head_gate
        odor_basis = odor_basis / (odor_basis.norm(dim=-1, keepdim=True) + self.eps)
        signed_scores = (token_basis.unsqueeze(1) * odor_basis.unsqueeze(2)).sum(dim=-1)
        return signed_scores, odor_head_logits, odor_head_gate

    def _compute_agreement_score(
        self,
        visual_head_probs: torch.Tensor,
        query_head_gate: torch.Tensor,
        hypothesis_head_gate: torch.Tensor,
        margin_norm: torch.Tensor,
    ) -> torch.Tensor:
        joint_gate = query_head_gate.unsqueeze(1) * hypothesis_head_gate
        joint_gate = joint_gate / joint_gate.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        agreement = (visual_head_probs * joint_gate).sum(dim=-1).clamp_min(0.0)
        return torch.sqrt(agreement * margin_norm + self.eps)

    def _compute_cross_modal_similarity(
        self,
        token_features: torch.Tensor,
        odor_cue: torch.Tensor,
    ) -> torch.Tensor:
        token_features = token_features.float()
        odor_cue = odor_cue.float()
        vision_basis = token_features @ self.vision_align_matrix.t()
        vision_basis = F.normalize(vision_basis, dim=-1)

        odor_centered = odor_cue - odor_cue.mean(dim=-1, keepdim=True)
        odor_scale = odor_centered.std(dim=-1, keepdim=True, unbiased=False) + self.eps
        odor_basis = (odor_centered / odor_scale) @ self.odor_align_matrix.t()
        odor_basis = F.normalize(odor_basis, dim=-1)

        similarity = (vision_basis * odor_basis.unsqueeze(1)).sum(dim=-1)
        similarity = similarity.clamp(min=-1.0, max=1.0)
        return 0.5 * (similarity + 1.0)

    def _unit_normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        min_vals = scores.min(dim=1, keepdim=True).values
        max_vals = scores.max(dim=1, keepdim=True).values
        return (scores - min_vals) / (max_vals - min_vals + self.eps)

    def _positive_normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        positive = scores.clamp_min(0.0)
        max_vals = positive.max(dim=1, keepdim=True).values
        return positive / (max_vals + self.eps)

    def _window_stats(
        self,
        scores: torch.Tensor,
        windows: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        safe_windows = windows.masked_fill(~valid, 0)
        window_scores = scores[:, safe_windows]
        window_scores = window_scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
        top_count = min(2, window_scores.size(-1))
        top_vals, _ = window_scores.topk(k=top_count, dim=-1)
        finite_mask = torch.isfinite(top_vals)
        top_vals = torch.where(finite_mask, top_vals, torch.zeros_like(top_vals))
        denom = finite_mask.sum(dim=-1).clamp_min(1)
        return top_vals.sum(dim=-1) / denom

    def _select_ivc_indices(
        self,
        num_tokens: int,
        keep_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.ivc_keep_ratio <= self.eps or num_tokens <= 0 or keep_count <= 0:
            return torch.empty(0, device=device, dtype=torch.long)

        # Treat the IVC anchors as a light supplemental scaffold on top of the
        # main foraging route, rather than letting them dominate the retained budget.
        total_anchor_keep = max(1, int(round(keep_count * self.ivc_keep_ratio)))
        total_anchor_keep = min(total_anchor_keep, max(1, keep_count // 4))
        per_axis_keep = max(1, math.ceil(total_anchor_keep / 2.0))
        if total_anchor_keep <= 0 or per_axis_keep <= 0:
            return torch.empty(0, device=device, dtype=torch.long)

        positions = torch.arange(num_tokens, device=device, dtype=torch.float32).unsqueeze(1)
        pair_ids = torch.arange(self.ivc_rope_dim // 2, device=device, dtype=torch.float32).unsqueeze(0)
        theta = torch.pow(10000.0, -2.0 * pair_ids / float(self.ivc_rope_dim))
        phases = positions * theta
        v_scores = torch.cos(phases).sum(dim=1)
        u_scores = torch.sin(phases).sum(dim=1)

        top_v = torch.topk(v_scores, k=min(per_axis_keep, num_tokens), dim=0).indices
        top_u = torch.topk(u_scores, k=min(per_axis_keep, num_tokens), dim=0).indices
        ivc_indices = torch.cat([top_v, top_u], dim=0).unique(sorted=True)
        if ivc_indices.numel() > total_anchor_keep:
            ivc_indices = ivc_indices[:total_anchor_keep]
        return ivc_indices

    def _allocate_window_budget(
        self,
        total_budget: int,
        weights: torch.Tensor,
        caps: torch.Tensor,
    ) -> torch.Tensor:
        allocation = torch.zeros_like(caps)
        if total_budget <= 0 or caps.sum().item() <= 0:
            return allocation
        positive_caps = caps > 0
        if positive_caps.sum().item() == 0:
            return allocation
        weights = weights.clamp_min(0.0)
        if weights[positive_caps].sum().item() <= 0:
            weights = positive_caps.float()
        raw = total_budget * weights / weights.sum().clamp_min(self.eps)
        allocation = torch.minimum(torch.floor(raw).long(), caps)
        remaining = min(total_budget - int(allocation.sum().item()), int((caps - allocation).sum().item()))
        if remaining <= 0:
            return allocation
        residual = raw - allocation.float()
        for _ in range(remaining):
            masked_residual = residual.masked_fill(allocation >= caps, float("-inf"))
            idx = int(masked_residual.argmax().item())
            if not torch.isfinite(masked_residual[idx]):
                break
            allocation[idx] += 1
        return allocation

    def _compute_visit_penalty(
        self,
        candidate_indices: torch.Tensor,
        selected_indices: torch.Tensor,
        coords: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        if candidate_indices.numel() == 0 or selected_indices.numel() == 0 or self.visit_weight <= 0:
            return torch.zeros(candidate_indices.numel(), device=coords.device, dtype=torch.float32)
        candidate_coords = coords.index_select(0, candidate_indices)
        selected_coords = coords.index_select(0, selected_indices)
        dist2 = (candidate_coords[:, None, :] - selected_coords[None, :, :]).pow(2).sum(dim=-1)
        return torch.exp(-dist2 / (2.0 * sigma * sigma + self.eps)).amax(dim=1)

    def _window_grid_coords(self, grid_size: tuple[int, int], device: torch.device) -> torch.Tensor:
        grid_h, grid_w = grid_size
        num_windows_h = (grid_h + self.local_window_size - 1) // self.local_window_size
        num_windows_w = (grid_w + self.local_window_size - 1) // self.local_window_size
        rows = torch.arange(num_windows_h, device=device, dtype=torch.float32)
        cols = torch.arange(num_windows_w, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(rows, cols, indexing="ij")
        return torch.stack([yy.flatten(), xx.flatten()], dim=-1)

    def _spread_window_selection(
        self,
        candidate_indices: torch.Tensor,
        coords: torch.Tensor,
        count: int,
        seed_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if count <= 0 or candidate_indices.numel() == 0:
            return torch.empty(0, device=coords.device, dtype=torch.long)

        remaining = candidate_indices.clone()
        selected = [] if seed_indices is None else seed_indices.detach().cpu().tolist()
        seed_coords = coords.index_select(0, seed_indices) if seed_indices is not None and seed_indices.numel() > 0 else None

        for _ in range(min(count, remaining.numel())):
            candidate_coords = coords.index_select(0, remaining)
            if seed_coords is None and not selected:
                center = candidate_coords.mean(dim=0, keepdim=True)
                score = (candidate_coords - center).pow(2).sum(dim=-1)
            else:
                if seed_coords is None:
                    selected_tensor = torch.tensor(selected, device=coords.device, dtype=torch.long)
                    reference_coords = coords.index_select(0, selected_tensor)
                else:
                    if selected:
                        selected_tensor = torch.tensor(selected, device=coords.device, dtype=torch.long)
                        reference_coords = torch.cat([seed_coords, coords.index_select(0, selected_tensor)], dim=0)
                    else:
                        reference_coords = seed_coords
                dist2 = (candidate_coords[:, None, :] - reference_coords[None, :, :]).pow(2).sum(dim=-1)
                score = dist2.min(dim=1).values

            best_pos = int(score.argmax().item())
            picked = int(remaining[best_pos].item())
            selected.append(picked)
            keep_mask = torch.ones_like(remaining, dtype=torch.bool)
            keep_mask[best_pos] = False
            remaining = remaining[keep_mask]

        if seed_indices is not None and seed_indices.numel() > 0:
            filtered = [idx for idx in selected if idx not in set(seed_indices.detach().cpu().tolist())]
        else:
            filtered = selected
        return torch.tensor(filtered, device=coords.device, dtype=torch.long)

    def _select_scaffold_windows(
        self,
        keep_count: int,
        num_tokens: int,
        region_strength: torch.Tensor,
        region_agreement: torch.Tensor,
        region_uncertainty: torch.Tensor,
        valid: torch.Tensor,
        grid_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = region_strength.device
        num_windows = region_strength.numel()
        if num_windows == 0:
            empty = torch.empty(0, device=device, dtype=torch.long)
            return empty, empty, empty

        window_coords = self._window_grid_coords(grid_size=grid_size, device=device)
        avg_window_capacity = valid.sum(dim=-1).float().mean().item()
        avg_window_capacity = max(avg_window_capacity, 1.0)

        active_budget = int(round((keep_count / avg_window_capacity) * 0.40))
        active_budget = max(1, active_budget)
        active_budget = min(active_budget, num_windows)

        min_active = min(num_windows, max(1, num_windows // 5))
        active_budget = max(active_budget, min_active)

        context_budget = int(round(active_budget * 0.6))
        context_budget = min(max(1, context_budget), max(0, num_windows - active_budget))

        window_priority = (
            region_strength
            * (1.0 + self.region_agreement_weight * region_agreement)
            + 0.15 * torch.sqrt(region_agreement.clamp_min(0.0) + self.eps)
            - 0.20 * region_uncertainty
        )
        _, active_pos = window_priority.topk(k=active_budget, dim=0)
        active_windows = active_pos.sort().values

        if context_budget <= 0 or active_budget >= num_windows:
            empty = torch.empty(0, device=device, dtype=torch.long)
            return active_windows, empty, empty

        active_mask = torch.zeros(num_windows, dtype=torch.bool, device=device)
        active_mask[active_windows] = True
        remaining = torch.nonzero(~active_mask, as_tuple=False).squeeze(-1)
        active_coords = window_coords.index_select(0, active_windows)
        remaining_coords = window_coords.index_select(0, remaining)
        dist2 = (remaining_coords[:, None, :] - active_coords[None, :, :]).pow(2).sum(dim=-1)
        nearest_dist2 = dist2.min(dim=1).values
        adjacency = torch.exp(-nearest_dist2 / (2.0 * 1.15 * 1.15 + self.eps))
        context_priority = (
            0.55 * adjacency
            + 0.25 * region_strength.index_select(0, remaining)
            + 0.15 * region_agreement.index_select(0, remaining)
            + 0.05 * (1.0 - region_uncertainty.index_select(0, remaining))
        )
        context_k = min(context_budget, int(remaining.numel()))
        _, context_pos = context_priority.topk(k=context_k, dim=0)
        context_windows = remaining.index_select(0, context_pos).sort().values

        used_mask = active_mask.clone()
        used_mask[context_windows] = True
        anchor_budget = min(max(1, num_windows // 12), int((~used_mask).sum().item()))
        if anchor_budget <= 0:
            anchor_windows = torch.empty(0, device=device, dtype=torch.long)
        else:
            fallback = torch.nonzero(~used_mask, as_tuple=False).squeeze(-1)
            anchor_windows = self._spread_window_selection(
                candidate_indices=fallback,
                coords=window_coords,
                count=anchor_budget,
                seed_indices=torch.cat([active_windows, context_windows], dim=0),
            )
        return active_windows, context_windows, anchor_windows

    def _select_window_candidates(
        self,
        candidate_indices: torch.Tensor,
        base_indices: torch.Tensor,
        num_select: int,
        support_scores: torch.Tensor,
        agreement_scores: torch.Tensor,
        uncertainty_scores: torch.Tensor,
        token_features: torch.Tensor,
        coords: torch.Tensor,
        already_selected: torch.Tensor,
    ) -> list[int]:
        if num_select <= 0 or candidate_indices.numel() == 0:
            return []
        chosen: list[int] = []
        if base_indices.numel() > 0:
            anchor_feature = token_features.index_select(0, base_indices).mean(dim=0, keepdim=True)
            anchor_feature = F.normalize(anchor_feature, dim=-1)
            anchor_coords = coords.index_select(0, base_indices).mean(dim=0, keepdim=True)
        else:
            anchor_feature = None
            anchor_coords = None

        remaining = candidate_indices.clone()
        dynamic_selected = already_selected.clone()
        sigma = max(float(self.local_window_size), 1.0)
        for _ in range(num_select):
            if remaining.numel() == 0:
                break
            score = support_scores.index_select(0, remaining)
            score = score + self.agreement_weight * agreement_scores.index_select(0, remaining)
            score = score - 0.5 * self.uncertainty_weight * uncertainty_scores.index_select(0, remaining)
            if anchor_feature is not None:
                candidate_features = F.normalize(token_features.index_select(0, remaining), dim=-1)
                affinity = (candidate_features * anchor_feature).sum(dim=-1)
                score = score + self.lockon_weight * affinity
            if anchor_coords is not None:
                candidate_coords = coords.index_select(0, remaining)
                dist2 = (candidate_coords - anchor_coords).pow(2).sum(dim=-1)
                proximity = torch.exp(-dist2 / (2.0 * sigma * sigma + self.eps))
                score = score + 0.5 * self.lockon_weight * proximity
            if dynamic_selected.numel() > 0 and self.visit_weight > 0:
                visit_penalty = self._compute_visit_penalty(
                    candidate_indices=remaining,
                    selected_indices=dynamic_selected,
                    coords=coords,
                    sigma=sigma,
                )
                score = score - self.visit_weight * visit_penalty

            pick_pos = int(score.argmax().item())
            picked = int(remaining[pick_pos].item())
            chosen.append(picked)
            picked_tensor = remaining.new_tensor([picked])
            dynamic_selected = torch.cat([dynamic_selected, picked_tensor], dim=0)
            keep_mask = torch.ones_like(remaining, dtype=torch.bool)
            keep_mask[pick_pos] = False
            remaining = remaining[keep_mask]
        return chosen

    def _compute_smell_field(self, scores: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
        if self.smell_weight <= 0:
            return scores
        batch = scores.size(0)
        grid_h, grid_w = grid_size
        score_map = scores.view(batch, 1, grid_h, grid_w)
        smell = F.avg_pool2d(score_map, kernel_size=3, stride=1, padding=1)
        return scores + self.smell_weight * smell.flatten(1)

    def _compute_lockon_field(
        self,
        smell_scores: torch.Tensor,
        local_scores: torch.Tensor,
        grid_size: tuple[int, int],
    ) -> torch.Tensor:
        if self.lockon_weight <= 0:
            return local_scores
        batch = local_scores.size(0)
        grid_h, grid_w = grid_size
        smell_map = smell_scores.view(batch, 1, grid_h, grid_w)
        local_map = local_scores.view(batch, 1, grid_h, grid_w)
        pooled_smell = F.avg_pool2d(smell_map, kernel_size=3, stride=1, padding=1)
        pooled_local = F.max_pool2d(local_map, kernel_size=3, stride=1, padding=1)
        return local_scores + self.lockon_weight * (0.5 * pooled_smell.flatten(1) + 0.5 * pooled_local.flatten(1))

    def _compute_detail_field(
        self,
        token_features: torch.Tensor,
        grid_size: tuple[int, int],
    ) -> torch.Tensor:
        batch, _, channels = token_features.shape
        grid_h, grid_w = grid_size
        feature_map = F.normalize(token_features.float(), dim=-1).view(batch, grid_h, grid_w, channels)
        detail_map = torch.zeros((batch, grid_h, grid_w), device=token_features.device, dtype=torch.float32)

        if grid_w > 1:
            diff_w = (feature_map[:, :, 1:, :] - feature_map[:, :, :-1, :]).pow(2).mean(dim=-1)
            detail_map[:, :, 1:] += diff_w
            detail_map[:, :, :-1] += diff_w
        if grid_h > 1:
            diff_h = (feature_map[:, 1:, :, :] - feature_map[:, :-1, :, :]).pow(2).mean(dim=-1)
            detail_map[:, 1:, :] += diff_h
            detail_map[:, :-1, :] += diff_h

        pooled = F.avg_pool2d(detail_map.unsqueeze(1), kernel_size=3, stride=1, padding=1).flatten(1)
        return self._unit_normalize_scores(pooled)

    def _topk_masked_indices(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        if k <= 0 or mask.sum().item() <= 0:
            return torch.empty(0, device=scores.device, dtype=torch.long)
        masked_scores = scores.masked_fill(~mask, float("-inf"))
        topk = min(k, int(mask.sum().item()))
        _, top_pos = masked_scores.topk(k=topk, dim=0)
        return top_pos.sort().values

    def _token_to_window_index(
        self,
        windows: torch.Tensor,
        valid: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        device = windows.device
        token_to_window = torch.full((num_tokens,), -1, dtype=torch.long, device=device)
        safe_windows = windows.masked_fill(~valid, 0)
        window_ids = torch.arange(windows.size(0), device=device, dtype=torch.long).unsqueeze(1).expand_as(safe_windows)
        token_to_window[safe_windows[valid]] = window_ids[valid]
        return token_to_window

    def _neighbor_window_indices(
        self,
        active_windows: torch.Tensor,
        grid_size: tuple[int, int],
        num_windows: int,
    ) -> torch.Tensor:
        device = active_windows.device
        if active_windows.numel() == 0 or num_windows == 0:
            return torch.empty(0, device=device, dtype=torch.long)
        window_coords = self._window_grid_coords(grid_size=grid_size, device=device)
        active_coords = window_coords.index_select(0, active_windows)
        chebyshev = (window_coords.unsqueeze(1) - active_coords.unsqueeze(0)).abs().amax(dim=-1)
        neighbor_mask = chebyshev.le(1).any(dim=1)
        neighbor_mask[active_windows] = False
        return torch.nonzero(neighbor_mask, as_tuple=False).squeeze(-1)

    def _token_grid_coords(self, grid_size: tuple[int, int], device: torch.device) -> torch.Tensor:
        grid_h, grid_w = grid_size
        rows = torch.arange(grid_h, device=device, dtype=torch.float32)
        cols = torch.arange(grid_w, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(rows, cols, indexing="ij")
        if grid_h > 1:
            yy = yy / float(grid_h - 1)
        if grid_w > 1:
            xx = xx / float(grid_w - 1)
        return torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)

    def _hierarchical_position_bases(
        self,
        grid_size: tuple[int, int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coords = self._token_grid_coords(grid_size=grid_size, device=device)
        yy = coords[:, 0]
        xx = coords[:, 1]
        center_x = 1.0 - (2.0 * xx - 1.0).abs()
        center_y = 1.0 - (2.0 * yy - 1.0).abs()

        coarse_basis = torch.stack(
            [
                xx,
                yy,
                center_x,
                center_y,
                torch.sin(math.pi * xx),
                torch.cos(math.pi * xx),
                torch.sin(math.pi * yy),
                torch.cos(math.pi * yy),
            ],
            dim=-1,
        )
        fine_basis = torch.stack(
            [
                torch.sin(2.0 * math.pi * xx),
                torch.cos(2.0 * math.pi * xx),
                torch.sin(2.0 * math.pi * yy),
                torch.cos(2.0 * math.pi * yy),
                torch.sin(4.0 * math.pi * xx),
                torch.cos(4.0 * math.pi * xx),
                torch.sin(4.0 * math.pi * yy),
                torch.cos(4.0 * math.pi * yy),
                xx * yy,
                (xx - yy).abs(),
            ],
            dim=-1,
        )
        return coarse_basis, fine_basis

    def _score_weighted_position_prior(
        self,
        scores: torch.Tensor,
        basis: torch.Tensor,
    ) -> torch.Tensor:
        weights = scores.float().clamp_min(0.0)
        weight_sum = weights.sum(dim=1, keepdim=True)
        if (weight_sum <= self.eps).any():
            weights = weights + 1.0
            weight_sum = weights.sum(dim=1, keepdim=True)
        weights = weights / weight_sum.clamp_min(self.eps)
        norm_basis = F.normalize(basis.float(), dim=-1)
        prototype = weights @ norm_basis
        prototype = F.normalize(prototype, dim=-1)
        prior = (norm_basis.unsqueeze(0) * prototype.unsqueeze(1)).sum(dim=-1)
        return 0.5 * (prior.clamp(min=-1.0, max=1.0) + 1.0)

    def _compute_hierarchical_position_priors(
        self,
        scores: torch.Tensor,
        grid_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coarse_basis, fine_basis = self._hierarchical_position_bases(
            grid_size=grid_size,
            device=scores.device,
        )
        coarse_prior = self._score_weighted_position_prior(scores=scores, basis=coarse_basis)
        fine_prior = self._score_weighted_position_prior(scores=scores, basis=fine_basis)
        return coarse_prior, fine_prior

    def _compute_anchor_position_prior(
        self,
        anchor_indices: torch.Tensor,
        grid_size: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        num_tokens = grid_size[0] * grid_size[1]
        if anchor_indices.numel() == 0:
            return torch.zeros(num_tokens, device=device, dtype=torch.float32)
        coords = self._token_grid_coords(grid_size=grid_size, device=device)
        anchor_coords = coords.index_select(0, anchor_indices.unique(sorted=True))
        dist = torch.cdist(coords, anchor_coords, p=2)
        nearest = dist.min(dim=1).values
        sigma = 0.18 + 0.06 * max(self.local_window_size - 1, 0)
        prior = torch.exp(-nearest.pow(2) / (2.0 * sigma * sigma + self.eps))
        prior = prior / prior.max().clamp_min(self.eps)
        return prior



    def _resolve_keep_count(self, num_tokens: int, num_windows: int, final_keep: Optional[int], keep_ratio: Optional[float]) -> int:
        target_ratio = self.default_keep_ratio if keep_ratio is None else keep_ratio
        ratio_keep = int(round(num_tokens * target_ratio))
        if final_keep is None or final_keep <= 0:
            keep_count = ratio_keep
        else:
            keep_count = final_keep
        keep_count = max(1, keep_count)
        keep_count = min(keep_count, num_tokens)
        return keep_count

    def _greedy_relevance_diverse_subset(
        self,
        candidate_indices: torch.Tensor,
        candidate_scores: torch.Tensor,
        token_features: torch.Tensor,
        grid_size: tuple[int, int],
        keep_count: int,
        fixed_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if candidate_indices.numel() <= keep_count:
            return candidate_indices.sort().values

        device = candidate_indices.device
        candidate_indices = candidate_indices.unique(sorted=True)
        candidate_scores = candidate_scores.index_select(0, candidate_indices)
        candidate_features = token_features.index_select(0, candidate_indices)
        candidate_features = F.normalize(candidate_features.float(), dim=-1)
        coords = self._token_grid_coords(grid_size=grid_size, device=device).index_select(0, candidate_indices)

        score_term = self._unit_normalize_scores(candidate_scores.unsqueeze(0))[0]
        feature_distance = 1.0 - candidate_features @ candidate_features.t()
        spatial_distance = torch.cdist(coords, coords, p=2)
        if spatial_distance.max().item() > self.eps:
            spatial_distance = spatial_distance / spatial_distance.max().clamp_min(self.eps)

        fixed_local: list[int] = []
        if fixed_indices is not None and fixed_indices.numel() > 0:
            fixed_set = set(int(idx) for idx in fixed_indices.tolist())
            for local_idx, global_idx in enumerate(candidate_indices.tolist()):
                if global_idx in fixed_set:
                    fixed_local.append(local_idx)

        selected_local = list(dict.fromkeys(fixed_local))
        selected_mask = torch.zeros(candidate_indices.numel(), device=device, dtype=torch.bool)
        if selected_local:
            selected_mask[torch.tensor(selected_local, device=device, dtype=torch.long)] = True

        while len(selected_local) < keep_count:
            remaining = torch.nonzero(~selected_mask, as_tuple=False).squeeze(-1)
            if remaining.numel() == 0:
                break
            if selected_local:
                selected_tensor = torch.tensor(selected_local, device=device, dtype=torch.long)
                min_feat_div = feature_distance.index_select(0, remaining).index_select(1, selected_tensor).min(dim=1).values
                min_spatial_div = spatial_distance.index_select(0, remaining).index_select(1, selected_tensor).min(dim=1).values
            else:
                min_feat_div = torch.ones(remaining.numel(), device=device, dtype=torch.float32)
                min_spatial_div = torch.ones(remaining.numel(), device=device, dtype=torch.float32)

            combined = 0.60 * score_term.index_select(0, remaining)
            combined = combined + 0.25 * min_feat_div + 0.15 * min_spatial_div
            best_pos = int(combined.argmax().item())
            next_local = int(remaining[best_pos].item())
            selected_local.append(next_local)
            selected_mask[next_local] = True

        selected_tensor = torch.tensor(selected_local[:keep_count], device=device, dtype=torch.long)
        return candidate_indices.index_select(0, selected_tensor).sort().values

    def _foraging_select(
        self,
        keep_count: int,
        windows: torch.Tensor,
        valid: torch.Tensor,
        smell_scores: torch.Tensor,
        lockon_scores: torch.Tensor,
        jump_scores: torch.Tensor,
        task_scores: torch.Tensor,
        detail_scores: torch.Tensor,
        token_features: torch.Tensor,
        grid_size: tuple[int, int],
        anchor_prior: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = smell_scores.device
        num_tokens = smell_scores.numel()
        num_windows = windows.size(0)
        safe_windows = windows.masked_fill(~valid, 0)
        token_to_window = self._token_to_window_index(windows=windows, valid=valid, num_tokens=num_tokens)
        window_capacity = valid.sum(dim=-1).long().clamp_min(1)
        ivc_indices = self._select_ivc_indices(
            num_tokens=num_tokens,
            keep_count=keep_count,
            device=device,
        )
        ivc_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        if ivc_indices.numel() > 0:
            ivc_mask[ivc_indices] = True

        smell_map = smell_scores.view(1, -1)
        smell_window_strength = self._window_stats(smell_map, windows, valid)[0]
        if ivc_indices.numel() > 0 and self.ivc_window_bonus > self.eps:
            ivc_window_count = ivc_mask[safe_windows].sum(dim=-1).float()
            ivc_window_strength = ivc_window_count / window_capacity.float().clamp_min(1.0)
            smell_window_strength = smell_window_strength + self.ivc_window_bonus * ivc_window_strength
        avg_window_capacity = max(float(window_capacity.float().mean().item()), 1.0)
        coarse_window_budget = int(round(keep_count / max(avg_window_capacity * 1.15, 1.0)))
        coarse_window_budget = max(2, min(coarse_window_budget, num_windows, keep_count))

        _, active_pos = smell_window_strength.topk(k=coarse_window_budget, dim=0)
        active_windows = active_pos.sort().values
        context_windows = self._neighbor_window_indices(
            active_windows=active_windows,
            grid_size=grid_size,
            num_windows=num_windows,
        )
        anchor_windows = (
            token_to_window.index_select(0, ivc_indices).unique(sorted=True)
            if ivc_indices.numel() > 0
            else torch.empty(0, device=device, dtype=torch.long)
        )

        active_mask = torch.zeros(num_windows, dtype=torch.bool, device=device)
        active_mask[active_windows] = True
        context_mask = torch.zeros(num_windows, dtype=torch.bool, device=device)
        if context_windows.numel() > 0:
            context_mask[context_windows] = True
        anchor_mask = torch.zeros(num_windows, dtype=torch.bool, device=device)
        if anchor_windows.numel() > 0:
            anchor_mask[anchor_windows] = True
        candidate_window_mask = active_mask | context_mask | anchor_mask

        smell_bank = smell_scores[safe_windows.index_select(0, active_windows)]
        lockon_bank = lockon_scores[safe_windows.index_select(0, active_windows)]
        scaffold_bank = 0.55 * smell_bank + 0.45 * lockon_bank
        scaffold_bank = scaffold_bank.masked_fill(~valid.index_select(0, active_windows), float("-inf"))
        local_keep = min(self.scaffold_keep, smell_bank.size(-1))
        _, scaffold_pos = scaffold_bank.topk(k=local_keep, dim=-1)
        scaffold_indices = (
            safe_windows.index_select(0, active_windows).gather(dim=-1, index=scaffold_pos).reshape(-1).unique(sorted=True)
        )

        selected_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        if ivc_indices.numel() > 0:
            selected_mask[ivc_indices] = True
        selected_mask[scaffold_indices] = True

        fixed_keep_count = int(selected_mask.sum().item())
        remaining_budget = max(0, keep_count - fixed_keep_count)
        jump_keep = min(remaining_budget, int(round(keep_count * self.jump_ratio)))
        exploit_keep = remaining_budget - jump_keep

        token_window_bonus = smell_window_strength.index_select(0, token_to_window.clamp_min(0))
        candidate_mask = candidate_window_mask.index_select(0, token_to_window.clamp_min(0))
        candidate_mask = candidate_mask & ~selected_mask
        exploit_scores = lockon_scores + 0.15 * token_window_bonus + 0.20 * task_scores
        if anchor_prior is not None and self.anchor_pos_weight > self.eps:
            exploit_scores = exploit_scores + self.anchor_pos_weight * anchor_prior
        exploit_indices = self._topk_masked_indices(exploit_scores, candidate_mask, exploit_keep)
        if exploit_indices.numel() > 0:
            selected_mask[exploit_indices] = True

        selected_per_window = selected_mask[safe_windows].sum(dim=-1).float()
        coverage_deficit = 1.0 - selected_per_window / window_capacity.float()
        coverage_bonus = coverage_deficit.index_select(0, token_to_window.clamp_min(0))
        outside_bonus = (~candidate_window_mask).float().index_select(0, token_to_window.clamp_min(0))
        jump_mask = ~selected_mask

        detail_keep = min(jump_keep, max(0, int(round(keep_count * 0.05))))
        detail_rescue_scores = 0.55 * jump_scores + 0.30 * detail_scores + 0.25 * task_scores + 0.15 * coverage_bonus
        detail_indices = self._topk_masked_indices(detail_rescue_scores, jump_mask, detail_keep)
        if detail_indices.numel() > 0:
            selected_mask[detail_indices] = True
            jump_mask = ~selected_mask

        jump_remaining = max(0, jump_keep - int(detail_indices.numel()))
        jump_values = jump_scores + 0.50 * coverage_bonus + 0.20 * outside_bonus + 0.35 * task_scores
        if anchor_prior is not None and self.anchor_pos_weight > self.eps:
            jump_values = jump_values + self.anchor_pos_weight * (1.25 * anchor_prior + 0.10 * coverage_bonus)
        jump_indices = self._topk_masked_indices(jump_values, jump_mask, jump_remaining)
        if detail_indices.numel() > 0:
            jump_indices = torch.cat([detail_indices, jump_indices], dim=0).unique(sorted=True)
        if jump_indices.numel() > 0:
            selected_mask[jump_indices] = True

        fill_mask = ~selected_mask
        fill_scores = lockon_scores + 0.25 * coverage_bonus + 0.20 * task_scores
        if anchor_prior is not None and self.anchor_pos_weight > self.eps:
            fill_scores = fill_scores + 0.50 * self.anchor_pos_weight * anchor_prior
        fill_keep = max(0, keep_count - int(selected_mask.sum().item()))
        fill_indices = self._topk_masked_indices(fill_scores, fill_mask, fill_keep)
        if fill_indices.numel() > 0:
            selected_mask[fill_indices] = True

        selected_indices = torch.nonzero(selected_mask, as_tuple=False).squeeze(-1).sort().values
        if selected_indices.numel() > 0:
            candidate_seed_scores = lockon_scores + 0.35 * jump_scores + 0.20 * task_scores + 0.10 * detail_scores
            if anchor_prior is not None and self.anchor_pos_weight > self.eps:
                candidate_seed_scores = candidate_seed_scores + 0.50 * self.anchor_pos_weight * anchor_prior
            candidate_keep = min(num_tokens, max(keep_count * 3, keep_count + 32))
            _, candidate_pos = candidate_seed_scores.topk(k=candidate_keep, dim=0)
            candidate_indices = torch.cat([selected_indices, candidate_pos, scaffold_indices, ivc_indices], dim=0).unique(sorted=True)
            selected_indices = self._greedy_relevance_diverse_subset(
                candidate_indices=candidate_indices,
                candidate_scores=candidate_seed_scores,
                token_features=token_features,
                grid_size=grid_size,
                keep_count=keep_count,
                fixed_indices=torch.cat([scaffold_indices, ivc_indices], dim=0).unique(sorted=True),
            )
            selected_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
            selected_mask[selected_indices] = True

        if selected_indices.numel() > keep_count:
            trim_scores = (lockon_scores + 0.25 * jump_scores).index_select(0, selected_indices)
            _, top_pos = trim_scores.topk(k=keep_count, dim=0)
            selected_indices = selected_indices.index_select(0, top_pos).sort().values
            selected_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
            selected_mask[selected_indices] = True

        exploit_indices = exploit_indices[selected_mask[exploit_indices]]
        jump_indices = jump_indices[selected_mask[jump_indices]]
        fill_indices = fill_indices[selected_mask[fill_indices]]
        return selected_indices, {
            "scaffold_indices": scaffold_indices[selected_mask[scaffold_indices]],
            "exploit_indices": exploit_indices,
            "jump_indices": jump_indices,
            "fill_indices": fill_indices,
            "active_window_indices": active_windows,
            "context_window_indices": context_windows,
            "anchor_window_indices": anchor_windows,
            "ivc_indices": ivc_indices[selected_mask[ivc_indices]] if ivc_indices.numel() > 0 else ivc_indices,
        }



    def _divprune_official_indices(self, token_features: torch.Tensor, keep_count: int) -> torch.Tensor:
        num_tokens = token_features.size(0)
        if keep_count >= num_tokens:
            return torch.arange(num_tokens, device=token_features.device, dtype=torch.long)

        norm_tokens = F.normalize(token_features.float(), dim=-1)
        cosine_distance = 1.0 - (norm_tokens @ norm_tokens.t())
        selected = torch.empty(keep_count, device=token_features.device, dtype=torch.long)

        for step in range(keep_count):
            if step == 0:
                current = cosine_distance
                top2 = torch.topk(current, k=min(2, num_tokens), dim=0, largest=False).values
                if top2.size(0) == 1:
                    scores = top2[0]
                else:
                    scores = top2[1]
            else:
                chosen = selected[:step]
                current = torch.index_select(cosine_distance, 0, chosen)
                scores = current.min(dim=0).values

            next_idx = int(scores.argmax().item())
            selected[step] = next_idx

        return selected

    def _greedy_dpp_indices(self, kernel: torch.Tensor, keep_count: int) -> torch.Tensor:
        num_tokens = kernel.size(0)
        if keep_count >= num_tokens:
            return torch.arange(num_tokens, device=kernel.device, dtype=torch.long)

        cis = torch.zeros((keep_count, num_tokens), device=kernel.device, dtype=kernel.dtype)
        di2s = kernel.diag().clone()
        selected: list[int] = []

        while len(selected) < keep_count:
            next_idx = int(di2s.argmax().item())
            if not torch.isfinite(di2s[next_idx]) or di2s[next_idx].item() <= self.eps:
                break
            selected.append(next_idx)
            current = len(selected) - 1
            if current == keep_count - 1:
                break
            denom = torch.sqrt(di2s[next_idx].clamp_min(self.eps))
            if current == 0:
                eis = kernel[next_idx] / denom
            else:
                projection = torch.matmul(cis[:current, next_idx], cis[:current])
                eis = (kernel[next_idx] - projection) / denom
            cis[current] = eis
            di2s = di2s - eis.pow(2)
            di2s[selected] = float("-inf")

        if len(selected) < keep_count:
            remaining_mask = torch.ones(num_tokens, device=kernel.device, dtype=torch.bool)
            if selected:
                remaining_mask[torch.tensor(selected, device=kernel.device, dtype=torch.long)] = False
            remaining = torch.nonzero(remaining_mask, as_tuple=False).squeeze(-1)
            fill_count = min(keep_count - len(selected), int(remaining.numel()))
            if fill_count > 0:
                fill_scores = kernel.diag().index_select(0, remaining)
                _, fill_pos = fill_scores.topk(k=fill_count, dim=0)
                selected.extend(remaining.index_select(0, fill_pos).tolist())

        return torch.tensor(selected, device=kernel.device, dtype=torch.long)

    def _forward_divprune(
        self,
        visual_tokens: torch.Tensor,
        grid_size: tuple[int, int],
        final_keep: Optional[int],
        keep_ratio: Optional[float],
    ) -> RoutingResult:
        token_features = F.normalize(visual_tokens[0].float(), dim=-1)
        num_tokens = token_features.size(0)
        windows = _window_indices(grid_size[0], grid_size[1], self.local_window_size, visual_tokens.device)
        keep_count = self._resolve_keep_count(
            num_tokens=num_tokens,
            num_windows=windows.size(0),
            final_keep=final_keep,
            keep_ratio=keep_ratio,
        )
        pairwise_distance = 1.0 - token_features @ token_features.t()
        pairwise_distance = pairwise_distance.masked_fill(
            torch.eye(num_tokens, device=visual_tokens.device, dtype=torch.bool),
            0.0,
        )
        diversity_scores = self._unit_normalize_scores(pairwise_distance.mean(dim=1, keepdim=True).t())[0]
        selected_indices = self._divprune_official_indices(token_features=token_features, keep_count=keep_count)
        return self._build_simple_result(selected_indices=selected_indices, scores=diversity_scores, grid_size=grid_size)

    def _forward_cdprune(
        self,
        visual_tokens: torch.Tensor,
        odor_cue: torch.Tensor,
        grid_size: tuple[int, int],
        final_keep: Optional[int],
        keep_ratio: Optional[float],
    ) -> RoutingResult:
        token_features = visual_tokens[0].float()
        normalized_tokens = F.normalize(token_features, dim=-1)
        num_tokens = token_features.size(0)
        windows = _window_indices(grid_size[0], grid_size[1], self.local_window_size, visual_tokens.device)
        keep_count = self._resolve_keep_count(
            num_tokens=num_tokens,
            num_windows=windows.size(0),
            final_keep=final_keep,
            keep_ratio=keep_ratio,
        )

        relevance = self._compute_cross_modal_similarity(
            token_features=visual_tokens.float(),
            odor_cue=odor_cue.float(),
        )[0]
        relevance = self._unit_normalize_scores(relevance.unsqueeze(0))[0]
        if relevance.max().item() <= self.eps:
            relevance = torch.ones_like(relevance)
        similarity = normalized_tokens @ normalized_tokens.t()
        kernel = relevance.unsqueeze(1) * similarity * relevance.unsqueeze(0)
        selected_indices = self._greedy_dpp_indices(kernel=kernel, keep_count=keep_count)
        return self._build_simple_result(selected_indices=selected_indices, scores=relevance, grid_size=grid_size)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        odor_cue: torch.Tensor,
        grid_size: tuple[int, int],
        hypothesis_cues: Optional[torch.Tensor] = None,
        contrast_cues: Optional[torch.Tensor] = None,
        final_keep: Optional[int] = None,
        keep_ratio: Optional[float] = None,
    ) -> RoutingResult:
        if visual_tokens.dim() != 3:
            raise ValueError("visual_tokens must have shape [B, N, D]")
        if odor_cue.dim() != 2:
            raise ValueError("odor_cue must have shape [B, D]")
        batch, num_tokens, token_dim = visual_tokens.shape
        if batch != 1:
            raise NotImplementedError("The current Qwen wrapper supports one image at a time")
        if token_dim != self.token_dim:
            raise ValueError(f"Expected token dim {self.token_dim}, got {token_dim}")
        if odor_cue.size(-1) != self.odor_dim:
            raise ValueError(f"Expected odor dim {self.odor_dim}, got {odor_cue.size(-1)}")
        if hypothesis_cues is not None and (
            hypothesis_cues.dim() != 3 or hypothesis_cues.size(0) != batch or hypothesis_cues.size(-1) != self.odor_dim
        ):
            raise ValueError("hypothesis_cues must have shape [B, C, D] with D equal to odor_dim")
        if contrast_cues is not None and (
            contrast_cues.dim() != 3 or contrast_cues.size(0) != batch or contrast_cues.size(-1) != self.odor_dim
        ):
            raise ValueError("contrast_cues must have shape [B, C, D] with D equal to odor_dim")

        grid_h, grid_w = grid_size
        if grid_h * grid_w != num_tokens:
            raise ValueError("grid_size does not match the number of visual tokens")

        if self.routing_mode == "divprune":
            return self._forward_divprune(
                visual_tokens=visual_tokens,
                grid_size=grid_size,
                final_keep=final_keep,
                keep_ratio=keep_ratio,
            )
        if self.routing_mode == "cdprune":
            return self._forward_cdprune(
                visual_tokens=visual_tokens,
                odor_cue=odor_cue,
                grid_size=grid_size,
                final_keep=final_keep,
                keep_ratio=keep_ratio,
            )

        token_features = self._normalize_tokens(visual_tokens.float())
        visual_head_logits = self._compute_visual_head_logits(token_features)
        visual_head_probs = self._compute_visual_head_probs(visual_head_logits)
        query_scores_all, query_odor_logits_all, query_odor_gate_all = self._compute_cue_scores(
            visual_head_probs=visual_head_probs,
            odor_cues=odor_cue.unsqueeze(1),
        )
        query_scores = query_scores_all[:, 0, :]
        odor_head_logits = query_odor_logits_all[:, 0, :]
        odor_head_gate = query_odor_gate_all[:, 0, :]
        head_logits = visual_head_probs * odor_head_gate.unsqueeze(1)
        query_alignment_scores = self._compute_cross_modal_similarity(
            token_features=token_features,
            odor_cue=odor_cue,
        )

        query_norm = self._unit_normalize_scores(query_scores)
        query_alignment_norm = self._unit_normalize_scores(query_alignment_scores)
        query_coarse_scores = (
            (1.0 - self.cross_similarity_weight) * query_norm
            + self.cross_similarity_weight * query_alignment_norm
        )
        coarse_pos_prior, fine_pos_prior = self._compute_hierarchical_position_priors(
            scores=query_coarse_scores,
            grid_size=grid_size,
        )
        if self.coarse_pos_weight > self.eps:
            query_coarse_scores = query_coarse_scores + self.coarse_pos_weight * coarse_pos_prior
            query_coarse_scores = self._unit_normalize_scores(query_coarse_scores)
        base_scores = query_coarse_scores
        hypothesis_scores = None
        contrast_scores = None
        agreement_scores = None
        repulsion_scores = None
        uncertainty_scores = None
        region_agreement_scores = None

        if hypothesis_cues is not None and hypothesis_cues.size(1) > 0:
            hypothesis_score_bank, _, hypothesis_odor_gate = self._compute_cue_scores(
                visual_head_probs=visual_head_probs,
                odor_cues=hypothesis_cues,
            )
            top_support, _ = hypothesis_score_bank.topk(k=min(2, hypothesis_score_bank.size(1)), dim=1)
            hypothesis_scores = top_support[:, 0, :]
            hypothesis_norm = self._unit_normalize_scores(hypothesis_scores)
            best_hypothesis_idx = hypothesis_score_bank.argmax(dim=1)
            competitor_scores = torch.zeros_like(hypothesis_scores)
            margin_scores = top_support[:, 0, :]
            if hypothesis_score_bank.size(1) > 1:
                competitor_scores = top_support[:, 1, :]
                margin_scores = top_support[:, 0, :] - top_support[:, 1, :]
            competitor_norm = self._unit_normalize_scores(competitor_scores)
            margin_norm = self._positive_normalize_scores(margin_scores)
            best_hypothesis_gate = hypothesis_odor_gate.gather(
                dim=1,
                index=best_hypothesis_idx.unsqueeze(-1).expand(batch, num_tokens, self.num_heads),
            )
            agreement_scores = self._compute_agreement_score(
                visual_head_probs=visual_head_probs,
                query_head_gate=odor_head_gate,
                hypothesis_head_gate=best_hypothesis_gate,
                margin_norm=margin_norm,
            )
            uncertainty_scores = (1.0 - margin_norm).clamp(0.0, 1.0)
            repulsion_scores = competitor_norm
            base_scores = (
                base_scores
                + self.hypothesis_weight * hypothesis_norm
                + self.agreement_weight * agreement_scores
                - 0.25 * self.repulsion_weight * competitor_norm
            )
        else:
            agreement_scores = (visual_head_probs * odor_head_gate.unsqueeze(1)).sum(dim=-1)
            uncertainty_scores = 1.0 - agreement_scores

        if contrast_cues is not None and contrast_cues.size(1) > 0:
            contrast_score_bank, _, _ = self._compute_signed_cue_scores(
                visual_head_logits=visual_head_logits,
                odor_cues=contrast_cues,
            )
            contrast_scores = contrast_score_bank.amax(dim=1)
            contrast_norm = self._positive_normalize_scores(contrast_scores)
            contrast_repulsion = self._positive_normalize_scores((-contrast_score_bank).amax(dim=1))
            if repulsion_scores is None:
                repulsion_scores = contrast_repulsion
            else:
                repulsion_scores = 0.5 * (repulsion_scores + contrast_repulsion)
            uncertainty_scores = (
                contrast_repulsion
                if uncertainty_scores is None
                else 0.5 * (uncertainty_scores + contrast_repulsion)
            )
            base_scores = (
                base_scores
                + self.contrast_weight * contrast_norm
                - 0.5 * self.repulsion_weight * contrast_repulsion
            )

        detail_scores = self._compute_detail_field(token_features=token_features, grid_size=grid_size)
        smell_scores = self._compute_smell_field(query_coarse_scores, grid_size=grid_size)
        lockon_base_scores = smell_scores.clone()
        task_scores = agreement_scores.clone() if agreement_scores is not None else query_coarse_scores.clone()
        if hypothesis_scores is not None:
            hypothesis_norm = self._unit_normalize_scores(hypothesis_scores)
            lockon_base_scores = lockon_base_scores + self.hypothesis_weight * hypothesis_norm
            task_scores = 0.5 * task_scores + 0.5 * hypothesis_norm
        if agreement_scores is not None:
            lockon_base_scores = lockon_base_scores + self.agreement_weight * agreement_scores
            task_scores = 0.6 * task_scores + 0.4 * agreement_scores
        if repulsion_scores is not None:
            lockon_base_scores = lockon_base_scores - 0.25 * self.repulsion_weight * repulsion_scores
        task_detail_scores = detail_scores * (0.4 + 0.6 * task_scores)
        lockon_base_scores = lockon_base_scores + 0.20 * task_detail_scores
        if self.local_pos_weight > self.eps:
            local_gate = 0.35 + 0.65 * task_scores
            lockon_base_scores = lockon_base_scores + self.local_pos_weight * fine_pos_prior * local_gate
        lockon_scores = self._compute_lockon_field(
            smell_scores=smell_scores,
            local_scores=lockon_base_scores,
            grid_size=grid_size,
        )
        jump_scores = lockon_scores.clone()
        if uncertainty_scores is not None:
            jump_scores = jump_scores + self.uncertainty_weight * uncertainty_scores
        if contrast_scores is not None:
            contrast_norm = self._positive_normalize_scores(contrast_scores)
            jump_scores = jump_scores + 0.5 * self.contrast_weight * contrast_norm
        jump_scores = jump_scores + 0.35 * task_detail_scores
        route_scores = jump_scores

        windows = _window_indices(grid_h, grid_w, self.local_window_size, visual_tokens.device)
        num_windows = windows.size(0)
        keep_count = self._resolve_keep_count(
            num_tokens=num_tokens,
            num_windows=num_windows,
            final_keep=final_keep,
            keep_ratio=keep_ratio,
        )

        valid = windows >= 0
        region_agreement_scores = self._window_stats(agreement_scores, windows, valid)
        ivc_anchor_prior = None
        if self.anchor_pos_weight > self.eps:
            ivc_seed_indices = self._select_ivc_indices(
                num_tokens=num_tokens,
                keep_count=keep_count,
                device=visual_tokens.device,
            )
            ivc_anchor_prior = self._compute_anchor_position_prior(
                anchor_indices=ivc_seed_indices,
                grid_size=grid_size,
                device=visual_tokens.device,
            )
            jump_scores = jump_scores + 0.35 * self.anchor_pos_weight * ivc_anchor_prior

        if self.routing_mode == "foraging":
            selected_indices_1d, stage_indices = self._foraging_select(
                keep_count=keep_count,
                windows=windows,
                valid=valid,
                smell_scores=smell_scores[0],
                lockon_scores=lockon_scores[0],
                jump_scores=jump_scores[0],
                task_scores=task_scores[0],
                detail_scores=detail_scores[0],
                token_features=token_features[0],
                grid_size=grid_size,
                anchor_prior=ivc_anchor_prior,
            )
            selected_indices = selected_indices_1d.unsqueeze(0)
            scaffold_indices = stage_indices["scaffold_indices"]
            exploit_indices = stage_indices["exploit_indices"]
            jump_indices = stage_indices["jump_indices"]
            fill_indices = stage_indices["fill_indices"]
            active_window_indices = stage_indices["active_window_indices"]
            context_window_indices = stage_indices["context_window_indices"]
            anchor_window_indices = stage_indices["anchor_window_indices"]
            ivc_indices = stage_indices["ivc_indices"]
        else:
            safe_windows = windows.masked_fill(~valid, 0)
            window_scores = smell_scores[:, safe_windows]
            window_scores = window_scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
            local_keep = min(self.scaffold_keep, window_scores.size(-1))
            _, local_pos = window_scores.topk(k=local_keep, dim=-1)
            scaffold_indices = safe_windows.unsqueeze(0).expand(batch, -1, -1).gather(dim=-1, index=local_pos)
            scaffold_indices = scaffold_indices.reshape(batch, -1)

            selected_mask = torch.zeros(batch, num_tokens, dtype=torch.bool, device=visual_tokens.device)
            selected_mask.scatter_(dim=1, index=scaffold_indices, value=True)

            extra_keep = max(0, keep_count - scaffold_indices.size(1))
            if extra_keep > 0:
                remaining_scores = jump_scores.masked_fill(selected_mask, float("-inf"))
                _, extra_indices = remaining_scores.topk(k=extra_keep, dim=-1)
                selected_indices = torch.cat([scaffold_indices, extra_indices], dim=-1)
            else:
                selected_indices = scaffold_indices
            scaffold_indices = scaffold_indices[0]
            exploit_indices = torch.empty(0, device=visual_tokens.device, dtype=torch.long)
            jump_indices = torch.empty(0, device=visual_tokens.device, dtype=torch.long)
            fill_indices = torch.empty(0, device=visual_tokens.device, dtype=torch.long)
            active_window_indices = torch.empty(0, device=visual_tokens.device, dtype=torch.long)
            context_window_indices = torch.empty(0, device=visual_tokens.device, dtype=torch.long)
            anchor_window_indices = torch.empty(0, device=visual_tokens.device, dtype=torch.long)
            ivc_indices = torch.empty(0, device=visual_tokens.device, dtype=torch.long)

        selected_indices, _ = selected_indices.sort(dim=-1)
        selected_scores = jump_scores.gather(dim=1, index=selected_indices)

        return RoutingResult(
            selected_indices=selected_indices[0],
            selected_scores=selected_scores[0],
            base_scores=smell_scores[0],
            route_scores=route_scores[0],
            query_scores=smell_scores[0],
            hypothesis_scores=None if hypothesis_scores is None else hypothesis_scores[0],
            contrast_scores=None if contrast_scores is None else contrast_scores[0],
            agreement_scores=lockon_scores[0],
            repulsion_scores=None if repulsion_scores is None else repulsion_scores[0],
            uncertainty_scores=None if uncertainty_scores is None else uncertainty_scores[0],
            region_agreement_scores=None if region_agreement_scores is None else region_agreement_scores[0],
            odor_head_gate=odor_head_gate[0],
            head_logits=head_logits[0],
            kept_count=int(selected_indices.size(1)),
            grid_size=grid_size,
            scaffold_indices=scaffold_indices,
            exploit_indices=exploit_indices,
            jump_indices=jump_indices,
            fill_indices=fill_indices,
            active_window_indices=active_window_indices,
            context_window_indices=context_window_indices,
            anchor_window_indices=anchor_window_indices,
            ivc_indices=ivc_indices,
        )
