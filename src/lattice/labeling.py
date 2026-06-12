from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SparseTargets:
    from_squares: torch.Tensor
    to_squares: torch.Tensor
    promotions: torch.Tensor
    q_targets: torch.Tensor
    q_mask: torch.Tensor
    value_targets: torch.Tensor
    wdl_targets: torch.Tensor


def choose_sparse_expansions(
    q_scores: torch.Tensor,
    legal_mask: torch.Tensor,
    top_k: int,
    random_children: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if q_scores.shape != legal_mask.shape:
        raise ValueError("q_scores and legal_mask must have the same shape.")

    masked = q_scores.masked_fill(~legal_mask, -torch.inf)
    top = masked.topk(k=top_k, dim=-1).indices

    probs = legal_mask.float()
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1.0)
    random = torch.multinomial(
        probs,
        num_samples=random_children,
        replacement=True,
        generator=generator,
    )
    return torch.cat([top, random], dim=-1)


def build_negamax_targets(
    child_values: torch.Tensor,
    child_terminal_values: torch.Tensor,
    terminal_mask: torch.Tensor,
    outcome_wdl: torch.Tensor,
    consistency_weight: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if child_values.shape != child_terminal_values.shape:
        raise ValueError("child value tensors must match.")

    child_target = torch.where(terminal_mask, child_terminal_values, -child_values)
    value_target = child_target.max(dim=-1).values
    wdl_target = consistency_weight * value_target + (1.0 - consistency_weight) * outcome_wdl
    return value_target, wdl_target


def apply_terminal_overrides(
    child_values: torch.Tensor,
    terminal_values: torch.Tensor,
    terminal_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.where(terminal_mask, terminal_values, child_values)
