from __future__ import annotations

from dataclasses import dataclass

import torch

from lattice.model import LatticeModel, score_moves


@dataclass(frozen=True)
class EvaluationResult:
    name: str
    positions: int
    score: float
    checkpoint_eligible: bool


def searchless_select(
    model: LatticeModel,
    features: torch.Tensor,
    from_squares: torch.Tensor,
    to_squares: torch.Tensor,
    legal_mask: torch.Tensor,
    promotions: torch.Tensor | None = None,
) -> torch.Tensor:
    output = model(features)
    scores = score_moves(output, from_squares, to_squares, promotions)
    scores = scores.masked_fill(~legal_mask, -torch.inf)
    return scores.argmax(dim=-1)


def checkpoint_gate(name: str, scores: list[float], min_score: float = 0.0) -> EvaluationResult:
    if not scores:
        raise ValueError("scores must not be empty.")
    score = sum(scores) / len(scores)
    return EvaluationResult(
        name=name,
        positions=len(scores),
        score=score,
        checkpoint_eligible=score >= min_score,
    )
