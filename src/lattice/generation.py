from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EpisodeTable:
    game_ids: torch.Tensor
    start_slots: torch.Tensor
    lengths: torch.Tensor
    outcomes: torch.Tensor


def sample_actions(
    q_scores: torch.Tensor,
    legal_mask: torch.Tensor,
    temperature: float,
    epsilon: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        greedy = q_scores.masked_fill(~legal_mask, -torch.inf).argmax(dim=-1)
        return greedy

    masked = q_scores.masked_fill(~legal_mask, -torch.inf)
    policy = torch.softmax(masked / temperature, dim=-1)
    uniform = legal_mask.float()
    uniform = uniform / uniform.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mixed = (1.0 - epsilon) * policy + epsilon * uniform
    return torch.multinomial(mixed, num_samples=1, generator=generator).squeeze(-1)


def allocate_episode_table(num_games: int, device: torch.device | str) -> EpisodeTable:
    dev = torch.device(device)
    return EpisodeTable(
        game_ids=torch.arange(num_games, device=dev, dtype=torch.int64),
        start_slots=torch.zeros(num_games, device=dev, dtype=torch.int64),
        lengths=torch.zeros(num_games, device=dev, dtype=torch.int32),
        outcomes=torch.zeros(num_games, device=dev),
    )


def backfill_outcomes(
    wdl_targets: torch.Tensor,
    record_game_ids: torch.Tensor,
    finished_game_ids: torch.Tensor,
    outcomes: torch.Tensor,
    outcome_weight: float = 0.4,
) -> torch.Tensor:
    updated = wdl_targets.clone()
    for game_id, outcome in zip(finished_game_ids, outcomes, strict=True):
        mask = record_game_ids == game_id
        updated[mask] = (1.0 - outcome_weight) * updated[mask] + outcome_weight * outcome
    return updated


def generate_opening_seed_indices(
    seed_count: int,
    ply_min: int,
    ply_max: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if ply_min < 0 or ply_max < ply_min:
        raise ValueError("invalid opening ply range.")
    return torch.randint(ply_min, ply_max + 1, (seed_count,), device=device, generator=generator)
