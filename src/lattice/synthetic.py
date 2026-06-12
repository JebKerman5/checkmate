from __future__ import annotations

from dataclasses import dataclass

import torch

from lattice.config import LatticeConfig


@dataclass(frozen=True)
class SyntheticReplayBatch:
    features: torch.Tensor
    from_squares: torch.Tensor
    to_squares: torch.Tensor
    promotions: torch.Tensor
    q_targets: torch.Tensor
    q_mask: torch.Tensor
    wdl_targets: torch.Tensor
    value_targets: torch.Tensor
    moves_left_targets: torch.Tensor
    weights: torch.Tensor


def make_synthetic_replay_batch(
    cfg: LatticeConfig,
    device: torch.device | str,
    batch_size: int | None = None,
    moves_per_position: int | None = None,
    generator: torch.Generator | None = None,
) -> SyntheticReplayBatch:
    batch = batch_size or cfg.replay.learner_batch
    moves = moves_per_position or (cfg.labeling.top_k + cfg.labeling.random_children)
    dev = torch.device(device)

    features = torch.randn(
        batch,
        cfg.model.squares,
        cfg.model.input_features,
        device=dev,
        generator=generator,
    )
    from_squares = torch.randint(
        0,
        cfg.model.squares,
        (batch, moves),
        device=dev,
        generator=generator,
    )
    to_squares = torch.randint(
        0,
        cfg.model.squares,
        (batch, moves),
        device=dev,
        generator=generator,
    )
    promotions = torch.full((batch, moves), -1, device=dev, dtype=torch.long)
    q_targets = torch.empty(batch, moves, device=dev).uniform_(-1.0, 1.0, generator=generator)
    q_mask = torch.ones(batch, moves, device=dev, dtype=torch.bool)
    wdl_targets = torch.randint(0, 3, (batch,), device=dev, generator=generator)
    value_targets = torch.empty(batch, device=dev).uniform_(-1.0, 1.0, generator=generator)
    moves_left_targets = torch.empty(batch, device=dev).uniform_(1.0, 120.0, generator=generator)
    weights = torch.ones(batch, device=dev)

    return SyntheticReplayBatch(
        features=features,
        from_squares=from_squares,
        to_squares=to_squares,
        promotions=promotions,
        q_targets=q_targets,
        q_mask=q_mask,
        wdl_targets=wdl_targets,
        value_targets=value_targets,
        moves_left_targets=moves_left_targets,
        weights=weights,
    )
