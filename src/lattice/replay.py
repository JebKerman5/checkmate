from __future__ import annotations

from dataclasses import dataclass

import torch

from lattice.config import LatticeConfig
from lattice.synthetic import SyntheticReplayBatch


@dataclass(frozen=True)
class ReplayFreshness:
    filled: int
    capacity: int
    fill_ratio: float
    write_index: int
    expected_reuse: int


@dataclass
class ReplayShard:
    features: torch.Tensor
    from_squares: torch.Tensor
    to_squares: torch.Tensor
    promotions: torch.Tensor
    q_targets: torch.Tensor
    q_mask: torch.Tensor
    wdl_targets: torch.Tensor
    value_targets: torch.Tensor
    moves_left_targets: torch.Tensor
    priorities: torch.Tensor
    write_index: int = 0
    filled: int = 0

    @property
    def capacity(self) -> int:
        return int(self.features.shape[0])

    def freshness(self, expected_reuse: int) -> ReplayFreshness:
        return ReplayFreshness(
            filled=self.filled,
            capacity=self.capacity,
            fill_ratio=self.filled / max(self.capacity, 1),
            write_index=self.write_index,
            expected_reuse=expected_reuse,
        )


def allocate_replay_shard(
    cfg: LatticeConfig,
    device: torch.device | str,
    capacity: int | None = None,
) -> ReplayShard:
    cap = capacity or cfg.replay.records
    moves = cfg.labeling.top_k + cfg.labeling.random_children
    dev = torch.device(device)

    return ReplayShard(
        features=torch.zeros(cap, cfg.model.squares, cfg.model.input_features, device=dev),
        from_squares=torch.zeros(cap, moves, dtype=torch.long, device=dev),
        to_squares=torch.zeros(cap, moves, dtype=torch.long, device=dev),
        promotions=torch.full((cap, moves), -1, dtype=torch.long, device=dev),
        q_targets=torch.zeros(cap, moves, device=dev),
        q_mask=torch.zeros(cap, moves, dtype=torch.bool, device=dev),
        wdl_targets=torch.zeros(cap, dtype=torch.long, device=dev),
        value_targets=torch.zeros(cap, device=dev),
        moves_left_targets=torch.zeros(cap, device=dev),
        priorities=torch.ones(cap, device=dev),
    )


def write_batch(shard: ReplayShard, batch: SyntheticReplayBatch, priorities: torch.Tensor) -> None:
    count = int(batch.features.shape[0])
    if count > shard.capacity:
        raise ValueError("batch is larger than replay capacity.")
    if priorities.shape != (count,):
        raise ValueError("priorities must have shape [batch].")

    positions = (
        torch.arange(count, device=batch.features.device) + shard.write_index
    ) % shard.capacity
    shard.features[positions] = batch.features
    shard.from_squares[positions] = batch.from_squares
    shard.to_squares[positions] = batch.to_squares
    shard.promotions[positions] = batch.promotions
    shard.q_targets[positions] = batch.q_targets
    shard.q_mask[positions] = batch.q_mask
    shard.wdl_targets[positions] = batch.wdl_targets
    shard.value_targets[positions] = batch.value_targets
    shard.moves_left_targets[positions] = batch.moves_left_targets
    shard.priorities[positions] = priorities.clamp_min(1e-6)

    shard.write_index = (shard.write_index + count) % shard.capacity
    shard.filled = min(shard.capacity, shard.filled + count)


def sample_batch(
    shard: ReplayShard,
    batch_size: int,
    beta: float = 0.4,
    generator: torch.Generator | None = None,
    mode: str = "priority",
) -> SyntheticReplayBatch:
    if shard.filled <= 0:
        raise ValueError("cannot sample from an empty replay shard.")

    normalized = mode.lower().replace("_", "-")
    if normalized == "uniform":
        indices = torch.randint(
            0,
            shard.filled,
            (batch_size,),
            device=shard.features.device,
            generator=generator,
        )
        weights = torch.ones(batch_size, device=shard.features.device)
    elif normalized == "priority":
        priorities = shard.priorities[: shard.filled].float()
        probs = priorities / priorities.sum().clamp_min(1e-6)
        indices = torch.multinomial(probs, batch_size, replacement=True, generator=generator)
        weights = (shard.filled * probs[indices]).pow(-beta)
        weights = weights / weights.max().clamp_min(1e-6)
    else:
        raise ValueError("mode must be priority or uniform.")

    return SyntheticReplayBatch(
        features=shard.features[indices],
        from_squares=shard.from_squares[indices],
        to_squares=shard.to_squares[indices],
        promotions=shard.promotions[indices],
        q_targets=shard.q_targets[indices],
        q_mask=shard.q_mask[indices],
        wdl_targets=shard.wdl_targets[indices],
        value_targets=shard.value_targets[indices],
        moves_left_targets=shard.moves_left_targets[indices],
        weights=weights,
    )
