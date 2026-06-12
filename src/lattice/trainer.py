from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from lattice.config import LatticeConfig
from lattice.ema import clone_ema_model, update_ema_model
from lattice.model import LatticeModel
from lattice.replay import allocate_replay_shard, sample_batch, write_batch
from lattice.synthetic import make_synthetic_replay_batch
from lattice.training import train_step
from lattice.tripwire import TripwireState, evaluate_tripwires


@dataclass(frozen=True)
class SmokeRunSummary:
    seconds: float
    learner_steps: int
    learner_steps_per_second: float
    replay_fill: float
    tripwire_failures: list[str]


def run_synthetic_smoke(
    cfg: LatticeConfig,
    device: torch.device | str,
    seconds: float,
    replay_capacity: int = 1024,
) -> SmokeRunSummary:
    dev = torch.device(device)
    model = LatticeModel(cfg.model).to(dev)
    ema_fast = clone_ema_model(model)
    ema_slow = clone_ema_model(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    replay = allocate_replay_shard(cfg, dev, capacity=replay_capacity)

    warm_batch = make_synthetic_replay_batch(cfg, dev, batch_size=min(64, cfg.replay.learner_batch))
    write_batch(replay, warm_batch, torch.ones(warm_batch.features.shape[0], device=dev))

    started = perf_counter()
    steps = 0
    while perf_counter() - started < seconds:
        batch = sample_batch(replay, min(32, cfg.replay.learner_batch))
        train_step(model, optimizer, batch, use_bf16=dev.type == "cuda")
        update_ema_model(ema_fast, model, cfg.labeling.ema_decay_fast)
        update_ema_model(ema_slow, model, cfg.labeling.ema_decay_slow)
        steps += 1

    elapsed = max(perf_counter() - started, 1e-6)
    replay_fill = replay.filled / replay.capacity
    tripwires = evaluate_tripwires(
        TripwireState(
            value_drift=0.0,
            loss_ratio=1.0,
            replay_fill=replay_fill,
            anchor_delta=0.0,
        )
    )
    return SmokeRunSummary(
        seconds=elapsed,
        learner_steps=steps,
        learner_steps_per_second=steps / elapsed,
        replay_fill=replay_fill,
        tripwire_failures=tripwires,
    )
