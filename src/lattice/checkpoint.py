from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class CheckpointMeta:
    global_step: int
    rank: int
    world_size: int


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    meta: CheckpointMeta,
    ema_fast: torch.nn.Module | None = None,
    ema_slow: torch.nn.Module | None = None,
    scheduler: Any | None = None,
) -> None:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "meta": meta,
    }
    if ema_fast is not None:
        payload["ema_fast"] = ema_fast.state_dict()
    if ema_slow is not None:
        payload["ema_slow"] = ema_slow.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)
