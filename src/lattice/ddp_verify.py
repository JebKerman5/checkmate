from __future__ import annotations

from dataclasses import dataclass

import torch

from lattice.config import LatticeConfig
from lattice.model import LatticeModel
from lattice.runtime import RankInfo
from lattice.synthetic import make_synthetic_replay_batch
from lattice.training import train_step, wrap_ddp


@dataclass(frozen=True)
class WeightSyncResult:
    rank: int
    checksum: float
    max_checksum_delta: float
    synced: bool


def parameter_checksum(model: torch.nn.Module) -> torch.Tensor:
    total = torch.zeros((), device=next(model.parameters()).device)
    for param in model.parameters():
        total = total + param.detach().float().sum()
    return total


def verify_ddp_weight_sync(cfg: LatticeConfig, rank_info: RankInfo) -> WeightSyncResult:
    device = torch.device(f"cuda:{rank_info.device_index}")
    torch.manual_seed(1234)

    model = LatticeModel(cfg.model).to(device)
    ddp_model = wrap_ddp(model, rank_info.device_index)
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)
    batch = make_synthetic_replay_batch(
        cfg,
        device=device,
        batch_size=min(32, cfg.replay.learner_batch),
    )

    train_step(ddp_model, optimizer, batch)
    checksum = parameter_checksum(ddp_model.module)

    gathered = [torch.zeros_like(checksum) for _ in range(rank_info.world_size)]
    torch.distributed.all_gather(gathered, checksum)
    stacked = torch.stack(gathered)
    max_delta = (stacked - stacked[0]).abs().max()

    return WeightSyncResult(
        rank=rank_info.rank,
        checksum=float(checksum.detach().cpu()),
        max_checksum_delta=float(max_delta.detach().cpu()),
        synced=bool(max_delta.item() < 1e-4),
    )
