from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from lattice.losses import masked_huber_loss
from lattice.model import LatticeModel, score_moves
from lattice.synthetic import SyntheticReplayBatch


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    q: torch.Tensor
    value: torch.Tensor
    wdl: torch.Tensor
    moves_left: torch.Tensor


def compute_lattice_loss(
    model: LatticeModel | nn.parallel.DistributedDataParallel,
    batch: SyntheticReplayBatch,
) -> LossBreakdown:
    output = model(batch.features)
    q_pred = score_moves(output, batch.from_squares, batch.to_squares, batch.promotions)
    q_loss = masked_huber_loss(q_pred, batch.q_targets, batch.q_mask)
    value_loss = F.huber_loss(output.value, batch.value_targets)
    wdl_loss = F.cross_entropy(output.wdl_logits, batch.wdl_targets)
    moves_left_loss = F.huber_loss(output.moves_left, batch.moves_left_targets)
    total = q_loss + value_loss + wdl_loss + 0.3 * moves_left_loss

    return LossBreakdown(
        total=total,
        q=q_loss.detach(),
        value=value_loss.detach(),
        wdl=wdl_loss.detach(),
        moves_left=moves_left_loss.detach(),
    )


def train_step(
    model: LatticeModel | nn.parallel.DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    batch: SyntheticReplayBatch,
    grad_clip: float = 1.0,
    use_bf16: bool = True,
) -> LossBreakdown:
    optimizer.zero_grad(set_to_none=True)

    is_cuda = batch.features.is_cuda
    autocast_context: Any
    if is_cuda:
        autocast_context = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16)
    else:
        autocast_context = nullcontext()

    with autocast_context:
        losses = compute_lattice_loss(model, batch)

    losses.total.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return losses


def wrap_ddp(model: LatticeModel, device_index: int) -> nn.parallel.DistributedDataParallel:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before wrapping DDP.")

    return nn.parallel.DistributedDataParallel(model, device_ids=[device_index])
