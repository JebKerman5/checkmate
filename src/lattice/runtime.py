from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

from lattice.config import LatticeConfig


class RuntimeCheckError(RuntimeError):
    """Raised when the current host cannot run the requested DDP job."""


@dataclass(frozen=True)
class RankInfo:
    rank: int
    local_rank: int
    world_size: int
    device_index: int
    device_name: str
    bf16_supported: bool
    total_memory_bytes: int
    free_memory_bytes: int


@dataclass(frozen=True)
class BufferPlan:
    active_games_bytes: int
    child_positions_bytes: int
    replay_bytes: int
    learner_batch_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.active_games_bytes
            + self.child_positions_bytes
            + self.replay_bytes
            + self.learner_batch_bytes
        )


def torch_available() -> bool:
    return find_spec("torch") is not None


def require_torch() -> Any:
    if not torch_available():
        raise RuntimeCheckError("PyTorch is not installed. Run `uv sync --extra gpu` first.")

    import torch

    return torch


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        msg = f"Environment variable {name} must be an integer, got {raw!r}."
        raise RuntimeCheckError(msg) from exc


def read_rank_env() -> tuple[int, int, int]:
    return (
        env_int("RANK", 0),
        env_int("LOCAL_RANK", 0),
        env_int("WORLD_SIZE", 1),
    )


def estimate_buffer_plan(cfg: LatticeConfig) -> BufferPlan:
    bb_bytes = 8
    meta_bytes = 4
    status_bytes = 1
    move_bytes = 2
    bool_bytes = 1
    fp32_bytes = 4
    bf16_bytes = 2

    board_record_bytes = (
        cfg.board.piece_planes * bb_bytes
        + 2 * bb_bytes
        + meta_bytes
        + cfg.board.zobrist_history * bb_bytes
        + status_bytes
    )
    movegen_bytes = cfg.board.max_moves * (move_bytes + bool_bytes)
    active_games_bytes = cfg.buffers.active_games * (board_record_bytes + movegen_bytes)

    child_state_bytes = (
        cfg.buffers.active_games
        * cfg.buffers.child_positions
        * (cfg.board.piece_planes * bb_bytes + meta_bytes + status_bytes)
    )

    replay_record_bytes = (
        cfg.board.piece_planes * bb_bytes
        + meta_bytes
        + cfg.labeling.top_k * (move_bytes + fp32_bytes)
        + cfg.labeling.random_children * (move_bytes + fp32_bytes)
        + 3 * fp32_bytes
        + fp32_bytes
    )
    replay_bytes = cfg.buffers.replay_records * replay_record_bytes

    learner_feature_bytes = (
        cfg.buffers.learner_batch
        * cfg.model.squares
        * cfg.model.input_features
        * bf16_bytes
    )
    learner_label_bytes = cfg.buffers.learner_batch * (
        cfg.labeling.top_k * fp32_bytes + 6 * fp32_bytes
    )

    return BufferPlan(
        active_games_bytes=active_games_bytes,
        child_positions_bytes=child_state_bytes,
        replay_bytes=replay_bytes,
        learner_batch_bytes=learner_feature_bytes + learner_label_bytes,
    )


def mib(value: int) -> float:
    return value / (1024 * 1024)


def format_buffer_plan(plan: BufferPlan) -> list[tuple[str, str]]:
    return [
        ("active games", f"{mib(plan.active_games_bytes):,.1f} MiB"),
        ("child positions", f"{mib(plan.child_positions_bytes):,.1f} MiB"),
        ("replay shard", f"{mib(plan.replay_bytes):,.1f} MiB"),
        ("learner batch", f"{mib(plan.learner_batch_bytes):,.1f} MiB"),
        ("total planned", f"{mib(plan.total_bytes):,.1f} MiB"),
    ]


def validate_topology(cfg: LatticeConfig) -> list[str]:
    torch = require_torch()
    errors: list[str] = []

    if not torch.cuda.is_available():
        errors.append("CUDA is not available.")
        return errors

    device_count = torch.cuda.device_count()
    if device_count < cfg.runtime.required_world_size:
        errors.append(
            f"Need {cfg.runtime.required_world_size} CUDA devices, found {device_count}."
        )

    if cfg.runtime.require_bf16 and not torch.cuda.is_bf16_supported():
        errors.append("BF16 is not supported by this CUDA runtime/device.")

    if cfg.runtime.require_peer_access and device_count >= 2:
        for src in range(min(device_count, cfg.runtime.required_world_size)):
            for dst in range(min(device_count, cfg.runtime.required_world_size)):
                if src != dst and not torch.cuda.can_device_access_peer(src, dst):
                    errors.append(f"CUDA peer access is unavailable from GPU {src} to GPU {dst}.")

    return errors


def bind_rank_device(cfg: LatticeConfig) -> RankInfo:
    torch = require_torch()
    rank, local_rank, world_size = read_rank_env()

    if world_size != cfg.runtime.required_world_size:
        raise RuntimeCheckError(
            f"Expected WORLD_SIZE={cfg.runtime.required_world_size}, got {world_size}. "
            "Start with `torchrun --nproc_per_node=2 lattice ddp-smoke`."
        )

    topology_errors = validate_topology(cfg)
    if topology_errors:
        raise RuntimeCheckError("; ".join(topology_errors))

    if local_rank >= torch.cuda.device_count():
        raise RuntimeCheckError(
            f"LOCAL_RANK={local_rank} has no matching CUDA device "
            f"(found {torch.cuda.device_count()} devices)."
        )

    torch.cuda.set_device(local_rank)
    free_memory, total_memory = torch.cuda.mem_get_info(local_rank)
    props = torch.cuda.get_device_properties(local_rank)

    return RankInfo(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device_index=local_rank,
        device_name=props.name,
        bf16_supported=torch.cuda.is_bf16_supported(),
        total_memory_bytes=total_memory,
        free_memory_bytes=free_memory,
    )


def initialize_process_group(cfg: LatticeConfig) -> RankInfo:
    torch = require_torch()
    rank_info = bind_rank_device(cfg)

    if not torch.distributed.is_available():
        raise RuntimeCheckError("torch.distributed is not available in this PyTorch build.")

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend=cfg.runtime.backend,
            rank=rank_info.rank,
            world_size=rank_info.world_size,
        )

    return rank_info


def destroy_process_group() -> None:
    if not torch_available():
        return

    import torch

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
