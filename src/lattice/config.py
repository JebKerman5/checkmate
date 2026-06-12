from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    required_world_size: int = 2
    backend: str = "nccl"
    require_cuda: bool = True
    require_bf16: bool = True
    require_peer_access: bool = True
    master_addr: str = "127.0.0.1"
    master_port: int = 29500


@dataclass(frozen=True)
class BoardBatchConfig:
    games: int = 65_536
    piece_planes: int = 12
    max_moves: int = 96
    zobrist_history: int = 104


@dataclass(frozen=True)
class ModelConfig:
    name: str = "LATTICE-base"
    d_model: int = 192
    blocks: int = 8
    q_rank: int = 32
    mlp_multiplier: int = 3
    input_features: int = 35
    squares: int = 64


@dataclass(frozen=True)
class ReplayConfig:
    records: int = 4_000_000
    learner_batch: int = 2_048
    global_batch: int = 4_096
    expected_reuse: int = 8


@dataclass(frozen=True)
class StaticBufferConfig:
    active_games: int = 65_536
    child_positions: int = 5
    replay_records: int = 4_000_000
    learner_batch: int = 2_048


@dataclass(frozen=True)
class LabelingConfig:
    top_k: int = 4
    random_children: int = 1
    ema_decay_fast: float = 0.999
    ema_decay_slow: float = 0.9995
    refresh_steps: int = 500


@dataclass(frozen=True)
class LatticeConfig:
    runtime: RuntimeConfig = RuntimeConfig()
    board: BoardBatchConfig = BoardBatchConfig()
    model: ModelConfig = ModelConfig()
    replay: ReplayConfig = ReplayConfig()
    buffers: StaticBufferConfig = StaticBufferConfig()
    labeling: LabelingConfig = LabelingConfig()
