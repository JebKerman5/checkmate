from dataclasses import dataclass, replace


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
    sampling_mode: str = "priority"


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
class BenchmarkConfig:
    preset: str = "checkmate-alpha1"
    seconds: float = 1.0
    replay_capacity: int = 1024
    sample_batch: int = 32
    warmup_batches: int = 1
    use_compile: bool = False
    use_cuda_graphs: bool = False
    movegen_path: str = "reference"


@dataclass(frozen=True)
class LatticeConfig:
    runtime: RuntimeConfig = RuntimeConfig()
    board: BoardBatchConfig = BoardBatchConfig()
    model: ModelConfig = ModelConfig()
    replay: ReplayConfig = ReplayConfig()
    buffers: StaticBufferConfig = StaticBufferConfig()
    labeling: LabelingConfig = LabelingConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()


def config_preset(name: str) -> LatticeConfig:
    """Return an explicit config preset without changing the alpha defaults."""

    normalized = name.lower().replace("_", "-")
    if normalized == "local-contract":
        return LatticeConfig(
            runtime=RuntimeConfig(
                required_world_size=1,
                backend="gloo",
                require_cuda=False,
                require_bf16=False,
                require_peer_access=False,
            ),
            board=BoardBatchConfig(games=8, max_moves=96),
            model=ModelConfig(d_model=32, blocks=1, q_rank=8),
            replay=ReplayConfig(records=128, learner_batch=4, global_batch=4, expected_reuse=2),
            buffers=StaticBufferConfig(
                active_games=8,
                child_positions=2,
                replay_records=128,
                learner_batch=4,
            ),
            benchmark=BenchmarkConfig(
                preset=normalized,
                seconds=0.05,
                replay_capacity=64,
                sample_batch=4,
                warmup_batches=1,
            ),
        )
    if normalized == "single-gpu-smoke":
        return LatticeConfig(
            runtime=RuntimeConfig(
                required_world_size=1,
                backend="nccl",
                require_cuda=True,
                require_bf16=False,
                require_peer_access=False,
            ),
            board=BoardBatchConfig(games=1024, max_moves=96),
            model=ModelConfig(d_model=96, blocks=2, q_rank=16),
            replay=ReplayConfig(records=16_384, learner_batch=128, global_batch=128),
            buffers=StaticBufferConfig(
                active_games=1024,
                child_positions=5,
                replay_records=16_384,
                learner_batch=128,
            ),
            benchmark=BenchmarkConfig(
                preset=normalized,
                seconds=1.0,
                replay_capacity=4096,
                sample_batch=128,
                warmup_batches=2,
                movegen_path="vectorized-tensor",
            ),
        )
    if normalized == "dual-gpu-ddp-smoke":
        cfg = config_preset("single-gpu-smoke")
        return replace(
            cfg,
            runtime=RuntimeConfig(
                required_world_size=2,
                backend="nccl",
                require_cuda=True,
                require_bf16=False,
                require_peer_access=True,
            ),
            replay=replace(cfg.replay, global_batch=cfg.replay.learner_batch * 2),
            benchmark=replace(cfg.benchmark, preset=normalized, movegen_path="vectorized-tensor"),
        )
    if normalized == "checkmate-alpha1":
        return LatticeConfig(
            replay=ReplayConfig(sampling_mode="uniform"),
            benchmark=BenchmarkConfig(preset=normalized, movegen_path="vectorized-tensor"),
        )

    raise ValueError(
        "Unknown config preset. Expected local-contract, single-gpu-smoke, "
        "dual-gpu-ddp-smoke, or checkmate-alpha1."
    )
