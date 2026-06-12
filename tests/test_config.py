from lattice.config import LatticeConfig
from lattice.runtime import estimate_buffer_plan, format_buffer_plan


def test_spec_default_shapes() -> None:
    cfg = LatticeConfig()

    assert cfg.board.games == 65_536
    assert cfg.board.piece_planes == 12
    assert cfg.board.max_moves == 96
    assert cfg.board.zobrist_history == 104


def test_spec_default_training_budget() -> None:
    cfg = LatticeConfig()

    assert cfg.runtime.required_world_size == 2
    assert cfg.runtime.backend == "nccl"
    assert cfg.replay.records == 4_000_000
    assert cfg.replay.learner_batch == 2_048
    assert cfg.replay.global_batch == 4_096
    assert cfg.labeling.top_k == 4
    assert cfg.labeling.random_children == 1


def test_runtime_buffer_plan_is_positive() -> None:
    plan = estimate_buffer_plan(LatticeConfig())

    assert plan.active_games_bytes > 0
    assert plan.child_positions_bytes > 0
    assert plan.replay_bytes > 0
    assert plan.learner_batch_bytes > 0
    assert plan.total_bytes > plan.replay_bytes


def test_runtime_buffer_plan_format() -> None:
    rows = format_buffer_plan(estimate_buffer_plan(LatticeConfig()))

    assert rows[-1][0] == "total planned"
    assert all("MiB" in value for _, value in rows)
