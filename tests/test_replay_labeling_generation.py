import pytest

torch = pytest.importorskip("torch")

from lattice.config import (  # noqa: E402
    LatticeConfig,
    ModelConfig,
    ReplayConfig,
    StaticBufferConfig,
)
from lattice.generation import backfill_outcomes, sample_actions  # noqa: E402
from lattice.labeling import (  # noqa: E402
    apply_terminal_overrides,
    build_negamax_targets,
    choose_sparse_expansions,
)
from lattice.replay import allocate_replay_shard, sample_batch, write_batch  # noqa: E402
from lattice.synthetic import make_synthetic_replay_batch  # noqa: E402


def small_config() -> LatticeConfig:
    return LatticeConfig(
        model=ModelConfig(d_model=32, blocks=1, q_rank=8),
        replay=ReplayConfig(records=16, learner_batch=4, global_batch=8),
        buffers=StaticBufferConfig(
            active_games=8,
            child_positions=5,
            replay_records=16,
            learner_batch=4,
        ),
    )


def test_replay_write_and_sample() -> None:
    cfg = small_config()
    shard = allocate_replay_shard(cfg, "cpu", capacity=16)
    batch = make_synthetic_replay_batch(cfg, "cpu", batch_size=4)

    write_batch(shard, batch, torch.ones(4))
    sampled = sample_batch(shard, batch_size=4)

    assert shard.filled == 4
    assert shard.freshness(cfg.replay.expected_reuse).fill_ratio == 0.25
    assert sampled.features.shape == batch.features.shape


def test_choose_sparse_expansions_contract() -> None:
    q_scores = torch.tensor([[1.0, 5.0, 3.0, -1.0]])
    legal_mask = torch.tensor([[True, True, True, False]])

    indices = choose_sparse_expansions(q_scores, legal_mask, top_k=2, random_children=1)

    assert indices.shape == (1, 3)
    assert indices[0, 0].item() == 1


def test_negamax_target_contract() -> None:
    child_values = torch.tensor([[0.2, -0.5]])
    terminal_values = torch.tensor([[1.0, 0.0]])
    terminal_mask = torch.tensor([[False, True]])

    value_target, wdl_target = build_negamax_targets(
        child_values,
        terminal_values,
        terminal_mask,
        outcome_wdl=torch.tensor([0.5]),
    )

    assert value_target.shape == (1,)
    assert wdl_target.shape == (1,)


def test_terminal_overrides() -> None:
    values = apply_terminal_overrides(
        torch.tensor([[0.1, 0.2]]),
        torch.tensor([[1.0, -1.0]]),
        torch.tensor([[True, False]]),
    )

    assert values.tolist() == [[1.0, 0.2]]


def test_sample_actions_respects_mask() -> None:
    action = sample_actions(
        torch.tensor([[0.0, 1.0, 2.0]]),
        torch.tensor([[False, False, True]]),
        temperature=0.0,
        epsilon=0.0,
    )

    assert action.tolist() == [2]


def test_backfill_outcomes() -> None:
    targets = backfill_outcomes(
        wdl_targets=torch.zeros(4),
        record_game_ids=torch.tensor([0, 0, 1, 2]),
        finished_game_ids=torch.tensor([0]),
        outcomes=torch.tensor([1.0]),
    )

    assert targets.tolist() == [0.4, 0.4, 0.0, 0.0]
