import pytest

torch = pytest.importorskip("torch")

from lattice.config import (  # noqa: E402
    LatticeConfig,
    ModelConfig,
    ReplayConfig,
    StaticBufferConfig,
)
from lattice.model import LatticeModel  # noqa: E402
from lattice.synthetic import make_synthetic_replay_batch  # noqa: E402
from lattice.training import train_step  # noqa: E402


def small_config() -> LatticeConfig:
    return LatticeConfig(
        model=ModelConfig(d_model=32, blocks=1, q_rank=8),
        replay=ReplayConfig(records=128, learner_batch=4, global_batch=8),
        buffers=StaticBufferConfig(
            active_games=8,
            child_positions=5,
            replay_records=128,
            learner_batch=4,
        ),
    )


def test_synthetic_replay_batch_contract() -> None:
    cfg = small_config()
    batch = make_synthetic_replay_batch(cfg, device="cpu")

    assert batch.features.shape == (4, cfg.model.squares, cfg.model.input_features)
    assert batch.from_squares.shape == (4, 5)
    assert batch.to_squares.shape == (4, 5)
    assert batch.q_targets.shape == (4, 5)
    assert batch.q_mask.shape == (4, 5)
    assert batch.wdl_targets.shape == (4,)
    assert batch.value_targets.shape == (4,)


def test_train_step_updates_model() -> None:
    cfg = small_config()
    model = LatticeModel(cfg.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = make_synthetic_replay_batch(cfg, device="cpu")
    before = next(model.parameters()).detach().clone()

    losses = train_step(model, optimizer, batch, use_bf16=False)
    after = next(model.parameters()).detach()

    assert losses.total.isfinite()
    assert not torch.equal(before, after)
