import pytest

torch = pytest.importorskip("torch")

from lattice.config import ModelConfig  # noqa: E402
from lattice.losses import masked_huber_loss  # noqa: E402
from lattice.model import LatticeModel, score_moves  # noqa: E402


def test_lattice_forward_contract() -> None:
    cfg = ModelConfig(d_model=32, blocks=2, q_rank=8)
    model = LatticeModel(cfg)
    features = torch.randn(3, cfg.squares, cfg.input_features)

    output = model(features)

    assert output.from_features.shape == (3, cfg.squares, cfg.q_rank)
    assert output.to_features.shape == (3, cfg.squares, cfg.q_rank)
    assert output.promo_bias.shape == (5,)
    assert output.wdl_logits.shape == (3, 3)
    assert output.value.shape == (3,)
    assert output.moves_left.shape == (3,)


def test_score_moves_contract() -> None:
    cfg = ModelConfig(d_model=32, blocks=1, q_rank=8)
    model = LatticeModel(cfg)
    output = model(torch.randn(2, cfg.squares, cfg.input_features))

    from_squares = torch.tensor([[0, 1, 2], [8, 9, 10]])
    to_squares = torch.tensor([[16, 17, 18], [24, 25, 26]])
    promotions = torch.tensor([[-1, -1, 4], [-1, 1, -1]])

    scores = score_moves(output, from_squares, to_squares, promotions)

    assert scores.shape == (2, 3)


def test_masked_huber_loss_uses_only_masked_entries() -> None:
    prediction = torch.tensor([[0.0, 10.0, 2.0]])
    target = torch.tensor([[0.0, 0.0, 1.0]])
    mask = torch.tensor([[True, False, True]])

    loss = masked_huber_loss(prediction, target, mask)

    assert torch.isclose(loss, torch.tensor(0.25))
