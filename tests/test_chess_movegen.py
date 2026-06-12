import pytest

torch = pytest.importorskip("torch")

from lattice.chess.benchmark import benchmark_reference_movegen  # noqa: E402
from lattice.chess.encoding import encode_board_batch  # noqa: E402
from lattice.chess.movegen import (  # noqa: E402
    compare_with_reference,
    legal_moves_from_reference,
    prototype_pawn_knight_king_from_tensors,
    prototype_pawn_knight_king_moves,
)


def test_encode_startpos_batch() -> None:
    boards = encode_board_batch(["startpos"], "cpu")

    assert boards.bb.shape == (1, 12)
    assert boards.occ.sum().item() != 0


def test_reference_move_tensor_matches_startpos() -> None:
    result = legal_moves_from_reference(["startpos"], "cpu")

    assert result.legal_mask.sum().item() == 20
    assert not result.overflow.any().item()


def test_prototype_movegen_compares_with_reference() -> None:
    fen = "startpos"
    result = prototype_pawn_knight_king_moves([fen], "cpu")

    ok, detail = compare_with_reference(fen, result)

    assert ok, detail


def test_tensor_piece_prototype_has_startpos_non_slider_moves() -> None:
    boards = encode_board_batch(["startpos"], "cpu")
    result = prototype_pawn_knight_king_from_tensors(boards)

    assert result.legal_mask.sum().item() == 20


def test_movegen_benchmark_contract() -> None:
    result = benchmark_reference_movegen(["startpos"], iterations=1)

    assert result.positions == 1
    assert result.board_steps_per_second > 0
