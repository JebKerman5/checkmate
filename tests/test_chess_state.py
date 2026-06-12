import pytest

torch = pytest.importorskip("torch")

from lattice.chess.state import allocate_board_tensors, update_occupancy  # noqa: E402


def test_allocate_board_tensors_contract() -> None:
    boards = allocate_board_tensors(4, "cpu")

    assert boards.bb.shape == (4, 12)
    assert boards.occ.shape == (4, 2)
    assert boards.meta.shape == (4,)
    assert boards.zhist.shape == (4, 104)
    assert boards.status.shape == (4,)
    assert boards.batch_size == 4


def test_update_occupancy_contract() -> None:
    boards = allocate_board_tensors(1, "cpu")
    boards.bb[0, 0] = 0b0011
    boards.bb[0, 6] = 0b1100

    occ = update_occupancy(boards)

    assert occ.tolist() == [[0b0011, 0b1100]]
