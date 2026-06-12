import pytest

from lattice.chess.reference import legal_uci_moves, make_uci_move, perft, terminal_status


def test_reference_startpos_legal_moves() -> None:
    moves = legal_uci_moves("startpos")

    assert len(moves) == 20
    assert "e2e4" in moves
    assert "g1f3" in moves


def test_reference_make_move() -> None:
    fen = make_uci_move("startpos", "e2e4")

    assert " b " in fen
    assert fen.startswith("rnbqkbnr/pppppppp/8/8/4P3")


def test_reference_terminal_checkmate() -> None:
    status = terminal_status("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")

    assert status.kind == "checkmate"
    assert status.winner == "black"


def test_reference_perft_startpos_depth_2() -> None:
    assert perft("startpos", 2) == 400


def test_reference_rejects_negative_perft_depth() -> None:
    with pytest.raises(ValueError, match="depth"):
        perft("startpos", -1)
