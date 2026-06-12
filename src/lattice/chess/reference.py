from __future__ import annotations

from dataclasses import dataclass

import chess


@dataclass(frozen=True)
class TerminalStatus:
    kind: str
    winner: str | None = None


def board_from_fen(fen: str) -> chess.Board:
    if fen == "startpos":
        return chess.Board()
    return chess.Board(fen)


def legal_uci_moves(fen: str) -> list[str]:
    board = board_from_fen(fen)
    return sorted(move.uci() for move in board.legal_moves)


def make_uci_move(fen: str, move_uci: str) -> str:
    board = board_from_fen(fen)
    board.push_uci(move_uci)
    return board.fen()


def terminal_status(fen: str) -> TerminalStatus:
    board = board_from_fen(fen)

    if board.is_checkmate():
        winner = "black" if board.turn == chess.WHITE else "white"
        return TerminalStatus("checkmate", winner)
    if board.is_stalemate():
        return TerminalStatus("stalemate")
    if board.is_fifty_moves():
        return TerminalStatus("fifty-move")
    if board.is_repetition(3) or board.can_claim_threefold_repetition():
        return TerminalStatus("threefold")
    if board.is_insufficient_material():
        return TerminalStatus("insufficient-material")
    return TerminalStatus("ongoing")


def perft(fen: str, depth: int) -> int:
    if depth < 0:
        raise ValueError("depth must be non-negative.")

    board = board_from_fen(fen)

    def walk(node: chess.Board, remaining: int) -> int:
        if remaining == 0:
            return 1
        total = 0
        for move in node.legal_moves:
            node.push(move)
            total += walk(node, remaining - 1)
            node.pop()
        return total

    return walk(board, depth)
