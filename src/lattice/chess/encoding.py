from __future__ import annotations

import chess
import torch

from lattice.chess.reference import board_from_fen
from lattice.chess.state import (
    BoardTensors,
    PiecePlane,
    allocate_board_tensors,
    update_occupancy,
)

PIECE_TO_PLANE = {
    chess.Piece(chess.PAWN, chess.WHITE): PiecePlane.WHITE_PAWN,
    chess.Piece(chess.KNIGHT, chess.WHITE): PiecePlane.WHITE_KNIGHT,
    chess.Piece(chess.BISHOP, chess.WHITE): PiecePlane.WHITE_BISHOP,
    chess.Piece(chess.ROOK, chess.WHITE): PiecePlane.WHITE_ROOK,
    chess.Piece(chess.QUEEN, chess.WHITE): PiecePlane.WHITE_QUEEN,
    chess.Piece(chess.KING, chess.WHITE): PiecePlane.WHITE_KING,
    chess.Piece(chess.PAWN, chess.BLACK): PiecePlane.BLACK_PAWN,
    chess.Piece(chess.KNIGHT, chess.BLACK): PiecePlane.BLACK_KNIGHT,
    chess.Piece(chess.BISHOP, chess.BLACK): PiecePlane.BLACK_BISHOP,
    chess.Piece(chess.ROOK, chess.BLACK): PiecePlane.BLACK_ROOK,
    chess.Piece(chess.QUEEN, chess.BLACK): PiecePlane.BLACK_QUEEN,
    chess.Piece(chess.KING, chess.BLACK): PiecePlane.BLACK_KING,
}


def encode_board_batch(fens: list[str], device: torch.device | str) -> BoardTensors:
    target = torch.device(device)
    build_device = "cpu" if target.type == "cuda" else target
    boards = allocate_board_tensors(len(fens), build_device)
    for batch_index, fen in enumerate(fens):
        board = board_from_fen(fen)
        for square, piece in board.piece_map().items():
            plane = int(PIECE_TO_PLANE[piece])
            bit = torch.tensor(1 << square, dtype=boards.bb.dtype, device=boards.bb.device)
            boards.bb[batch_index, plane] = torch.bitwise_or(boards.bb[batch_index, plane], bit)

        side_to_move = 0 if board.turn == chess.WHITE else 1
        castling = int(board.has_kingside_castling_rights(chess.WHITE))
        castling |= int(board.has_queenside_castling_rights(chess.WHITE)) << 1
        castling |= int(board.has_kingside_castling_rights(chess.BLACK)) << 2
        castling |= int(board.has_queenside_castling_rights(chess.BLACK)) << 3
        ep_file = 8 if board.ep_square is None else chess.square_file(board.ep_square)
        halfmove = min(board.halfmove_clock, 127)
        boards.meta[batch_index] = side_to_move | (castling << 1) | (ep_file << 5) | (halfmove << 9)

    boards.occ.copy_(update_occupancy(boards))
    if target.type == "cuda":
        return BoardTensors(
            bb=boards.bb.to(target),
            occ=boards.occ.to(target),
            meta=boards.meta.to(target),
            zhist=boards.zhist.to(target),
            status=boards.status.to(target),
        )
    return boards
