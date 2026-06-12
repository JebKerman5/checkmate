from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch

from lattice.config import BoardBatchConfig


class PiecePlane(IntEnum):
    WHITE_PAWN = 0
    WHITE_KNIGHT = 1
    WHITE_BISHOP = 2
    WHITE_ROOK = 3
    WHITE_QUEEN = 4
    WHITE_KING = 5
    BLACK_PAWN = 6
    BLACK_KNIGHT = 7
    BLACK_BISHOP = 8
    BLACK_ROOK = 9
    BLACK_QUEEN = 10
    BLACK_KING = 11


class GameStatus(IntEnum):
    ONGOING = 0
    WHITE_WIN = 1
    BLACK_WIN = 2
    DRAW_STALEMATE = 3
    DRAW_FIFTY_MOVE = 4
    DRAW_REPETITION = 5
    DRAW_INSUFFICIENT = 6


@dataclass(frozen=True)
class BoardTensors:
    bb: torch.Tensor
    occ: torch.Tensor
    meta: torch.Tensor
    zhist: torch.Tensor
    status: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.bb.shape[0])


def allocate_board_tensors(
    batch_size: int,
    device: torch.device | str,
    cfg: BoardBatchConfig | None = None,
) -> BoardTensors:
    board_cfg = cfg or BoardBatchConfig()
    dev = torch.device(device)

    return BoardTensors(
        bb=torch.zeros(batch_size, board_cfg.piece_planes, dtype=torch.uint64, device=dev),
        occ=torch.zeros(batch_size, 2, dtype=torch.uint64, device=dev),
        meta=torch.zeros(batch_size, dtype=torch.int32, device=dev),
        zhist=torch.zeros(batch_size, board_cfg.zobrist_history, dtype=torch.uint64, device=dev),
        status=torch.zeros(batch_size, dtype=torch.int8, device=dev),
    )


def update_occupancy(boards: BoardTensors) -> torch.Tensor:
    white = boards.bb[:, 0].clone()
    for idx in range(1, 6):
        white = torch.bitwise_or(white, boards.bb[:, idx])

    black = boards.bb[:, 6].clone()
    for idx in range(7, 12):
        black = torch.bitwise_or(black, boards.bb[:, idx])

    return torch.stack([white, black], dim=1)
