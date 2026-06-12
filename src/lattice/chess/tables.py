from __future__ import annotations

from dataclasses import dataclass


def square_bit(square: int) -> int:
    return 1 << square


def rank_of(square: int) -> int:
    return square // 8


def file_of(square: int) -> int:
    return square % 8


def inside(rank: int, file: int) -> bool:
    return 0 <= rank < 8 and 0 <= file < 8


def ray(square: int, dr: int, df: int) -> int:
    mask = 0
    rank = rank_of(square) + dr
    file = file_of(square) + df
    while inside(rank, file):
        mask |= square_bit(rank * 8 + file)
        rank += dr
        file += df
    return mask


def jump_mask(square: int, jumps: tuple[tuple[int, int], ...]) -> int:
    mask = 0
    rank = rank_of(square)
    file = file_of(square)
    for dr, df in jumps:
        nr = rank + dr
        nf = file + df
        if inside(nr, nf):
            mask |= square_bit(nr * 8 + nf)
    return mask


KNIGHT_JUMPS = (
    (2, 1),
    (2, -1),
    (-2, 1),
    (-2, -1),
    (1, 2),
    (1, -2),
    (-1, 2),
    (-1, -2),
)
KING_JUMPS = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)


KNIGHT_ATTACKS = tuple(jump_mask(square, KNIGHT_JUMPS) for square in range(64))
KING_ATTACKS = tuple(jump_mask(square, KING_JUMPS) for square in range(64))
BISHOP_RAYS = tuple(
    ray(square, 1, 1) | ray(square, 1, -1) | ray(square, -1, 1) | ray(square, -1, -1)
    for square in range(64)
)
ROOK_RAYS = tuple(
    ray(square, 1, 0) | ray(square, -1, 0) | ray(square, 0, 1) | ray(square, 0, -1)
    for square in range(64)
)
QUEEN_RAYS = tuple(BISHOP_RAYS[square] | ROOK_RAYS[square] for square in range(64))


@dataclass(frozen=True)
class AttackTables:
    knight: tuple[int, ...] = KNIGHT_ATTACKS
    king: tuple[int, ...] = KING_ATTACKS
    bishop_rays: tuple[int, ...] = BISHOP_RAYS
    rook_rays: tuple[int, ...] = ROOK_RAYS
    queen_rays: tuple[int, ...] = QUEEN_RAYS
