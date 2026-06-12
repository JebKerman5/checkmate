from __future__ import annotations

from dataclasses import dataclass

import torch

from lattice.chess.reference import board_from_fen, legal_uci_moves
from lattice.chess.state import BoardTensors, PiecePlane
from lattice.chess.tables import KING_ATTACKS, KNIGHT_ATTACKS
from lattice.config import BoardBatchConfig


@dataclass(frozen=True)
class MovegenResult:
    from_squares: torch.Tensor
    to_squares: torch.Tensor
    move_codes: torch.Tensor
    legal_mask: torch.Tensor
    overflow: torch.Tensor


def encode_move(from_square: int, to_square: int, promotion: int = 0) -> int:
    return from_square | (to_square << 6) | (promotion << 12)


def empty_movegen_result(batch: int, max_moves: int, device: torch.device | str) -> MovegenResult:
    dev = torch.device(device)
    return MovegenResult(
        from_squares=torch.zeros(batch, max_moves, dtype=torch.long, device=dev),
        to_squares=torch.zeros(batch, max_moves, dtype=torch.long, device=dev),
        move_codes=torch.zeros(batch, max_moves, dtype=torch.int16, device=dev),
        legal_mask=torch.zeros(batch, max_moves, dtype=torch.bool, device=dev),
        overflow=torch.zeros(batch, dtype=torch.bool, device=dev),
    )


def legal_moves_from_reference(
    fens: list[str],
    device: torch.device | str,
    cfg: BoardBatchConfig | None = None,
) -> MovegenResult:
    board_cfg = cfg or BoardBatchConfig()
    result = empty_movegen_result(len(fens), board_cfg.max_moves, device)

    for batch_index, fen in enumerate(fens):
        board = board_from_fen(fen)
        moves = list(board.legal_moves)
        if len(moves) > board_cfg.max_moves:
            result.overflow[batch_index] = True
        for move_index, move in enumerate(moves[: board_cfg.max_moves]):
            result.from_squares[batch_index, move_index] = move.from_square
            result.to_squares[batch_index, move_index] = move.to_square
            promo = int(move.promotion or 0)
            result.move_codes[batch_index, move_index] = encode_move(
                move.from_square,
                move.to_square,
                promo,
            )
            result.legal_mask[batch_index, move_index] = True

    return result


def prototype_pawn_knight_king_moves(
    fens: list[str],
    device: torch.device | str,
    cfg: BoardBatchConfig | None = None,
) -> MovegenResult:
    """Small-batch prototype path.

    This intentionally uses the CPU reference for legal filtering while preserving the tensor
    output contract used by the future GPU kernel. It lets the rest of the training stack
    integrate before the CUDA/Triton movegen path is finished.
    """

    return legal_moves_from_reference(fens, device, cfg)


def prototype_pawn_knight_king_from_tensors(
    boards: BoardTensors,
    cfg: BoardBatchConfig | None = None,
) -> MovegenResult:
    """Small-batch tensor prototype for non-slider pieces.

    This path is intentionally simple and useful for shape integration. The production path
    still needs CUDA/Triton kernels to avoid host-side square iteration.
    """

    board_cfg = cfg or BoardBatchConfig()
    result = empty_movegen_result(boards.batch_size, board_cfg.max_moves, boards.bb.device)
    for batch_index in range(boards.batch_size):
        own_occ = int(boards.occ[batch_index, 0].item())
        opp_occ = int(boards.occ[batch_index, 1].item())
        all_occ = own_occ | opp_occ
        write = 0
        pieces = (
            (int(PiecePlane.WHITE_PAWN), _white_pawn_targets),
            (int(PiecePlane.WHITE_KNIGHT), lambda sq, occ, own, opp: KNIGHT_ATTACKS[sq] & ~own),
            (int(PiecePlane.WHITE_KING), lambda sq, occ, own, opp: KING_ATTACKS[sq] & ~own),
        )
        for plane, target_fn in pieces:
            bitboard = int(boards.bb[batch_index, plane].item())
            for from_square in _iter_squares(bitboard):
                targets = target_fn(from_square, all_occ, own_occ, opp_occ)
                for to_square in _iter_squares(targets):
                    if write >= board_cfg.max_moves:
                        result.overflow[batch_index] = True
                        break
                    result.from_squares[batch_index, write] = from_square
                    result.to_squares[batch_index, write] = to_square
                    result.move_codes[batch_index, write] = encode_move(from_square, to_square)
                    result.legal_mask[batch_index, write] = True
                    write += 1

    return result


def compare_with_reference(fen: str, generated: MovegenResult, row: int = 0) -> tuple[bool, str]:
    expected = set(legal_uci_moves(fen))
    actual: set[str] = set()
    for idx in torch.nonzero(generated.legal_mask[row], as_tuple=False).flatten().tolist():
        from_sq = int(generated.from_squares[row, idx].item())
        to_sq = int(generated.to_squares[row, idx].item())
        actual.add(f"{_square_name(from_sq)}{_square_name(to_sq)}")

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        return False, f"missing={missing[:8]} extra={extra[:8]}"
    return True, "ok"


def _square_name(square: int) -> str:
    files = "abcdefgh"
    return f"{files[square % 8]}{1 + square // 8}"


def _iter_squares(bitboard: int) -> list[int]:
    squares: list[int] = []
    while bitboard:
        lsb = bitboard & -bitboard
        squares.append(lsb.bit_length() - 1)
        bitboard ^= lsb
    return squares


def _white_pawn_targets(square: int, occupancy: int, own_occ: int, opp_occ: int) -> int:
    rank = square // 8
    file = square % 8
    targets = 0
    one = square + 8
    if one < 64 and not (occupancy & (1 << one)):
        targets |= 1 << one
        two = square + 16
        if rank == 1 and not (occupancy & (1 << two)):
            targets |= 1 << two
    if file > 0:
        cap = square + 7
        if cap < 64 and (opp_occ & (1 << cap)):
            targets |= 1 << cap
    if file < 7:
        cap = square + 9
        if cap < 64 and (opp_occ & (1 << cap)):
            targets |= 1 << cap
    return targets & ~own_occ
