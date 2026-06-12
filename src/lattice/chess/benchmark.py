from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from lattice.chess.encoding import encode_board_batch
from lattice.chess.movegen import (
    legal_moves_from_reference,
    prototype_pawn_knight_king_from_tensors,
    prototype_pawn_knight_king_moves,
)


@dataclass(frozen=True)
class MovegenBenchmark:
    positions: int
    iterations: int
    elapsed_seconds: float
    board_steps_per_second: float
    host_syncs: int


def benchmark_reference_movegen(
    fens: list[str],
    iterations: int,
    device: str = "cpu",
) -> MovegenBenchmark:
    return benchmark_movegen_path(fens, iterations=iterations, path="reference", device=device)


def benchmark_movegen_path(
    fens: list[str],
    iterations: int,
    path: str,
    device: str = "cpu",
) -> MovegenBenchmark:
    if iterations <= 0:
        raise ValueError("iterations must be positive.")

    normalized = path.lower().replace("_", "-")
    if normalized not in {"reference", "prototype", "vectorized-tensor"}:
        raise ValueError("path must be reference, prototype, or vectorized-tensor.")

    boards = None
    if normalized == "vectorized-tensor":
        boards = encode_board_batch(fens, device)

    started = perf_counter()
    for _ in range(iterations):
        if normalized == "reference":
            legal_moves_from_reference(fens, device=device)
        elif normalized == "prototype":
            prototype_pawn_knight_king_moves(fens, device=device)
        elif boards is not None:
            prototype_pawn_knight_king_from_tensors(boards)
    elapsed = max(perf_counter() - started, 1e-9)
    steps = len(fens) * iterations
    host_syncs = (
        iterations if normalized == "vectorized-tensor" and device.startswith("cuda") else 0
    )
    return MovegenBenchmark(
        positions=len(fens),
        iterations=iterations,
        elapsed_seconds=elapsed,
        board_steps_per_second=steps / elapsed,
        host_syncs=host_syncs,
    )
