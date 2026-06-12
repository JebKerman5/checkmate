from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from lattice.chess.movegen import prototype_pawn_knight_king_moves


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
    if iterations <= 0:
        raise ValueError("iterations must be positive.")

    started = perf_counter()
    for _ in range(iterations):
        prototype_pawn_knight_king_moves(fens, device=device)
    elapsed = max(perf_counter() - started, 1e-9)
    steps = len(fens) * iterations
    return MovegenBenchmark(
        positions=len(fens),
        iterations=iterations,
        elapsed_seconds=elapsed,
        board_steps_per_second=steps / elapsed,
        host_syncs=0,
    )
