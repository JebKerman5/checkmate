# LATTICE Implementation Plan

## Product Target

Build a chess engine training system where two GPUs do the expensive work:
batched chess simulation, sparse negamax labeling, replay sampling, and model
training. The CPU handles process launch, UCI parsing, scalar logging, reference
tests, and checkpoints.

## First Engineering Slice

Start with the DDP runtime skeleton. It gives every later task a stable launch
and logging path without running a heavy job:

- one process per GPU through `torchrun`
- rank/device binding
- NCCL and CUDA topology checks
- per-rank static buffer sizing
- a smoke command that validates the runtime and exits

The next slice is the GPU chess kernel. It decides whether the rest of the system
can hit the spec. It should expose:

- packed batched board state with bitboards, occupancy, metadata, Zobrist history,
  and status tensors
- legal move generation for all piece types
- branch-minimized `make` for selected move indices
- terminal detection for mate, stalemate, fifty-move, threefold, and insufficient material
- differential tests against `python-chess`
- benchmark hooks for board-steps per second and host synchronization checks

## Model Slice

After the runtime skeleton exists, add LATTICE-base and train it first on
synthetic GPU replay:

- per-square feature encoder to `[B, 64, d]`
- axis mixer branches for files, ranks, diagonals, and anti-diagonals
- SwiGLU per-square MLP
- Q, WDL, V, and moves-left heads

Keep `KnightMix` behind an ablation flag.

## Training Slice

Build the real loop only after move generation and model shapes hold under tests:

- VRAM replay ring
- generation stream with Q-softmax move choice
- sparse labeler with top-k plus random child expansion
- double EMA labelers
- outcome anchoring, endgame anchors, prioritized replay, and rollback tripwire

The M3 gate controls success: GPU SM-busy at least 90 percent, CPU at most one
core, at least 2.5k labeled positions per second for 24 hours, and no host syncs
inside the steady-state path.
