## Why

The current LATTICE spec targets a single GPU and spreads delivery across long milestone phases. This change turns it into a dual-GPU, DDP-only build that reaches a runnable GPU-heavy training loop as fast as possible.

## What Changes

- Build the project around PyTorch DDP with one process per GPU and NCCL gradient all-reduce.
- Run generation, labeling, replay sampling, and learner work on every GPU rank instead of dedicating one GPU to generation and one to training.
- Keep replay rank-local in VRAM for the first implementation to avoid central queue overhead.
- Use custom CUDA or Triton kernels for chess state transitions once the Python reference path locks correctness.
- Replace long milestone gates with a short implementation path: runtime skeleton, GPU chess correctness, local replay, DDP learner, integrated smoke run, profiler pass.
- Keep the original LATTICE goals as performance targets, but let early tasks ship a measurable vertical slice before the full strength system exists.

## Capabilities

### New Capabilities

- `ddp-runtime`: Dual-GPU process launch, rank setup, NCCL validation, config, logging, and static buffer allocation.
- `gpu-chess-kernel`: Batched GPU chess state, legal move generation, move application, terminal detection, and CPU differential checks.
- `gpu-self-play-generation`: Per-rank GPU self-play, Q-guided action selection, sparse child expansion, label creation, opening seed generation, and outcome backfill.
- `vram-replay-training`: Rank-local VRAM replay shards, prioritized sampling, DDP learner steps, EMA labelers, optimizer state, and divergence tripwires.
- `lattice-model-evaluation`: LATTICE model heads, checkpointing, UCI inference path, fixed evaluation suites, and profiler-backed acceptance gates.

### Modified Capabilities

None. No baseline OpenSpec capabilities exist yet.

## Impact

- Adds a PyTorch DDP training architecture for a dual-GPU VM.
- Changes the project target from single-process/single-GPU to multi-process/single-node DDP.
- Requires CUDA-capable development and profiling tools for the GPU chess kernel and steady-state loop.
- Keeps CPU work limited to launch, logging, checkpoints, CPU reference tests, and optional UCI parsing.
