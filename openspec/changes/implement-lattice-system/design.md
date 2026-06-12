## Context

`LATTICE-SPEC.md` defines a single-GPU system where chess simulation, sparse labeling, replay, and learning live in VRAM. The target now changes to a dual-GPU VM with PyTorch DDP only.

The design keeps the same core bet: the CPU launches jobs, logs scalars, writes checkpoints, and runs reference tests. The GPUs do chess stepping, label creation, replay sampling, and learner work.

## Goals / Non-Goals

**Goals:**

- Reach a runnable two-GPU DDP vertical slice quickly.
- Keep both GPUs doing symmetric generation and learner work.
- Keep replay rank-local in VRAM for the first working implementation.
- Make GPU chess correctness testable against `python-chess`.
- Preserve the original LATTICE performance direction without blocking implementation on full final throughput.

**Non-Goals:**

- No CPU self-play generation path for production training.
- No PGN ingestion pipeline.
- No central replay server.
- No multi-node design in the first implementation.
- No LATTICE-big or lattice beam until LATTICE-base runs under DDP.

## Decisions

### Use symmetric DDP ranks

Each GPU runs the same workload shape:

```text
torchrun --nproc_per_node=2

Rank 0 / GPU 0                         Rank 1 / GPU 1
----------------                         ----------------
active games                             active games
GPU movegen                              GPU movegen
EMA labeler forwards                     EMA labeler forwards
local VRAM replay shard                  local VRAM replay shard
DDP learner batch                        DDP learner batch
        \                                  /
         \                                /
          NCCL gradient all-reduce only
```

This avoids stale-policy data from a generator-only GPU and keeps both ranks eligible to saturate tensor cores during learner steps.

Alternative considered: one GPU generates while one GPU trains. That design reduces DDP value, creates a producer-consumer bottleneck, and requires cross-GPU replay transfer before the core system proves itself.

### Keep replay local per rank

Each rank writes generated records into its own VRAM replay shard and samples from that shard for local learner batches. DDP synchronizes gradients, not replay records.

Alternative considered: shared cross-rank replay. That can improve data mixing later, but it adds communication and scheduling complexity before the first training loop works.

### Start with global batch 4096

Use local batch 2048 per rank for the first DDP learner so the global batch matches the original spec. Make local batch configurable after the profiler shows the learner/generator balance.

Alternative considered: local batch 4096 per rank. That doubles global batch and may improve utilization, but it changes optimizer behavior and replay reuse math on day one.

### Build GPU chess behind a stable API

Expose board-state, movegen, make, and terminal detection through a narrow Python API. Use `python-chess` only for tests and differential checks. Implement the production hot path with Triton or CUDA kernels once the API and tests settle.

Alternative considered: pure PyTorch bitboard tensor ops for the full system. That is useful for early shape validation, but move packing, magic lookup, overflow handling, and terminal checks will likely need lower-level kernels.

### Use local EMA labelers

DDP keeps online model weights synchronized after each optimizer step. Each rank can update EMA labeler snapshots from those synchronized weights without cross-rank host coordination.

Alternative considered: rank 0 broadcasts labelers. That adds synchronization traffic and gives no benefit while model states remain identical.

### Track episodes on GPU

The replay record stores `game_id`, ply, target slots, and outcome state. When a game ends, a GPU backfill path updates records for that game's open episode range.

```text
active_game[game_id]
        |
        v
record write: game_id, ply, board, sparse targets
        |
        v
terminal result
        |
        v
GPU backfill: WDL outcome component for open records
```

This keeps outcome anchoring inside the GPU loop and avoids CPU scans over replay records.

### Gate by profiler output, not calendar phases

Milestone labels in the source spec mean checkpoints. They do not need to become months of planning. The implementation order uses short runnable slices:

1. DDP runtime starts two ranks and allocates buffers.
2. GPU chess API passes CPU differential tests on small batches.
3. One rank generates labeled records into replay.
4. Two ranks run DDP learner on synthetic replay.
5. Generator, replay, labeler, and learner run together.
6. Profiler pass removes host syncs and moves hotspots into Triton or CUDA.

## Risks / Trade-offs

- GPU movegen bugs -> keep CPU differential tests close to every kernel stage and fail on the first mismatch.
- DDP stragglers -> keep per-rank replay warm and measure step wait time around all-reduce.
- Prioritized replay bottleneck -> start with simple GPU priority sampling, then replace it with bucket or segment-tree sampling if profiling shows it dominates.
- Outcome backfill complexity -> store explicit episode metadata and cap maximum open records per game batch.
- Memory pressure on smaller dual-GPU VMs -> make replay size, active game batch, child batch, and learner batch configurable.
- Weak fixed-point learning -> keep outcome anchoring, endgame anchors, double EMA labelers, and rollback tripwires in the first integrated loop.

## Migration Plan

This is a new build. No production migration is required.

Implementation can start from the existing Python scaffold, but the README and config must change from single-GPU wording to dual-GPU DDP wording.

Rollback means returning to the last gated checkpoint during training. Development rollback means disabling integrated generation and running synthetic replay through DDP until the learner path passes again.

## Open Questions

- Which VM shape will be the reference target: 2x12GB, 2x24GB, or larger?
- Do we prefer Triton first for faster iteration, or CUDA/C++ first for bit-level control?
- Should endgame tablebase anchors stay as a CPU-prepared one-time artifact, or should the first version use generated mate-distance probes until tablebase integration lands?
