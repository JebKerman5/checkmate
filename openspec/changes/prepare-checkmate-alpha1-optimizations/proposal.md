## Why

The base LATTICE DDP scaffold is close to the point where real dual-GPU testing starts, but the project still lacks alpha-grade performance gates, speed regression tests, and optimized hot paths for the GPU-heavy loop. Before spending money on a dual-GPU training session, the repo needs a measurable optimization pass so `checkmate-alpha1` can prove that the GPU, not Python or the CPU reference path, is the bottleneck.

## What Changes

- Add a benchmark suite that measures learner throughput, model forward/backward speed, replay write/sample throughput, movegen throughput, DDP all-reduce overhead, host synchronization count, CPU usage, and profiler acceptance gaps.
- Add machine-readable benchmark outputs so VM smoke runs can compare current speed against an alpha baseline without relying on screenshots or manual notes.
- Replace or bypass prototype bottlenecks that block alpha testing, especially CPU reference-backed movegen, Python loops in tensor paths, tiny synthetic training batches, and uncompiled model/training execution.
- Add GPU-first optimization options for PyTorch compile, CUDA graphs where practical, static buffer reuse, fused tensor operations, pinned logging boundaries, and Triton/CUDA kernels for the highest-impact chess hot path.
- Add alpha readiness checks, docs, and commands that verify the repo is ready for a paid dual-GPU DDP run without leaving local background services or running GPU-heavy jobs on the gaming PC.

## Capabilities

### New Capabilities

- `alpha-performance-benchmarks`: Defines required speed tests, metrics, JSON report formats, and pass/fail thresholds for local CPU-safe tests and VM-only GPU/DDP tests.
- `alpha-hot-path-optimization`: Defines optimization requirements for model, learner, replay, movegen, profiling, and host synchronization removal before `checkmate-alpha1`.
- `checkmate-alpha1-readiness`: Defines the release gate for alpha1, including runbook commands, artifact manifests, config presets, checkpoint/profile outputs, and no-local-bloat safeguards.

### Modified Capabilities

None. There are no finalized main specs in `openspec/specs/` yet; this change adds alpha optimization capabilities that build on the active `implement-lattice-system` work.

## Impact

- Affected code: `src/lattice/model.py`, `src/lattice/training.py`, `src/lattice/trainer.py`, `src/lattice/replay.py`, `src/lattice/generation.py`, `src/lattice/chess/*`, `src/lattice/profiling.py`, `src/lattice/cli.py`, and runtime/config modules.
- Affected tests: new benchmark contract tests, performance smoke tests with skip markers for missing CUDA/DDP, and regression checks for JSON metric output.
- Affected docs: `README.md`, `docs/VM_RUNBOOK.md`, and alpha release notes/runbooks.
- Affected systems: local development remains CPU-safe by default; GPU-heavy benchmark and training commands are VM-only and must be explicit.
