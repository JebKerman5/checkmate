## Context

The repository already has a dual-GPU DDP scaffold, LATTICE-base model, synthetic learner path, replay tensors, profiling summary objects, and a prototype chess API. The remaining base implementation tasks are the real integrated DDP run, profiling that run, and moving the highest-impact chess hot path out of prototype code.

The current speed risk is not the model shape; it is that several alpha-blocking paths are still intentionally simple:

- movegen benchmark uses a CPU reference-backed path
- tensor prototype movegen still has Python loops and `.item()` extraction
- synthetic smoke uses tiny replay capacity and tiny sampled batches
- benchmark results are human-readable tables instead of regression-testable JSON
- profiler metrics are accepted as manually entered values instead of emitted by benchmark commands
- GPU-heavy commands are not clearly separated from local-safe commands

The user wants this prepared quickly for `checkmate-alpha1`, with DDP-only training on a dual-GPU VM and no persistent local processes or gaming-PC GPU load.

## Goals / Non-Goals

**Goals:**

- Add a fast, repeatable alpha benchmark suite before paid dual-GPU training.
- Produce JSON benchmark artifacts that can be compared across commits and VM shapes.
- Optimize the bottlenecks that matter for alpha1: learner step speed, replay write/sample speed, movegen/make throughput, DDP all-reduce share, host synchronization, and CPU usage.
- Keep local commands safe by default: no local CUDA job unless explicitly requested, no background services, no global installs, no auto-started daemons.
- Make `checkmate-alpha1` a concrete release gate with commands, metrics, artifacts, and pass/fail criteria.

**Non-Goals:**

- No multi-node training.
- No non-DDP training architecture.
- No CPU self-play production path.
- No PGN ingestion pipeline.
- No attempt to finish LATTICE-big or lattice beam before alpha1.
- No indefinite benchmark suite that runs for hours by default; long runs are explicit VM-only commands.

## Decisions

### Use a benchmark ladder instead of one giant training run

Implement benchmarks in increasing cost order:

1. Contract/schema tests that run locally without CUDA.
2. CPU-safe smoke benchmarks that skip GPU sections when torch/CUDA is missing.
3. Single-GPU CUDA microbenchmarks on a low-cost VM.
4. Two-GPU DDP smoke benchmark on the target VM.
5. Ten-minute integrated DDP smoke.
6. One-hour alpha candidate run.

This gives a clear failure location and avoids wasting dual-GPU time on bugs that a smaller benchmark could catch.

Alternative considered: run the dual-GPU training session first and inspect failures afterward. That is faster in wall-clock planning but more expensive and less useful when a Python-loop or dtype issue blocks the run.

### Emit machine-readable benchmark artifacts

Every alpha benchmark command writes a JSON report with:

- schema version
- git commit
- command/profile
- environment summary
- metrics
- thresholds
- pass/fail gaps
- skip reasons

Tables can still be printed for humans, but JSON is the source of truth for tests, regressions, and alpha artifacts.

Alternative considered: rely on terminal tables and manual notes. That is too easy to miscompare across commits or VM instances.

### Keep local development safe by default

Commands that can stress a GPU must require an explicit CUDA/DDP profile or a `torchrun` invocation. Local default commands run short CPU-safe checks, validate schemas, and print what was skipped.

The project must not install persistent services, watchers, launch agents, or background daemons. VM setup is reproducible from GitHub and `uv`, not from long-lived local state.

Alternative considered: local GPU auto-detection that runs CUDA benchmarks when available. That violates the gaming-PC constraint and makes accidental load too likely.

### Use profiling to choose the first Triton/CUDA kernel

The optimization pass should not blindly rewrite all chess code. First add timing and host-sync instrumentation, then move the highest-impact hot path into Triton or CUDA. The expected candidates are:

- batched legal move generation and make
- slider attack lookup
- terminal detection
- replay priority sampling

For alpha1, it is acceptable to leave lower-impact code in PyTorch if the gates pass and the profiler shows GPU-heavy learner utilization.

Alternative considered: port the entire chess kernel to CUDA before measuring. That risks spending time on low-impact paths while DDP/learner bottlenecks remain unknown.

### Make compile and CUDA graph optional but benchmarked

Add flags/config for `torch.compile`, static buffer reuse, and CUDA graph capture where shapes are stable. Benchmark eager, compiled, and graph-captured variants where practical, but keep eager as a debuggable fallback.

Alternative considered: require compiled mode for all runs. That can make early debugging harder and may hide graph-capture blockers before the base loop is stable.

### Treat alpha1 readiness as a gate, not a version string

`checkmate-alpha1` is ready when commands can produce the required artifacts and thresholds on a dual-GPU VM:

- tests pass
- local-safe benchmark contracts pass
- VM GPU benchmarks pass
- two-rank DDP benchmark passes
- integrated smoke emits throughput/loss/profile artifacts
- profile summary identifies remaining acceptance gaps
- docs state exactly how to reproduce the run from the public repo

Alternative considered: tag alpha1 after implementation tasks are checked off. That does not prove the system is fast enough to justify real training.

## Risks / Trade-offs

- Benchmark noise -> use warmups, repeated samples, median values, and explicit hardware metadata.
- CUDA graph capture blockers -> keep graph capture optional and report why it was skipped.
- Triton/CUDA kernel correctness bugs -> keep CPU differential tests and small deterministic fixture tests next to every optimized chess path.
- Overfitting to one VM shape -> record GPU model, driver, CUDA, PyTorch, world size, and per-rank batch sizes in every report.
- Too much local setup -> keep GPU dependencies optional and VM-only; do not require local torch for schema tests.
- Premature optimization -> require profiler evidence before replacing large code paths.
- DDP stragglers -> report per-rank throughput, all-reduce share, and rank skew.

## Migration Plan

1. Add benchmark report dataclasses, JSON serialization, and threshold evaluation with unit tests.
2. Add local-safe benchmark CLI that can run without CUDA and emits skip reasons.
3. Add CUDA microbenchmarks for model/learner/replay/movegen and DDP benchmark commands.
4. Add profiling hooks and host-sync counters around the integrated loop.
5. Run the microbenchmarks on a low-cost VM and choose the first hot path to move into Triton/CUDA.
6. Add the optimized path behind a feature flag and keep the reference path for differential tests.
7. Add `checkmate-alpha1` readiness command/docs and artifact manifest.
8. On the dual-GPU VM, run DDP smoke, ten-minute integrated smoke, and one-hour alpha candidate.

Rollback is straightforward: disable optimized kernels/compiled mode through config flags and fall back to the current eager/reference paths while retaining benchmark reporting.

## Open Questions

- Which exact dual-GPU VM shape will be the alpha1 reference target?
- Should the first optimized chess kernel be Triton for iteration speed or CUDA/C++ for bit-level control?
- What minimum alpha1 throughput should we enforce before a longer training run: the original 2.5k labeled positions/s target, a lower prototype threshold, or both as separate warning/fail gates?
- Should benchmark artifacts be stored only under ignored `runs/`, or should selected alpha summaries be committed under `docs/benchmarks/`?
