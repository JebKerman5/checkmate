## 1. Benchmark Report Foundation

- [x] 1.1 Add benchmark report dataclasses for schema version, git commit, profile, environment, metrics, thresholds, gaps, and skip reasons.
- [x] 1.2 Add JSON serialization and validation helpers for benchmark reports.
- [x] 1.3 Add threshold evaluation helpers that return deterministic pass/fail gaps.
- [x] 1.4 Add unit tests for report schema validation, skipped hardware sections, and threshold evaluation without importing torch.
- [x] 1.5 Add git/environment capture that records commit, dirty state, Python, platform, torch/CUDA availability, GPU names, world size, and selected config preset.

## 2. Local-Safe Benchmark CLI

- [x] 2.1 Add a local-safe `alpha-bench` CLI profile that runs without CUDA and never starts background processes.
- [x] 2.2 Make the local profile emit valid JSON with GPU/DDP benchmark sections skipped when hardware or torch is unavailable.
- [x] 2.3 Add a bounded-duration smoke metric so the local profile finishes quickly by default.
- [x] 2.4 Add tests that prove the local profile does not require torch GPU extras.
- [x] 2.5 Update README wording so local-safe commands and VM-only GPU commands are visibly separated.

## 3. CUDA And DDP Benchmark Suite

- [x] 3.1 Add CUDA timing utilities that use warmup iterations, repeated samples, medians, and synchronization only around timing boundaries.
- [x] 3.2 Add model forward and learner forward/backward microbenchmarks for eager mode.
- [x] 3.3 Add replay write and replay sample throughput microbenchmarks with priority sampling cost.
- [x] 3.4 Add movegen/make throughput benchmark hooks that distinguish reference, vectorized tensor, Triton, and CUDA paths.
- [x] 3.5 Add optional compiled learner benchmark mode and record compile skip/failure reasons.
- [x] 3.6 Add two-rank DDP benchmark command that gathers per-rank metrics and writes one aggregate report from rank 0.
- [x] 3.7 Add DDP all-reduce share, rank skew, global batch throughput, and synchronization health metrics.
- [x] 3.8 Add pytest markers/skips for CUDA and DDP performance tests so unavailable hardware skips cleanly.

## 4. Profiling And Host-Sync Instrumentation

- [x] 4.1 Add profile event records for generation, labeling, replay write, replay sample, learner forward/backward, optimizer step, EMA update, and all-reduce.
- [x] 4.2 Add host synchronization counters for the integrated loop and benchmark sections.
- [x] 4.3 Extend `profile-summary` to load benchmark JSON and report alpha acceptance gaps directly.
- [x] 4.4 Add tests for host-sync gap reporting and profile summary JSON parsing.
- [x] 4.5 Run the available benchmark suite on a low-cost CUDA VM and record the top bottleneck in an alpha profile artifact.

## 5. Hot-Path Optimization

- [x] 5.1 Select the first chess or replay hot path from profiler evidence and document the before metric.
- [x] 5.2 Implement the optimized path behind a feature/config flag while keeping the reference path for tests.
- [x] 5.3 Remove Python per-record or per-square loops from the selected alpha hot path.
- [x] 5.4 Add differential correctness tests against `python-chess` or existing reference fixtures for the optimized path.
- [x] 5.5 Add benchmark comparisons for reference versus optimized path and record the after metric.
- [x] 5.6 Make alpha VM profiles fail readiness if they fall back to CPU reference movegen for training.

## 6. Learner And Replay Optimization

- [x] 6.1 Add config presets for local contract, single-GPU smoke, dual-GPU DDP smoke, and `checkmate-alpha1`.
- [x] 6.2 Make synthetic smoke use configurable replay capacity, sample batch, warmup count, and benchmark duration.
- [x] 6.3 Add static batch/buffer reuse in learner smoke paths to reduce avoidable allocations.
- [x] 6.4 Benchmark BF16 eager versus compiled learner modes on CUDA and record throughput deltas.
- [x] 6.5 Add replay allocation/throughput diagnostics that flag priority sampling if it dominates learner step time.
- [x] 6.6 Verify DDP learner benchmark keeps rank weights synchronized after benchmark steps.

## 7. Alpha1 Readiness And Runbooks

- [x] 7.1 Add a `checkmate-alpha1` readiness command that validates required reports, OpenSpec task gates, config preset, docs, and artifact manifest.
- [x] 7.2 Add alpha artifact manifest generation for smoke and candidate runs.
- [x] 7.3 Update `docs/VM_RUNBOOK.md` with low-cost smoke, CUDA microbench, two-rank DDP bench, ten-minute smoke, one-hour alpha candidate, artifact download, and teardown steps.
- [x] 7.4 Add readiness tests for missing reports, missing manifest files, incomplete OpenSpec gates, and passing local-safe state.
- [x] 7.5 Add `docs/checkmate-alpha1.md` with the exact acceptance gates, expected artifacts, and known gaps format.

## 8. Verification

- [x] 8.1 Run local `uv run pytest` and `uv run ruff check src tests` without GPU extras.
- [x] 8.2 Run local-safe `alpha-bench` and confirm no CUDA workload or background process is started.
- [x] 8.3 Run single-GPU CUDA benchmarks on a low-cost VM and save the JSON report.
- [x] 8.4 Run two-rank DDP benchmark on a dual-GPU VM and save the JSON report.
- [ ] 8.5 Run the ten-minute integrated DDP smoke and save throughput, loss, profile, and manifest artifacts.
- [ ] 8.6 Run the one-hour alpha candidate only after readiness gaps from the ten-minute smoke are resolved.
