# checkmate-alpha1

`checkmate-alpha1` is a readiness gate for the first serious dual-GPU DDP run.
It is not ready just because code exists; it is ready when the required commands
produce the expected artifacts and remaining gaps are explicit.

## Required Gates

- Local tests and ruff pass without GPU extras.
- `lattice alpha-bench --profile local` writes a valid JSON report without
  importing torch or touching CUDA.
- Single-GPU CUDA benchmark runs on a low-cost VM and writes JSON.
- Two-rank DDP benchmark runs on the target VM and writes JSON from rank 0.
- Ten-minute integrated DDP smoke writes throughput, loss, profile, and manifest
  artifacts.
- Profile summary reports GPU/CPU/host-sync gaps directly from benchmark JSON.
- Alpha readiness command validates docs, config, manifests, benchmark reports,
  and OpenSpec task gates.

## Expected Artifacts

All paths are relative to `runs/checkmate-alpha1/` unless noted.

- `single-gpu-alpha-bench.json`
- `ddp-alpha-bench.json`
- `profile-summary.json` or equivalent profile report
- throughput summary for labeled positions/s and learner steps/s
- loss curve summary
- checkpoint metadata for any saved checkpoint
- `manifest.json`
- `readiness.json`

## Known Gaps Format

Record each gap as a concrete action, not a general concern:

```text
GPU SM busy below 90% on 2xA6000: profile shows replay priority sampling dominates.
CPU reference movegen still enabled in alpha profile: replace before one-hour candidate.
```

## Local Safety

The local PC path is only for contract checks and docs. CUDA microbenchmarks,
DDP benchmarks, timed smokes, and candidate training runs are VM-only unless the
user explicitly chooses to occupy local GPUs.
