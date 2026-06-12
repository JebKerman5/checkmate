# Dual-GPU VM Runbook

Use these commands on the dual-GPU VM, not on the gaming PC.

## Setup

```powershell
uv sync --extra dev --extra gpu
uv run lattice doctor
```

## Runtime Checks

```powershell
uv run torchrun --nproc_per_node=2 -m lattice.cli ddp-smoke
uv run torchrun --nproc_per_node=2 -m lattice.cli ddp-verify-sync
```

## Low-Cost Single-GPU Smoke

Use this on a cheap one-GPU VM before paying for the dual-GPU run.

```powershell
uv run lattice alpha-bench --profile cuda --preset single-gpu-smoke --seconds 1 --output runs/checkmate-alpha1/single-gpu-alpha-bench.json
uv run lattice synthetic-smoke --seconds 1 --device cuda --sample-batch 128 --replay-capacity 4096
```

## Dual-GPU Alpha Benchmarks

Run these only on the target dual-GPU VM.

```powershell
uv run torchrun --nproc_per_node=2 -m lattice.cli ddp-alpha-bench --seconds 5 --output runs/checkmate-alpha1/ddp-alpha-bench.json
uv run lattice profile-summary --report runs/checkmate-alpha1/ddp-alpha-bench.json
```

## Synthetic DDP Smoke

Short run:

```powershell
uv run torchrun --nproc_per_node=2 -m lattice.cli ddp-synthetic-smoke --seconds 30
```

Ten-minute run for the OpenSpec smoke task:

```powershell
uv run torchrun --nproc_per_node=2 -m lattice.cli ddp-synthetic-smoke --seconds 600
```

## Profiling

Capture Nsight or PyTorch profiler output on the VM, then summarize key numbers:

```powershell
uv run lattice profile-summary --gpu-sm-busy 0.90 --cpu-cores 1.0 --host-syncs 0 --labeled-positions-per-second 2500
```

The current implementation keeps the production CUDA/Triton chess hot path as the next real
engineering step. Do not treat the reference-backed movegen prototype as the final kernel.

## Alpha Candidate Manifest

After the smoke or candidate run writes reports/checkpoints, create a manifest:

```powershell
uv run lattice alpha-manifest --run-dir runs/checkmate-alpha1 --vm-shape "2x GPU VM" --benchmark-report runs/checkmate-alpha1/ddp-alpha-bench.json
uv run lattice checkmate-alpha1 --run-dir runs/checkmate-alpha1 --output runs/checkmate-alpha1/readiness.json
```

Download or commit only the small summary artifacts you need. Large checkpoints
should stay in ignored run storage or external storage.
