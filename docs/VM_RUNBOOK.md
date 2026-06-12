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
