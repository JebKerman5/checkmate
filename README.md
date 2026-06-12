# LATTICE Checkmate

This repository implements the LATTICE spec in `LATTICE-SPEC.md` as a dual-GPU,
PyTorch DDP project. Each rank owns one GPU and runs generation, labeling, replay,
and learner work locally. DDP synchronizes model gradients through NCCL.

## Environment

This project uses `uv` and keeps dependencies in `.venv`.

```powershell
uv venv --python 3.12
uv sync --extra dev
```

Install the GPU extra only after the base environment works:

```powershell
uv sync --extra dev --extra gpu
```

Run the environment check:

```powershell
uv run lattice doctor
```

The runtime smoke command validates the dual-GPU topology and exits. It does not
start a long training job.

```powershell
uv run torchrun --nproc_per_node=2 -m lattice.cli ddp-smoke
```

Print the planned per-rank GPU buffer sizes without allocating them:

```powershell
uv run lattice buffer-plan
```

VM-only commands live in `docs/VM_RUNBOOK.md`. Do not run timed smoke or profiler
jobs on a desktop used for gaming unless you intend to occupy the GPUs.

## Build Order

1. DDP runtime skeleton and dual-GPU smoke command.
2. LATTICE-base model and synthetic DDP learner.
3. GPU chess state, legal move generation, move application, terminal checks,
   CPU differential tests, and throughput profiling.
4. Rank-local VRAM replay and sparse negamax labeling.
5. Integrated self-play training loop.
6. Evaluation, profiling, and CUDA/Triton hot-path optimization.
