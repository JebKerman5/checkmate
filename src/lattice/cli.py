from __future__ import annotations

import platform
import sys
from importlib.util import find_spec
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from lattice.config import LatticeConfig
from lattice.runtime import (
    RuntimeCheckError,
    destroy_process_group,
    estimate_buffer_plan,
    format_buffer_plan,
    initialize_process_group,
    mib,
    read_rank_env,
    validate_topology,
)

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def doctor() -> None:
    """Print the local environment and core LATTICE defaults."""

    cfg = LatticeConfig()
    table = Table(title="LATTICE Environment")
    table.add_column("Check")
    table.add_column("Value")

    table.add_row("Python", sys.version.split()[0])
    table.add_row("Executable", sys.executable)
    table.add_row("Platform", platform.platform())
    table.add_row("DDP world size", str(cfg.runtime.required_world_size))
    table.add_row("DDP backend", cfg.runtime.backend)
    table.add_row("Batch games / rank", f"{cfg.board.games:,}")
    table.add_row("Max moves", str(cfg.board.max_moves))
    table.add_row("Model", f"{cfg.model.name}, d={cfg.model.d_model}, blocks={cfg.model.blocks}")
    table.add_row("Replay / rank", f"{cfg.replay.records:,} records")
    table.add_row("Learner batch / rank", f"{cfg.replay.learner_batch:,}")

    torch_spec = find_spec("torch")
    if torch_spec is None:
        table.add_row("Torch", "not installed; run `uv sync --extra gpu`")
    else:
        import torch

        cuda = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda else 0
        device = torch.cuda.get_device_name(0) if cuda else "no CUDA device"
        table.add_row("Torch", torch.__version__)
        table.add_row("CUDA", f"{cuda} ({device_count} visible; first: {device})")
        table.add_row("BF16", str(torch.cuda.is_bf16_supported() if cuda else False))

        errors = validate_topology(cfg)
        table.add_row("DDP topology", "ok" if not errors else "; ".join(errors))

    plan = estimate_buffer_plan(cfg)
    table.add_row("Planned buffers / rank", f"{mib(plan.total_bytes):,.1f} MiB")

    console.print(table)


@app.command("buffer-plan")
def buffer_plan() -> None:
    """Print planned static GPU buffer sizes per rank without allocating them."""

    table = Table(title="LATTICE Planned Buffers Per Rank")
    table.add_column("Buffer")
    table.add_column("Size")

    for name, size in format_buffer_plan(estimate_buffer_plan(LatticeConfig())):
        table.add_row(name, size)

    console.print(table)


@app.command("ddp-smoke")
def ddp_smoke() -> None:
    """Initialize DDP, validate runtime topology, print rank buffers, and exit."""

    cfg = LatticeConfig()
    rank, local_rank, world_size = read_rank_env()
    console.print(
        f"Starting DDP smoke rank={rank} local_rank={local_rank} world_size={world_size}"
    )

    try:
        rank_info = initialize_process_group(cfg)
        table = Table(title=f"LATTICE DDP Smoke Rank {rank_info.rank}")
        table.add_column("Check")
        table.add_column("Value")
        table.add_row("rank", str(rank_info.rank))
        table.add_row("local rank", str(rank_info.local_rank))
        table.add_row("world size", str(rank_info.world_size))
        table.add_row("device", f"cuda:{rank_info.device_index} {rank_info.device_name}")
        table.add_row("BF16", str(rank_info.bf16_supported))
        table.add_row("free VRAM", f"{mib(rank_info.free_memory_bytes):,.1f} MiB")
        table.add_row("total VRAM", f"{mib(rank_info.total_memory_bytes):,.1f} MiB")

        for name, size in format_buffer_plan(estimate_buffer_plan(cfg)):
            table.add_row(f"planned {name}", size)

        console.print(table)
    except RuntimeCheckError as exc:
        console.print(f"DDP smoke failed: {exc}")
        raise typer.Exit(1) from exc
    finally:
        destroy_process_group()


@app.command("ddp-verify-sync")
def ddp_verify_sync() -> None:
    """Run one synthetic DDP learner step and verify rank weight checksums match."""

    cfg = LatticeConfig()
    try:
        rank_info = initialize_process_group(cfg)
        from lattice.ddp_verify import verify_ddp_weight_sync

        result = verify_ddp_weight_sync(cfg, rank_info)
        console.print(
            "DDP sync "
            f"rank={result.rank} checksum={result.checksum:.6f} "
            f"max_delta={result.max_checksum_delta:.6g} synced={result.synced}"
        )
        if not result.synced:
            raise typer.Exit(1)
    except RuntimeCheckError as exc:
        console.print(f"DDP sync verification failed: {exc}")
        raise typer.Exit(1) from exc
    finally:
        destroy_process_group()


@app.command("movegen-bench")
def movegen_bench(iterations: int = 10) -> None:
    """Run the lightweight reference-backed movegen benchmark."""

    from lattice.chess.benchmark import benchmark_reference_movegen

    result = benchmark_reference_movegen(["startpos"], iterations=iterations)
    table = Table(title="LATTICE Movegen Benchmark")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("positions", str(result.positions))
    table.add_row("iterations", str(result.iterations))
    table.add_row("elapsed", f"{result.elapsed_seconds:.6f}s")
    table.add_row("board steps/s", f"{result.board_steps_per_second:,.1f}")
    table.add_row("host syncs", str(result.host_syncs))
    console.print(table)


@app.command("synthetic-smoke")
def synthetic_smoke(seconds: float = 1.0, device: str = "cpu") -> None:
    """Run a short synthetic learner smoke loop. Use CUDA only when explicitly requested."""

    from lattice.trainer import run_synthetic_smoke

    summary = run_synthetic_smoke(LatticeConfig(), device=device, seconds=seconds)
    table = Table(title="LATTICE Synthetic Smoke")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("seconds", f"{summary.seconds:.3f}")
    table.add_row("learner steps", str(summary.learner_steps))
    table.add_row("steps/s", f"{summary.learner_steps_per_second:.3f}")
    table.add_row("replay fill", f"{summary.replay_fill:.3f}")
    table.add_row("tripwires", ", ".join(summary.tripwire_failures) or "none")
    console.print(table)


@app.command("ddp-synthetic-smoke")
def ddp_synthetic_smoke(seconds: float = 10.0) -> None:
    """Run a short per-rank synthetic smoke loop under torchrun."""

    cfg = LatticeConfig()
    try:
        rank_info = initialize_process_group(cfg)
        from lattice.trainer import run_synthetic_smoke

        summary = run_synthetic_smoke(
            cfg,
            device=f"cuda:{rank_info.device_index}",
            seconds=seconds,
        )
        console.print(
            "DDP synthetic smoke "
            f"rank={rank_info.rank} seconds={summary.seconds:.3f} "
            f"steps={summary.learner_steps} steps/s={summary.learner_steps_per_second:.3f} "
            f"replay_fill={summary.replay_fill:.3f}"
        )
    except RuntimeCheckError as exc:
        console.print(f"DDP synthetic smoke failed: {exc}")
        raise typer.Exit(1) from exc
    finally:
        destroy_process_group()


@app.command("profile-summary")
def profile_summary(
    gpu_sm_busy: float | None = None,
    cpu_cores: float | None = None,
    host_syncs: int | None = None,
    labeled_positions_per_second: float | None = None,
) -> None:
    """Summarize profiler metrics and report acceptance gaps."""

    from lattice.profiling import ProfileSummary, acceptance_gaps, summarize_profile

    summary = ProfileSummary(
        gpu_sm_busy=gpu_sm_busy,
        cpu_cores=cpu_cores,
        host_syncs=host_syncs,
        labeled_positions_per_second=labeled_positions_per_second,
    )
    table = Table(title="LATTICE Profile Summary")
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in summarize_profile(summary).items():
        table.add_row(key, "unset" if value is None else str(value))
    gaps = acceptance_gaps(summary)
    table.add_row("acceptance gaps", ", ".join(gaps) or "none")
    console.print(table)


@app.command("checkpoint-info")
def checkpoint_info(path: Path) -> None:
    """Print checkpoint metadata without starting training."""

    from lattice.checkpoint import load_checkpoint

    payload = load_checkpoint(path)
    console.print(payload.get("meta", "checkpoint has no meta field"))


if __name__ == "__main__":
    app()
