from __future__ import annotations

import platform
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Annotated, Any

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
BENCHMARK_REPORT_OPTION = typer.Option("--benchmark-report")
PROFILE_SUMMARY_OPTION = typer.Option("--profile-summary")
CHECKPOINT_OPTION = typer.Option("--checkpoint")
LOSS_CURVE_OPTION = typer.Option("--loss-curve")
THROUGHPUT_SUMMARY_OPTION = typer.Option("--throughput-summary")
KNOWN_GAP_OPTION = typer.Option("--known-gap")


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
def synthetic_smoke(
    seconds: float = 1.0,
    device: str = "cpu",
    replay_capacity: int = 1024,
    sample_batch: int = 32,
    warmup_batches: int = 1,
    static_batch: bool = False,
    compile_model: bool = False,
) -> None:
    """Run a short synthetic learner smoke loop. Use CUDA only when explicitly requested."""

    from lattice.trainer import run_synthetic_smoke

    summary = run_synthetic_smoke(
        LatticeConfig(),
        device=device,
        seconds=seconds,
        replay_capacity=replay_capacity,
        sample_batch_size=sample_batch,
        warmup_batches=warmup_batches,
        static_batch=static_batch,
        use_compile=compile_model,
    )
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
def ddp_synthetic_smoke(
    seconds: float = 10.0,
    sample_batch: int | None = None,
    replay_capacity: int | None = None,
) -> None:
    """Run a short per-rank synthetic smoke loop under torchrun."""

    cfg = LatticeConfig()
    try:
        rank_info = initialize_process_group(cfg)
        from lattice.trainer import run_synthetic_smoke

        summary = run_synthetic_smoke(
            cfg,
            device=f"cuda:{rank_info.device_index}",
            seconds=seconds,
            replay_capacity=replay_capacity or cfg.benchmark.replay_capacity,
            sample_batch_size=sample_batch or cfg.benchmark.sample_batch,
            warmup_batches=cfg.benchmark.warmup_batches,
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


@app.command("alpha-bench")
def alpha_bench(
    profile: str = "local",
    output: Path | None = None,
    preset: str = "local-contract",
    seconds: float = 0.05,
    compile_model: bool = False,
) -> None:
    """Run alpha benchmark profiles. The default local profile is CPU-safe."""

    from lattice.benchmarking import (
        run_cuda_microbenchmarks,
        run_local_safe_benchmark,
        write_report,
    )

    normalized = profile.lower().replace("_", "-")
    if normalized == "local":
        report = run_local_safe_benchmark(Path.cwd(), seconds=seconds)
    elif normalized == "cuda":
        report = run_cuda_microbenchmarks(
            Path.cwd(),
            preset=preset,
            seconds=seconds,
            include_compile=compile_model,
        )
    else:
        raise typer.BadParameter("profile must be local or cuda")

    _print_benchmark_report(report)
    if output is not None:
        write_report(report, output)
        console.print(f"Wrote benchmark report: {output}")


@app.command("ddp-alpha-bench")
def ddp_alpha_bench(
    output: Path = Path("runs/checkmate-alpha1/ddp-alpha-bench.json"),
    seconds: float = 5.0,
    preset: str = "dual-gpu-ddp-smoke",
) -> None:
    """Run the alpha DDP benchmark under torchrun on a dual-GPU VM."""

    from lattice.benchmarking import run_ddp_rank_benchmark, write_report
    from lattice.config import config_preset

    cfg = config_preset(preset)
    try:
        rank_info = initialize_process_group(cfg)
        report = run_ddp_rank_benchmark(cfg, rank_info, seconds=seconds)
        if report is not None:
            write_report(report, output)
            _print_benchmark_report(report)
            console.print(f"Wrote DDP benchmark report: {output}")
    except RuntimeCheckError as exc:
        console.print(f"DDP alpha benchmark failed: {exc}")
        raise typer.Exit(1) from exc
    finally:
        destroy_process_group()


@app.command("profile-summary")
def profile_summary(
    gpu_sm_busy: float | None = None,
    cpu_cores: float | None = None,
    host_syncs: int | None = None,
    labeled_positions_per_second: float | None = None,
    report: Path | None = None,
) -> None:
    """Summarize profiler metrics and report acceptance gaps."""

    from lattice.profiling import (
        ProfileSummary,
        acceptance_gaps,
        acceptance_gaps_from_benchmark,
        profile_from_benchmark,
        summarize_profile,
    )

    if report is None:
        summary = ProfileSummary(
            gpu_sm_busy=gpu_sm_busy,
            cpu_cores=cpu_cores,
            host_syncs=host_syncs,
            labeled_positions_per_second=labeled_positions_per_second,
        )
        gaps = acceptance_gaps(summary)
    else:
        summary = profile_from_benchmark(report)
        gaps = acceptance_gaps_from_benchmark(report)

    table = Table(title="LATTICE Profile Summary")
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in summarize_profile(summary).items():
        table.add_row(key, "unset" if value is None else str(value))
    table.add_row("acceptance gaps", ", ".join(gaps) or "none")
    console.print(table)


@app.command("checkpoint-info")
def checkpoint_info(path: Path) -> None:
    """Print checkpoint metadata without starting training."""

    from lattice.checkpoint import load_checkpoint

    payload = load_checkpoint(path)
    console.print(payload.get("meta", "checkpoint has no meta field"))


@app.command("alpha-manifest")
def alpha_manifest(
    run_dir: Path = Path("runs/checkmate-alpha1"),
    run_id: str = "checkmate-alpha1",
    config_preset: str = "checkmate-alpha1",
    vm_shape: str = "unset",
    benchmark_report: Annotated[list[Path] | None, BENCHMARK_REPORT_OPTION] = None,
    profile_summary_path: Annotated[Path | None, PROFILE_SUMMARY_OPTION] = None,
    checkpoint: Annotated[list[Path] | None, CHECKPOINT_OPTION] = None,
    loss_curve: Annotated[list[Path] | None, LOSS_CURVE_OPTION] = None,
    throughput_summary: Annotated[list[Path] | None, THROUGHPUT_SUMMARY_OPTION] = None,
    known_gap: Annotated[list[str] | None, KNOWN_GAP_OPTION] = None,
) -> None:
    """Write an alpha artifact manifest for a completed smoke or candidate run."""

    from lattice.alpha import AlphaManifest, write_manifest
    from lattice.benchmarking import collect_environment

    env = collect_environment(config_preset, Path.cwd(), probe_torch=False)
    manifest = AlphaManifest(
        run_id=run_id,
        git_commit=env.get("git_commit"),
        config_preset=config_preset,
        vm_shape=vm_shape,
        benchmark_reports=[
            _relative_to_run_dir(run_dir, path) for path in benchmark_report or []
        ],
        profile_summary=(
            _relative_to_run_dir(run_dir, profile_summary_path)
            if profile_summary_path is not None
            else None
        ),
        checkpoints=[_relative_to_run_dir(run_dir, path) for path in checkpoint or []],
        loss_curves=[_relative_to_run_dir(run_dir, path) for path in loss_curve or []],
        throughput_summaries=[
            _relative_to_run_dir(run_dir, path) for path in throughput_summary or []
        ],
        known_gaps=list(known_gap or []),
    )
    path = write_manifest(run_dir, manifest)
    console.print(f"Wrote alpha manifest: {path}")


@app.command("checkmate-alpha1")
def checkmate_alpha1(
    run_dir: Path = Path("runs/checkmate-alpha1"),
    output: Path | None = None,
    require_vm_artifacts: bool = True,
    require_openspec_complete: bool = True,
) -> None:
    """Validate whether the repository is ready to tag and run checkmate-alpha1."""

    from lattice.alpha import evaluate_alpha_readiness

    result = evaluate_alpha_readiness(
        Path.cwd(),
        run_dir,
        require_vm_artifacts=require_vm_artifacts,
        require_openspec_complete=require_openspec_complete,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        import json

        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        console.print(f"Wrote alpha readiness report: {output}")

    table = Table(title="checkmate-alpha1 readiness")
    table.add_column("Check")
    table.add_column("Value")
    table.add_row("ready", str(result["ready"]))
    gaps = result["gaps"]
    table.add_row("gaps", "\n".join(gaps) if gaps else "none")
    console.print(table)
    if not result["ready"]:
        raise typer.Exit(1)


def _print_benchmark_report(report: Any) -> None:
    table = Table(title=f"LATTICE Alpha Benchmark ({report.profile})")
    table.add_column("Section")
    table.add_column("Status")
    table.add_column("Key Metrics")
    table.add_column("Gaps / Skip")
    for section in report.sections:
        metrics = ", ".join(f"{key}={value}" for key, value in section.metrics.items())
        detail = section.skip_reason or ", ".join(section.gaps) or "none"
        table.add_row(section.name, section.status, metrics[:120], detail[:120])
    console.print(table)


def _relative_to_run_dir(run_dir: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(run_dir.resolve()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    app()
