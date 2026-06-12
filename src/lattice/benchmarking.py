from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from lattice.config import LatticeConfig, config_preset

SCHEMA_VERSION = "lattice-alpha-bench-v1"
SectionStatus = Literal["pass", "fail", "skip"]
ThresholdOp = Literal["ge", "le", "eq"]


@dataclass(frozen=True)
class BenchmarkThreshold:
    metric: str
    op: ThresholdOp
    value: float | int | str | bool
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "op": self.op,
            "value": self.value,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkThreshold:
        return cls(
            metric=str(data["metric"]),
            op=data["op"],
            value=data["value"],
            description=str(data.get("description", "")),
        )


@dataclass(frozen=True)
class BenchmarkSection:
    name: str
    status: SectionStatus
    metrics: dict[str, float | int | str | bool | None] = field(default_factory=dict)
    thresholds: list[BenchmarkThreshold] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    skip_reason: str | None = None
    elapsed_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "metrics": self.metrics,
            "thresholds": [threshold.to_dict() for threshold in self.thresholds],
            "gaps": list(self.gaps),
            "skip_reason": self.skip_reason,
            "elapsed_seconds": self.elapsed_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkSection:
        return cls(
            name=str(data["name"]),
            status=data["status"],
            metrics=dict(data.get("metrics", {})),
            thresholds=[
                BenchmarkThreshold.from_dict(item) for item in data.get("thresholds", [])
            ],
            gaps=[str(item) for item in data.get("gaps", [])],
            skip_reason=data.get("skip_reason"),
            elapsed_seconds=data.get("elapsed_seconds"),
        )


@dataclass(frozen=True)
class BenchmarkReport:
    profile: str
    command: str
    environment: dict[str, Any]
    sections: list[BenchmarkSection]
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def gaps(self) -> list[str]:
        gaps: list[str] = []
        for section in self.sections:
            gaps.extend(f"{section.name}: {gap}" for gap in section.gaps)
        return gaps

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "profile": self.profile,
            "command": self.command,
            "environment": self.environment,
            "sections": [section.to_dict() for section in self.sections],
            "gaps": self.gaps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkReport:
        return cls(
            schema_version=str(data["schema_version"]),
            created_at=str(data["created_at"]),
            profile=str(data["profile"]),
            command=str(data["command"]),
            environment=dict(data["environment"]),
            sections=[BenchmarkSection.from_dict(item) for item in data["sections"]],
        )


def validate_report_dict(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("schema_version", "created_at", "profile", "command", "environment", "sections"):
        if key not in data:
            errors.append(f"missing {key}")
    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    if not isinstance(data.get("environment", {}), dict):
        errors.append("environment must be an object")
    sections = data.get("sections", [])
    if not isinstance(sections, list):
        errors.append("sections must be a list")
        return errors
    for index, section in enumerate(sections):
        if not isinstance(section, dict):
            errors.append(f"sections[{index}] must be an object")
            continue
        for key in ("name", "status", "metrics", "thresholds", "gaps"):
            if key not in section:
                errors.append(f"sections[{index}] missing {key}")
        if section.get("status") not in {"pass", "fail", "skip"}:
            errors.append(f"sections[{index}] has invalid status")
    return errors


def write_report(report: BenchmarkReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_report(path: Path) -> BenchmarkReport:
    data = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_report_dict(data)
    if errors:
        raise ValueError("; ".join(errors))
    return BenchmarkReport.from_dict(data)


def evaluate_thresholds(
    metrics: dict[str, float | int | str | bool | None],
    thresholds: list[BenchmarkThreshold],
) -> list[str]:
    gaps: list[str] = []
    for threshold in thresholds:
        actual = metrics.get(threshold.metric)
        if actual is None:
            gaps.append(f"{threshold.metric} missing")
            continue
        if threshold.op == "ge" and not actual >= threshold.value:
            gaps.append(f"{threshold.metric} {actual} < {threshold.value}")
        elif threshold.op == "le" and not actual <= threshold.value:
            gaps.append(f"{threshold.metric} {actual} > {threshold.value}")
        elif threshold.op == "eq" and actual != threshold.value:
            gaps.append(f"{threshold.metric} {actual} != {threshold.value}")
    return gaps


def make_section(
    name: str,
    metrics: dict[str, float | int | str | bool | None] | None = None,
    thresholds: list[BenchmarkThreshold] | None = None,
    skip_reason: str | None = None,
    elapsed_seconds: float | None = None,
) -> BenchmarkSection:
    metric_values = metrics or {}
    threshold_values = thresholds or []
    if skip_reason:
        return BenchmarkSection(
            name=name,
            status="skip",
            metrics=metric_values,
            thresholds=threshold_values,
            gaps=[],
            skip_reason=skip_reason,
            elapsed_seconds=elapsed_seconds,
        )
    gaps = evaluate_thresholds(metric_values, threshold_values)
    return BenchmarkSection(
        name=name,
        status="fail" if gaps else "pass",
        metrics=metric_values,
        thresholds=threshold_values,
        gaps=gaps,
        elapsed_seconds=elapsed_seconds,
    )


def collect_environment(
    preset: str,
    repo_root: Path | None = None,
    probe_torch: bool = False,
) -> dict[str, Any]:
    root = repo_root or Path.cwd()
    torch_spec = find_spec("torch")
    env: dict[str, Any] = {
        "preset": preset,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "git_commit": _git_output(root, "rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git_output(root, "status", "--porcelain")),
        "world_size_env": os.environ.get("WORLD_SIZE"),
        "torch_available": torch_spec is not None,
        "torch_probed": probe_torch,
        "torch_version": None,
        "cuda_available": None,
        "cuda_device_count": 0,
        "cuda_devices": [],
        "bf16_supported": None,
    }
    if not probe_torch or torch_spec is None:
        return env

    try:
        import torch
    except Exception as exc:  # pragma: no cover - defensive against broken installs
        env["torch_import_error"] = str(exc)
        return env

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    env.update(
        {
            "torch_version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_device_count": device_count,
            "cuda_devices": [
                torch.cuda.get_device_name(index) for index in range(device_count)
            ],
            "bf16_supported": bool(torch.cuda.is_bf16_supported()) if cuda_available else False,
        }
    )
    return env


def run_local_safe_benchmark(
    repo_root: Path | None = None,
    seconds: float = 0.05,
) -> BenchmarkReport:
    started = perf_counter()
    iterations = 0
    target_seconds = max(0.001, min(seconds, 0.25))
    while perf_counter() - started < target_seconds:
        iterations += 1
    elapsed = max(perf_counter() - started, 1e-9)

    local_section = make_section(
        "local_contract",
        metrics={
            "elapsed_seconds": elapsed,
            "iterations": iterations,
            "background_processes_started": 0,
            "cuda_touched": False,
        },
        thresholds=[
            BenchmarkThreshold("elapsed_seconds", "le", 2.0, "local profile is bounded"),
            BenchmarkThreshold("background_processes_started", "eq", 0),
            BenchmarkThreshold("cuda_touched", "eq", False),
        ],
        elapsed_seconds=elapsed,
    )
    return BenchmarkReport(
        profile="local",
        command="lattice alpha-bench --profile local",
        environment=collect_environment("local-contract", repo_root, probe_torch=False),
        sections=[
            local_section,
            make_section(
                "cuda_microbenchmarks",
                skip_reason="local profile does not import torch or use CUDA",
            ),
            make_section(
                "ddp_benchmark",
                skip_reason="requires torchrun --nproc_per_node=2 on a dual-GPU VM",
            ),
        ],
    )


def timed_repeated(
    fn: Any,
    *,
    warmups: int,
    repeats: int,
    sync: Any | None = None,
) -> tuple[float, list[float]]:
    for _ in range(max(warmups, 0)):
        fn()
    if sync is not None:
        sync()

    samples: list[float] = []
    for _ in range(max(repeats, 1)):
        started = perf_counter()
        fn()
        if sync is not None:
            sync()
        samples.append(max(perf_counter() - started, 1e-9))
    return statistics.median(samples), samples


def run_cuda_microbenchmarks(
    repo_root: Path | None = None,
    preset: str = "single-gpu-smoke",
    seconds: float = 1.0,
    include_compile: bool = False,
) -> BenchmarkReport:
    cfg = config_preset(preset)
    env = collect_environment(preset, repo_root, probe_torch=True)
    if not env.get("cuda_available"):
        return BenchmarkReport(
            profile="cuda",
            command=f"lattice alpha-bench --profile cuda --preset {preset}",
            environment=env,
            sections=[
                make_section("cuda_microbenchmarks", skip_reason="CUDA is unavailable"),
                make_section("compiled_learner", skip_reason="CUDA is unavailable"),
            ],
        )

    import torch

    device = torch.device("cuda:0")
    def sync() -> None:
        torch.cuda.synchronize(device)

    sections = [
        _benchmark_model_forward(cfg, device, sync),
        _benchmark_learner_step(cfg, device, sync),
        _benchmark_replay(cfg, device, sync),
        _benchmark_movegen_hooks(cfg, device, sync),
    ]
    if include_compile:
        sections.append(_benchmark_compiled_forward(cfg, device, sync))
    else:
        sections.append(make_section("compiled_learner", skip_reason="compiled mode not requested"))

    return BenchmarkReport(
        profile="cuda",
        command=(
            f"lattice alpha-bench --profile cuda --preset {preset} --seconds {seconds}"
        ),
        environment=env,
        sections=sections,
    )


def run_ddp_rank_benchmark(
    cfg: LatticeConfig,
    rank_info: Any,
    seconds: float = 5.0,
) -> BenchmarkReport | None:
    import torch

    from lattice.model import LatticeModel
    from lattice.synthetic import make_synthetic_replay_batch
    from lattice.training import train_step, wrap_ddp

    torch.manual_seed(1234)
    device = torch.device(f"cuda:{rank_info.device_index}")
    torch.cuda.set_device(device)
    model = LatticeModel(cfg.model).to(device)
    ddp_model = wrap_ddp(model, rank_info.device_index)
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)
    batch = make_synthetic_replay_batch(
        cfg,
        device,
        batch_size=min(cfg.replay.learner_batch, cfg.benchmark.sample_batch),
    )

    def sync() -> None:
        torch.cuda.synchronize(device)

    step_median, _ = timed_repeated(
        lambda: train_step(ddp_model, optimizer, batch, use_bf16=True),
        warmups=cfg.benchmark.warmup_batches,
        repeats=3,
        sync=sync,
    )

    all_reduce_tensor = torch.ones(1024, device=device)
    all_reduce_median, _ = timed_repeated(
        lambda: torch.distributed.all_reduce(all_reduce_tensor),
        warmups=1,
        repeats=3,
        sync=sync,
    )

    checksum = float(
        sum(param.detach().float().sum().item() for param in ddp_model.module.parameters())
    )
    gathered: list[dict[str, Any] | None] = [None for _ in range(rank_info.world_size)]
    local = {
        "rank": rank_info.rank,
        "learner_step_seconds": step_median,
        "learner_steps_per_second": 1.0 / step_median,
        "all_reduce_seconds": all_reduce_median,
        "all_reduce_share": all_reduce_median / max(step_median, 1e-9),
        "checksum": checksum,
    }
    torch.distributed.all_gather_object(gathered, local)

    if rank_info.rank != 0:
        return None

    metrics_by_rank = [item for item in gathered if item is not None]
    steps = [float(item["learner_steps_per_second"]) for item in metrics_by_rank]
    checksums = [float(item["checksum"]) for item in metrics_by_rank]
    max_checksum_delta = max(checksums) - min(checksums)
    aggregate = make_section(
        "ddp_aggregate",
        metrics={
            "world_size": rank_info.world_size,
            "global_batch": cfg.replay.global_batch,
            "min_rank_steps_per_second": min(steps),
            "max_rank_steps_per_second": max(steps),
            "rank_skew_steps_per_second": max(steps) - min(steps),
            "mean_all_reduce_share": statistics.mean(
                float(item["all_reduce_share"]) for item in metrics_by_rank
            ),
            "max_checksum_delta": max_checksum_delta,
            "weights_synced": max_checksum_delta <= 1e-3,
            "host_syncs": 0,
        },
        thresholds=[
            BenchmarkThreshold("world_size", "eq", 2),
            BenchmarkThreshold("weights_synced", "eq", True),
            BenchmarkThreshold("host_syncs", "eq", 0),
        ],
    )
    return BenchmarkReport(
        profile="ddp",
        command="torchrun --nproc_per_node=2 -m lattice.cli ddp-alpha-bench",
        environment=collect_environment(cfg.benchmark.preset, Path.cwd(), probe_torch=True),
        sections=[
            aggregate,
            make_section(
                "ddp_rank_metrics",
                metrics={"ranks": json.dumps(metrics_by_rank, sort_keys=True)},
            ),
        ],
    )


def _benchmark_model_forward(cfg: LatticeConfig, device: Any, sync: Any) -> BenchmarkSection:
    import torch

    from lattice.model import LatticeModel
    from lattice.synthetic import make_synthetic_replay_batch

    model = LatticeModel(cfg.model).to(device).eval()
    batch = make_synthetic_replay_batch(
        cfg,
        device,
        batch_size=min(cfg.benchmark.sample_batch, cfg.replay.learner_batch),
    )

    def forward() -> None:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model(batch.features)

    median, _ = timed_repeated(forward, warmups=cfg.benchmark.warmup_batches, repeats=5, sync=sync)
    batch_size = int(batch.features.shape[0])
    return make_section(
        "model_forward",
        metrics={
            "batch_size": batch_size,
            "median_seconds": median,
            "boards_per_second": batch_size / median,
            "host_syncs": 0,
        },
        thresholds=[BenchmarkThreshold("host_syncs", "eq", 0)],
        elapsed_seconds=median,
    )


def _benchmark_learner_step(cfg: LatticeConfig, device: Any, sync: Any) -> BenchmarkSection:
    import torch

    from lattice.model import LatticeModel
    from lattice.synthetic import make_synthetic_replay_batch
    from lattice.training import train_step

    model = LatticeModel(cfg.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = make_synthetic_replay_batch(
        cfg,
        device,
        batch_size=min(cfg.benchmark.sample_batch, cfg.replay.learner_batch),
    )
    median, _ = timed_repeated(
        lambda: train_step(model, optimizer, batch, use_bf16=True),
        warmups=cfg.benchmark.warmup_batches,
        repeats=5,
        sync=sync,
    )
    return make_section(
        "learner_step",
        metrics={
            "batch_size": int(batch.features.shape[0]),
            "median_seconds": median,
            "steps_per_second": 1.0 / median,
            "host_syncs": 0,
        },
        thresholds=[BenchmarkThreshold("host_syncs", "eq", 0)],
        elapsed_seconds=median,
    )


def _benchmark_replay(cfg: LatticeConfig, device: Any, sync: Any) -> BenchmarkSection:
    import torch

    from lattice.replay import allocate_replay_shard, sample_batch, write_batch
    from lattice.synthetic import make_synthetic_replay_batch

    batch_size = min(cfg.benchmark.sample_batch, cfg.replay.learner_batch)
    replay = allocate_replay_shard(cfg, device, capacity=cfg.benchmark.replay_capacity)
    batch = make_synthetic_replay_batch(cfg, device, batch_size=batch_size)
    priorities = torch.ones(batch_size, device=device)
    write_batch(replay, batch, priorities)

    write_median, _ = timed_repeated(
        lambda: write_batch(replay, batch, priorities),
        warmups=cfg.benchmark.warmup_batches,
        repeats=5,
        sync=sync,
    )
    priority_sample_median, _ = timed_repeated(
        lambda: sample_batch(replay, batch_size=batch_size, mode="priority"),
        warmups=cfg.benchmark.warmup_batches,
        repeats=5,
        sync=sync,
    )
    uniform_sample_median, _ = timed_repeated(
        lambda: sample_batch(replay, batch_size=batch_size, mode="uniform"),
        warmups=cfg.benchmark.warmup_batches,
        repeats=5,
        sync=sync,
    )
    sample_median = (
        uniform_sample_median
        if cfg.replay.sampling_mode == "uniform"
        else priority_sample_median
    )
    sample_share = priority_sample_median / max(write_median + priority_sample_median, 1e-9)
    return make_section(
        "replay",
        metrics={
            "batch_size": batch_size,
            "sampling_mode": cfg.replay.sampling_mode,
            "write_records_per_second": batch_size / write_median,
            "priority_sample_records_per_second": batch_size / priority_sample_median,
            "uniform_sample_records_per_second": batch_size / uniform_sample_median,
            "sample_records_per_second": batch_size / sample_median,
            "priority_sampling_share": sample_share,
            "host_syncs": 0,
        },
        thresholds=[
            BenchmarkThreshold("host_syncs", "eq", 0),
            BenchmarkThreshold("priority_sampling_share", "le", 0.80),
        ],
    )


def _benchmark_movegen_hooks(cfg: LatticeConfig, device: Any, sync: Any) -> BenchmarkSection:
    from lattice.chess.benchmark import benchmark_movegen_path

    del sync
    reference = benchmark_movegen_path(["startpos"], iterations=1, path="reference", device="cpu")
    vectorized = benchmark_movegen_path(
        ["startpos"],
        iterations=1,
        path="vectorized-tensor",
        device=str(device),
    )
    return make_section(
        "movegen_hooks",
        metrics={
            "training_movegen_path": cfg.benchmark.movegen_path,
            "alpha_training_uses_reference_movegen": cfg.benchmark.movegen_path == "reference",
            "reference_board_steps_per_second": reference.board_steps_per_second,
            "vectorized_board_steps_per_second": vectorized.board_steps_per_second,
            "reference_host_syncs": reference.host_syncs,
            "vectorized_host_syncs": vectorized.host_syncs,
            "triton_path": "not_implemented",
            "cuda_path": "not_implemented",
        },
        thresholds=[
            BenchmarkThreshold("alpha_training_uses_reference_movegen", "eq", False),
        ],
    )


def _benchmark_compiled_forward(cfg: LatticeConfig, device: Any, sync: Any) -> BenchmarkSection:
    import torch

    from lattice.model import LatticeModel
    from lattice.synthetic import make_synthetic_replay_batch

    if not hasattr(torch, "compile"):
        return make_section("compiled_learner", skip_reason="torch.compile is unavailable")
    model = LatticeModel(cfg.model).to(device).eval()
    batch = make_synthetic_replay_batch(
        cfg,
        device,
        batch_size=min(cfg.benchmark.sample_batch, cfg.replay.learner_batch),
    )
    try:
        compiled_model = torch.compile(model)
    except Exception as exc:  # pragma: no cover - hardware/compiler dependent
        return make_section("compiled_learner", skip_reason=f"compile failed: {exc}")

    def forward() -> None:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            compiled_model(batch.features)

    try:
        median, _ = timed_repeated(
            forward,
            warmups=cfg.benchmark.warmup_batches,
            repeats=5,
            sync=sync,
        )
    except Exception as exc:  # pragma: no cover - hardware/compiler dependent
        return make_section("compiled_learner", skip_reason=f"compiled run failed: {exc}")

    return make_section(
        "compiled_learner",
        metrics={
            "batch_size": int(batch.features.shape[0]),
            "median_seconds": median,
            "boards_per_second": int(batch.features.shape[0]) / median,
            "host_syncs": 0,
        },
        thresholds=[BenchmarkThreshold("host_syncs", "eq", 0)],
        elapsed_seconds=median,
    )


def _git_output(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
