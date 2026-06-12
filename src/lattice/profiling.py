from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

PROFILE_EVENT_NAMES = (
    "generation",
    "labeling",
    "replay_write",
    "replay_sample",
    "learner_forward_backward",
    "optimizer_step",
    "ema_update",
    "all_reduce",
)


@dataclass(frozen=True)
class ProfileSummary:
    gpu_sm_busy: float | None = None
    tensor_core_utilization: float | None = None
    cpu_cores: float | None = None
    all_reduce_share: float | None = None
    labeled_positions_per_second: float | None = None
    learner_steps_per_second: float | None = None
    host_syncs: int | None = None


@dataclass(frozen=True)
class ProfileEvent:
    name: str
    elapsed_seconds: float
    metadata: dict[str, float | int | str | bool] = field(default_factory=dict)


@dataclass
class ProfileRecorder:
    events: list[ProfileEvent] = field(default_factory=list)

    def record(
        self,
        name: str,
        elapsed_seconds: float,
        metadata: dict[str, float | int | str | bool] | None = None,
    ) -> None:
        self.events.append(ProfileEvent(name, elapsed_seconds, metadata or {}))

    def timed(self, name: str) -> ProfileTimer:
        return ProfileTimer(self, name)


@dataclass
class ProfileTimer:
    recorder: ProfileRecorder
    name: str
    started: float = 0.0

    def __enter__(self) -> ProfileTimer:
        self.started = perf_counter()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.recorder.record(self.name, max(perf_counter() - self.started, 1e-9))


@dataclass
class HostSyncCounter:
    count: int = 0
    sites: dict[str, int] = field(default_factory=dict)

    def mark(self, site: str) -> None:
        self.count += 1
        self.sites[site] = self.sites.get(site, 0) + 1


def summarize_profile(summary: ProfileSummary) -> dict[str, float | int | None]:
    return {
        "gpu_sm_busy": summary.gpu_sm_busy,
        "tensor_core_utilization": summary.tensor_core_utilization,
        "cpu_cores": summary.cpu_cores,
        "all_reduce_share": summary.all_reduce_share,
        "labeled_positions_per_second": summary.labeled_positions_per_second,
        "learner_steps_per_second": summary.learner_steps_per_second,
        "host_syncs": summary.host_syncs,
    }


def acceptance_gaps(summary: ProfileSummary) -> list[str]:
    gaps: list[str] = []
    if summary.gpu_sm_busy is not None and summary.gpu_sm_busy < 0.90:
        gaps.append("GPU SM busy below 90%")
    if summary.cpu_cores is not None and summary.cpu_cores > 1.0:
        gaps.append("CPU usage above one core")
    if summary.host_syncs is not None and summary.host_syncs > 0:
        gaps.append("host syncs detected")
    if (
        summary.labeled_positions_per_second is not None
        and summary.labeled_positions_per_second < 2_500
    ):
        gaps.append("labeled positions per second below 2500")
    return gaps


def profile_from_benchmark(path: Path) -> ProfileSummary:
    from lattice.benchmarking import load_report

    report = load_report(path)
    merged: dict[str, float | int] = {}
    for section in report.sections:
        for key, value in section.metrics.items():
            if isinstance(value, int | float):
                merged[key] = value
        if section.metrics.get("alpha_training_uses_reference_movegen") is True:
            merged["alpha_training_uses_reference_movegen"] = 1

    return ProfileSummary(
        gpu_sm_busy=_float_or_none(merged.get("gpu_sm_busy")),
        tensor_core_utilization=_float_or_none(merged.get("tensor_core_utilization")),
        cpu_cores=_float_or_none(merged.get("cpu_cores")),
        all_reduce_share=_float_or_none(
            merged.get("mean_all_reduce_share", merged.get("all_reduce_share"))
        ),
        labeled_positions_per_second=_float_or_none(
            merged.get("labeled_positions_per_second")
        ),
        learner_steps_per_second=_float_or_none(
            merged.get("steps_per_second", merged.get("min_rank_steps_per_second"))
        ),
        host_syncs=_int_or_none(merged.get("host_syncs")),
    )


def acceptance_gaps_from_benchmark(path: Path) -> list[str]:
    from lattice.benchmarking import load_report

    report = load_report(path)
    gaps = acceptance_gaps(profile_from_benchmark(path))
    gaps.extend(report.gaps)
    for section in report.sections:
        if section.metrics.get("alpha_training_uses_reference_movegen") is True:
            gaps.append("alpha profile uses CPU reference movegen")
    for section in report.sections:
        if section.skip_reason and section.name not in {"compiled_learner"}:
            gaps.append(f"{section.name} skipped: {section.skip_reason}")
    return sorted(set(gaps))


def _float_or_none(value: float | int | None) -> float | None:
    return None if value is None else float(value)


def _int_or_none(value: float | int | None) -> int | None:
    return None if value is None else int(value)
