from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileSummary:
    gpu_sm_busy: float | None = None
    tensor_core_utilization: float | None = None
    cpu_cores: float | None = None
    all_reduce_share: float | None = None
    labeled_positions_per_second: float | None = None
    learner_steps_per_second: float | None = None
    host_syncs: int | None = None


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
