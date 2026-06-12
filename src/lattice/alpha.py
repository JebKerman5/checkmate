from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lattice.benchmarking import collect_environment, load_report
from lattice.config import config_preset


@dataclass(frozen=True)
class AlphaManifest:
    run_id: str
    git_commit: str | None
    config_preset: str
    vm_shape: str
    benchmark_reports: list[str] = field(default_factory=list)
    profile_summary: str | None = None
    checkpoints: list[str] = field(default_factory=list)
    loss_curves: list[str] = field(default_factory=list)
    throughput_summaries: list[str] = field(default_factory=list)
    known_gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "git_commit": self.git_commit,
            "config_preset": self.config_preset,
            "vm_shape": self.vm_shape,
            "benchmark_reports": list(self.benchmark_reports),
            "profile_summary": self.profile_summary,
            "checkpoints": list(self.checkpoints),
            "loss_curves": list(self.loss_curves),
            "throughput_summaries": list(self.throughput_summaries),
            "known_gaps": list(self.known_gaps),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlphaManifest:
        return cls(
            run_id=str(data["run_id"]),
            git_commit=data.get("git_commit"),
            config_preset=str(data["config_preset"]),
            vm_shape=str(data["vm_shape"]),
            benchmark_reports=[str(item) for item in data.get("benchmark_reports", [])],
            profile_summary=data.get("profile_summary"),
            checkpoints=[str(item) for item in data.get("checkpoints", [])],
            loss_curves=[str(item) for item in data.get("loss_curves", [])],
            throughput_summaries=[
                str(item) for item in data.get("throughput_summaries", [])
            ],
            known_gaps=[str(item) for item in data.get("known_gaps", [])],
        )


def write_manifest(run_dir: Path, manifest: AlphaManifest) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_manifest(path: Path) -> AlphaManifest:
    return AlphaManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def validate_manifest_paths(run_dir: Path, manifest: AlphaManifest) -> list[str]:
    gaps: list[str] = []
    for collection_name, paths in (
        ("benchmark_reports", manifest.benchmark_reports),
        ("checkpoints", manifest.checkpoints),
        ("loss_curves", manifest.loss_curves),
        ("throughput_summaries", manifest.throughput_summaries),
    ):
        for item in paths:
            if not (run_dir / item).exists():
                gaps.append(f"{collection_name} missing {item}")
    if manifest.profile_summary and not (run_dir / manifest.profile_summary).exists():
        gaps.append(f"profile_summary missing {manifest.profile_summary}")
    for report_path in manifest.benchmark_reports:
        try:
            load_report(run_dir / report_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            gaps.append(f"benchmark report invalid {report_path}: {exc}")
    return gaps


def evaluate_alpha_readiness(
    repo_root: Path,
    run_dir: Path,
    require_vm_artifacts: bool = True,
    require_openspec_complete: bool = True,
) -> dict[str, Any]:
    gaps: list[str] = []
    cfg = config_preset("checkmate-alpha1")
    if cfg.runtime.required_world_size != 2:
        gaps.append("checkmate-alpha1 preset must require DDP world size 2")
    if not cfg.runtime.require_cuda:
        gaps.append("checkmate-alpha1 preset must require CUDA")
    if not cfg.runtime.require_bf16:
        gaps.append("checkmate-alpha1 preset must require BF16")
    if not cfg.runtime.require_peer_access:
        gaps.append("checkmate-alpha1 preset must require peer access")

    for relative in ("README.md", "docs/VM_RUNBOOK.md", "docs/checkmate-alpha1.md"):
        if not (repo_root / relative).exists():
            gaps.append(f"missing documentation {relative}")

    manifest_path = run_dir / "manifest.json"
    if require_vm_artifacts:
        if not manifest_path.exists():
            gaps.append(f"missing alpha manifest {manifest_path}")
        else:
            manifest = load_manifest(manifest_path)
            gaps.extend(validate_manifest_paths(run_dir, manifest))

    if require_openspec_complete:
        gaps.extend(_openspec_task_gaps(repo_root, "implement-lattice-system"))
        gaps.extend(_openspec_task_gaps(repo_root, "prepare-checkmate-alpha1-optimizations"))

    return {
        "ready": not gaps,
        "gaps": sorted(set(gaps)),
        "environment": collect_environment("checkmate-alpha1", repo_root, probe_torch=False),
    }


def _openspec_task_gaps(repo_root: Path, change: str) -> list[str]:
    tasks_path = repo_root / "openspec" / "changes" / change / "tasks.md"
    if not tasks_path.exists():
        return [f"missing OpenSpec tasks for {change}"]
    unchecked = [
        line.strip()[6:]
        for line in tasks_path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("- [ ]")
    ]
    if unchecked:
        return [f"{change} has {len(unchecked)} incomplete tasks"]
    return []
