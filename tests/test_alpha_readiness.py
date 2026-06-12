import json
from pathlib import Path

from lattice.alpha import (
    AlphaManifest,
    evaluate_alpha_readiness,
    validate_manifest_paths,
    write_manifest,
)
from lattice.benchmarking import run_local_safe_benchmark, write_report


def test_manifest_validation_reports_missing_files(tmp_path: Path) -> None:
    manifest = AlphaManifest(
        run_id="test",
        git_commit="abc123",
        config_preset="checkmate-alpha1",
        vm_shape="test-vm",
        benchmark_reports=["missing.json"],
    )

    gaps = validate_manifest_paths(tmp_path, manifest)

    assert "benchmark_reports missing missing.json" in gaps


def test_manifest_validation_accepts_existing_benchmark(tmp_path: Path) -> None:
    report_path = tmp_path / "local.json"
    write_report(run_local_safe_benchmark(tmp_path, seconds=0.001), report_path)
    manifest = AlphaManifest(
        run_id="test",
        git_commit="abc123",
        config_preset="checkmate-alpha1",
        vm_shape="test-vm",
        benchmark_reports=["local.json"],
    )

    gaps = validate_manifest_paths(tmp_path, manifest)

    assert gaps == []


def test_readiness_reports_missing_docs_and_manifest(tmp_path: Path) -> None:
    result = evaluate_alpha_readiness(
        tmp_path,
        tmp_path / "runs",
        require_vm_artifacts=True,
        require_openspec_complete=False,
    )

    assert result["ready"] is False
    assert any("missing documentation" in gap for gap in result["gaps"])
    assert any("missing alpha manifest" in gap for gap in result["gaps"])


def test_write_manifest_round_trip(tmp_path: Path) -> None:
    manifest = AlphaManifest(
        run_id="test",
        git_commit="abc123",
        config_preset="checkmate-alpha1",
        vm_shape="test-vm",
        known_gaps=["not ready"],
    )

    path = write_manifest(tmp_path, manifest)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["known_gaps"] == ["not ready"]
