import json
from pathlib import Path

from lattice.benchmarking import (
    BenchmarkThreshold,
    collect_environment,
    evaluate_thresholds,
    load_report,
    make_section,
    run_local_safe_benchmark,
    validate_report_dict,
    write_report,
)


def test_threshold_evaluation_is_deterministic() -> None:
    gaps = evaluate_thresholds(
        {"steps_per_second": 1.0, "host_syncs": 1},
        [
            BenchmarkThreshold("steps_per_second", "ge", 2.0),
            BenchmarkThreshold("host_syncs", "eq", 0),
        ],
    )

    assert gaps == ["steps_per_second 1.0 < 2.0", "host_syncs 1 != 0"]


def test_skipped_section_serializes() -> None:
    section = make_section("cuda", skip_reason="CUDA unavailable")

    assert section.status == "skip"
    assert section.to_dict()["skip_reason"] == "CUDA unavailable"


def test_local_safe_benchmark_does_not_probe_torch(tmp_path: Path) -> None:
    report = run_local_safe_benchmark(tmp_path, seconds=0.001)

    assert report.profile == "local"
    assert report.environment["torch_probed"] is False
    assert report.sections[0].metrics["cuda_touched"] is False
    assert report.sections[1].status == "skip"


def test_report_validation_and_round_trip(tmp_path: Path) -> None:
    report = run_local_safe_benchmark(tmp_path, seconds=0.001)
    path = tmp_path / "bench.json"

    write_report(report, path)
    loaded = load_report(path)
    errors = validate_report_dict(json.loads(path.read_text(encoding="utf-8")))

    assert loaded.profile == "local"
    assert errors == []


def test_collect_environment_without_torch_probe(tmp_path: Path) -> None:
    env = collect_environment("local-contract", tmp_path, probe_torch=False)

    assert env["preset"] == "local-contract"
    assert env["torch_probed"] is False
