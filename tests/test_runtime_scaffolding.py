from lattice.benchmarking import BenchmarkThreshold, make_section, write_report
from lattice.chess.tables import AttackTables
from lattice.profiling import (
    PROFILE_EVENT_NAMES,
    HostSyncCounter,
    ProfileRecorder,
    ProfileSummary,
    acceptance_gaps,
    acceptance_gaps_from_benchmark,
    profile_from_benchmark,
    summarize_profile,
)
from lattice.tripwire import TripwireState, evaluate_tripwires


def test_attack_tables_have_64_entries() -> None:
    tables = AttackTables()

    assert len(tables.knight) == 64
    assert len(tables.king) == 64
    assert len(tables.bishop_rays) == 64
    assert len(tables.rook_rays) == 64
    assert len(tables.queen_rays) == 64


def test_tripwire_reports_failures() -> None:
    failures = evaluate_tripwires(
        TripwireState(value_drift=0.2, loss_ratio=1.0, replay_fill=0.01, anchor_delta=0.0)
    )

    assert "value drift" in failures
    assert "replay starvation" in failures


def test_profile_acceptance_gaps() -> None:
    summary = ProfileSummary(
        gpu_sm_busy=0.5,
        cpu_cores=2.0,
        host_syncs=1,
        labeled_positions_per_second=100.0,
    )

    gaps = acceptance_gaps(summary)

    assert "GPU SM busy below 90%" in gaps
    assert "CPU usage above one core" in gaps
    assert summarize_profile(summary)["host_syncs"] == 1


def test_profile_recorder_and_host_sync_counter() -> None:
    recorder = ProfileRecorder()
    syncs = HostSyncCounter()

    with recorder.timed("learner_forward_backward"):
        syncs.mark("manual-test")

    assert recorder.events[0].name == "learner_forward_backward"
    assert syncs.count == 1
    assert syncs.sites["manual-test"] == 1
    assert "replay_sample" in PROFILE_EVENT_NAMES


def test_profile_summary_loads_benchmark_report(tmp_path) -> None:
    from lattice.benchmarking import BenchmarkReport

    report = BenchmarkReport(
        profile="test",
        command="test",
        environment={},
        sections=[
            make_section(
                "learner",
                metrics={"steps_per_second": 1.5, "host_syncs": 1},
                thresholds=[BenchmarkThreshold("host_syncs", "eq", 0)],
            )
        ],
    )
    path = tmp_path / "bench.json"
    write_report(report, path)

    summary = profile_from_benchmark(path)
    gaps = acceptance_gaps_from_benchmark(path)

    assert summary.learner_steps_per_second == 1.5
    assert "learner: host_syncs 1 != 0" in gaps
