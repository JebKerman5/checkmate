from lattice.chess.tables import AttackTables
from lattice.profiling import ProfileSummary, acceptance_gaps, summarize_profile
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
