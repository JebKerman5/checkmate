from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TripwireConfig:
    max_value_drift: float = 0.15
    max_loss_spike: float = 5.0
    min_replay_fill: float = 0.1
    min_anchor_delta: float = -50.0


@dataclass(frozen=True)
class TripwireState:
    value_drift: float
    loss_ratio: float
    replay_fill: float
    anchor_delta: float


def evaluate_tripwires(state: TripwireState, cfg: TripwireConfig | None = None) -> list[str]:
    limits = cfg or TripwireConfig()
    failures: list[str] = []
    if abs(state.value_drift) > limits.max_value_drift:
        failures.append("value drift")
    if state.loss_ratio > limits.max_loss_spike:
        failures.append("loss spike")
    if state.replay_fill < limits.min_replay_fill:
        failures.append("replay starvation")
    if state.anchor_delta < limits.min_anchor_delta:
        failures.append("anchor regression")
    return failures
