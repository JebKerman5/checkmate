## ADDED Requirements

### Requirement: LATTICE model outputs
The model SHALL produce Q scores for legal move selection, WDL logits, scalar value, and moves-left prediction from the same board encoding.

#### Scenario: Forward output contract
- **WHEN** the learner runs a forward pass
- **THEN** the model returns tensors compatible with sparse Q loss, WDL cross-entropy, value loss, and moves-left loss.

### Requirement: Base configuration first
The project SHALL implement and gate LATTICE-base before adding LATTICE-big or lattice-beam strength mode.

#### Scenario: Initial training config
- **WHEN** the operator starts the first integrated training run
- **THEN** the default model config uses the base dimensions and disables big-model-only features.

### Requirement: Checkpoint and resume
The trainer SHALL save and resume online model weights, optimizer state, EMA labelers, scheduler state, rank configuration, and counters needed for deterministic continuation.

#### Scenario: Resume from checkpoint
- **WHEN** the operator resumes from a checkpoint
- **THEN** the DDP job restores synchronized model state and continues with consistent global step counters.

### Requirement: GPU-heavy acceptance metrics
The system SHALL report dual-GPU utilization, per-rank generation throughput, learner steps per second, all-reduce cost, replay reuse, and CPU usage.

#### Scenario: Steady-state profile
- **WHEN** a smoke run reaches steady state
- **THEN** the report includes SM busy for both GPUs, CPU core usage, labeled positions per second, learner steps per second, and all-reduce time share.

### Requirement: Evaluation harness
The project SHALL provide fixed evaluation modes for checkpoint comparison, tactical positions, anchor openings, and searchless UCI inference.

#### Scenario: Checkpoint gate
- **WHEN** a candidate checkpoint completes evaluation
- **THEN** rank 0 records the evaluation result and marks whether the checkpoint is eligible for future rollback.
