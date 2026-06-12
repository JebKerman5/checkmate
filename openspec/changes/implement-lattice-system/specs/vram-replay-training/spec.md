## ADDED Requirements

### Requirement: Rank-local VRAM replay shard
Each DDP rank SHALL store generated records in a local GPU replay shard sized by the current device memory budget.

#### Scenario: Record write
- **WHEN** the generator produces a labeled position
- **THEN** it writes board features, sparse Q targets, V/WDL targets, moves-left target, and priority into the local replay shard.

### Requirement: GPU prioritized sampling
The learner SHALL sample replay batches on the GPU using priorities without moving replay records through a CPU dataloader.

#### Scenario: Learner batch sample
- **WHEN** the learner needs a batch
- **THEN** it receives device tensors for model inputs, labels, masks, and importance weights.

### Requirement: DDP learner step
The learner SHALL train the online model under PyTorch DDP and synchronize gradients through NCCL.

#### Scenario: Optimizer step
- **WHEN** each rank finishes backward for its local batch
- **THEN** DDP all-reduces gradients and every rank applies the same optimizer step.

### Requirement: EMA labeler snapshots
The trainer SHALL maintain EMA labeler snapshots for target creation without requiring cross-rank host coordination.

#### Scenario: EMA refresh
- **WHEN** the configured refresh interval elapses
- **THEN** each rank updates its local EMA labeler snapshots from the synchronized online model weights.

### Requirement: Divergence tripwire
The trainer SHALL monitor value drift, anchor evaluation, replay freshness, and loss spikes during training.

#### Scenario: Tripwire breach
- **WHEN** a configured divergence tripwire fires
- **THEN** rank 0 restores the last gated checkpoint decision and all ranks resume from synchronized model state.
