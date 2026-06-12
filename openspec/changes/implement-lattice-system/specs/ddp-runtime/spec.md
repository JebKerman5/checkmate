## ADDED Requirements

### Requirement: Dual-GPU DDP launch
The system SHALL run as a single-node PyTorch DDP job with one process per CUDA GPU.

#### Scenario: Two-rank startup
- **WHEN** the operator starts training with `torchrun --nproc_per_node=2`
- **THEN** the system binds rank 0 to GPU 0, binds rank 1 to GPU 1, initializes NCCL, and logs both rank/device assignments.

### Requirement: Runtime topology validation
The system SHALL validate CUDA availability, NCCL initialization, peer access, BF16 support, and visible device count before training allocates large buffers.

#### Scenario: Invalid VM topology
- **WHEN** fewer than two CUDA GPUs are visible or NCCL initialization fails
- **THEN** the system exits before training and reports the failed runtime check.

### Requirement: Rank-owned static buffers
Each rank SHALL allocate its own static GPU buffers for active games, child positions, replay records, label targets, and learner batches.

#### Scenario: Buffer sizing report
- **WHEN** the DDP runtime starts
- **THEN** each rank reports planned VRAM allocation by subsystem before entering the training loop.

### Requirement: Rank-zero host responsibilities
Only rank 0 SHALL write checkpoints, persistent logs, and evaluation summaries unless a feature explicitly requires per-rank output.

#### Scenario: Checkpoint write
- **WHEN** the trainer reaches a checkpoint interval
- **THEN** rank 0 writes the checkpoint and other ranks skip host file writes.
