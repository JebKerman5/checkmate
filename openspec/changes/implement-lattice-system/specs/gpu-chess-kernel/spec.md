## ADDED Requirements

### Requirement: Batched GPU board state
The system SHALL represent active chess games as struct-of-arrays GPU tensors with bitboards, occupancy, metadata, repetition history, and terminal status.

#### Scenario: Initial batch allocation
- **WHEN** a rank creates a game batch
- **THEN** all board-state tensors are allocated on that rank's CUDA device with batch dimension first.

### Requirement: Legal move generation
The GPU chess kernel SHALL produce legal move codes and legal masks for each active game without copying board state to the CPU.

#### Scenario: Movegen batch step
- **WHEN** the generator requests legal moves for a batch
- **THEN** the kernel returns padded move codes, legal masks, and overflow flags on the same CUDA device.

### Requirement: CPU differential correctness
The project SHALL compare GPU move generation and make results against a CPU reference before the GPU loop can pass acceptance.

#### Scenario: Reference disagreement
- **WHEN** a generated test position disagrees with the CPU reference
- **THEN** the correctness test fails and reports the FEN, move list mismatch, and failing kernel stage.

### Requirement: Terminal detection
The GPU chess kernel SHALL detect checkmate, stalemate, fifty-move draws, threefold repetition, and insufficient material in tensor form.

#### Scenario: Terminal child label
- **WHEN** sparse labeling creates a child position that is terminal
- **THEN** the labeler uses the exact terminal value instead of an EMA network value.

### Requirement: Capturable step path
The hot board-step path SHALL avoid host synchronization and SHALL be compatible with CUDA graph capture after static buffers are allocated.

#### Scenario: Profiler check
- **WHEN** the profiler inspects a captured generation step
- **THEN** it shows no CPU board-state reads or synchronization inside the captured path.
