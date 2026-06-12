## ADDED Requirements

### Requirement: Steady-state host synchronization budget
The integrated training loop SHALL keep host synchronization out of the steady-state generate-label-train path except for explicit metrics, checkpoint, and benchmark timing boundaries.

#### Scenario: Host sync instrumentation
- **WHEN** the integrated smoke run completes
- **THEN** the profile report includes a host synchronization count and identifies any sync sites that occurred inside the steady-state path

#### Scenario: Host sync gate failure
- **WHEN** host synchronizations are detected inside the steady-state path
- **THEN** the benchmark report marks the relevant alpha gate as failed

### Requirement: Optimized chess hot path
The project SHALL replace the highest-impact prototype chess hot path with a GPU-resident Triton, CUDA, or vectorized tensor implementation before `checkmate-alpha1`.

#### Scenario: Hot path selection
- **WHEN** profiling identifies the top chess bottleneck
- **THEN** the implementation plan records which path is being optimized and why it has the highest expected impact

#### Scenario: Optimized path correctness
- **WHEN** the optimized chess path is enabled
- **THEN** differential tests compare it against the CPU reference on deterministic FEN fixtures and fail on the first mismatch with board and move diagnostics

### Requirement: No CPU reference in alpha training path
The alpha training path SHALL NOT depend on `python-chess` or CPU reference move generation for production generation, labeling, move application, or terminal detection.

#### Scenario: Alpha profile uses optimized movegen
- **WHEN** the user runs an alpha VM profile
- **THEN** the report states that the optimized GPU chess path is enabled or fails readiness with a clear reason

#### Scenario: Reference path remains test-only
- **WHEN** tests run differential movegen checks
- **THEN** the CPU reference path is used only as an oracle and not as the measured alpha training implementation

### Requirement: Learner optimization controls
The learner SHALL expose configurable controls for eager mode, `torch.compile`, static buffer reuse, BF16 autocast, and CUDA graph capture where shapes are stable.

#### Scenario: Compiled learner benchmark
- **WHEN** the CUDA benchmark is run with compiled learner mode enabled
- **THEN** the report compares compiled throughput against eager throughput or records a compile skip reason

#### Scenario: CUDA graph capture failure
- **WHEN** CUDA graph capture is requested but the loop contains an unsupported operation
- **THEN** the command falls back or fails with a clear reason instead of silently producing invalid metrics

### Requirement: Replay optimization
Replay writes and sampling SHALL use preallocated GPU tensors and avoid Python per-record loops, host transfers, and avoidable allocations in the steady-state path.

#### Scenario: Replay throughput benchmark
- **WHEN** the replay benchmark runs on CUDA
- **THEN** it reports write records/s, sample records/s, allocation behavior, and priority sampling cost

#### Scenario: Replay bottleneck detection
- **WHEN** priority sampling consumes a configured share of learner step time
- **THEN** the benchmark report flags replay sampling as an optimization gap

### Requirement: Bottleneck-driven optimization record
Each alpha optimization SHALL be tied to a before/after benchmark or profile artifact.

#### Scenario: Optimization is completed
- **WHEN** an optimization task is marked done
- **THEN** there is a benchmark artifact showing the previous metric, new metric, hardware context, and remaining gap
