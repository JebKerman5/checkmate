## ADDED Requirements

### Requirement: Local-safe benchmark command
The system SHALL provide a benchmark command that is safe to run on a development PC by default and does not require CUDA, PyTorch GPU extras, background services, or long-running workloads.

#### Scenario: Local benchmark without CUDA
- **WHEN** the user runs the local benchmark profile on a machine without CUDA or without torch installed
- **THEN** the command emits a valid benchmark report with GPU/DDP sections marked skipped and exits successfully unless a local contract check fails

#### Scenario: Local benchmark duration bound
- **WHEN** the user runs the local benchmark profile with default arguments
- **THEN** the command finishes within a short bounded duration and does not start persistent processes

### Requirement: Benchmark JSON report schema
Every benchmark command SHALL be able to write a machine-readable JSON report containing schema version, git commit, command profile, environment, metrics, thresholds, gaps, and skip reasons.

#### Scenario: JSON report is requested
- **WHEN** the user passes a JSON output path to a benchmark command
- **THEN** the command writes a report that validates against the project benchmark schema

#### Scenario: Benchmark section is skipped
- **WHEN** a benchmark section cannot run because CUDA, DDP, or required hardware is unavailable
- **THEN** the JSON report includes the skipped section with an explicit reason instead of omitting it

### Requirement: GPU microbenchmarks
The system SHALL provide VM-only CUDA microbenchmarks for model forward speed, learner step speed, replay write throughput, replay sample throughput, movegen/make throughput, host synchronization count, and CPU usage.

#### Scenario: CUDA microbenchmark on VM
- **WHEN** the user runs the CUDA benchmark profile on a CUDA VM
- **THEN** the report includes measured metrics for every available single-GPU benchmark section and pass/fail gaps for configured thresholds

#### Scenario: Timed CUDA region
- **WHEN** a CUDA benchmark measures a GPU operation
- **THEN** synchronization occurs outside the timed steady-state region except for explicit timing boundaries

### Requirement: DDP benchmark
The system SHALL provide a two-rank DDP benchmark command that measures per-rank learner throughput, global batch throughput, all-reduce time share, rank skew, and synchronization health.

#### Scenario: Two-rank DDP benchmark
- **WHEN** the user runs the DDP benchmark through `torchrun --nproc_per_node=2`
- **THEN** both ranks emit compatible metric records and rank 0 writes an aggregate benchmark report

#### Scenario: Incorrect world size
- **WHEN** the DDP benchmark is launched with a world size other than two
- **THEN** the command fails fast with a clear topology error and does not start a training loop

### Requirement: Performance regression tests
The test suite SHALL include contract tests for benchmark report validation, threshold evaluation, skipped hardware sections, and deterministic smoke metrics.

#### Scenario: Benchmark schema test
- **WHEN** unit tests run on the local development environment
- **THEN** benchmark schema and threshold tests pass without requiring CUDA

#### Scenario: Hardware performance tests
- **WHEN** CUDA or DDP benchmark tests require unavailable hardware
- **THEN** pytest marks those tests skipped with an explicit reason rather than failing
