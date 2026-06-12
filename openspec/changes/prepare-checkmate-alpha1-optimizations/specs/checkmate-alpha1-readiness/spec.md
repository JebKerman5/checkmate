## ADDED Requirements

### Requirement: Alpha readiness command
The system SHALL provide a `checkmate-alpha1` readiness command that validates required tests, benchmark artifacts, DDP topology assumptions, config presets, and documentation before a paid training run.

#### Scenario: Readiness passes
- **WHEN** all required alpha tests and VM benchmark artifacts satisfy their thresholds
- **THEN** the readiness command exits successfully and writes an alpha readiness report

#### Scenario: Readiness fails
- **WHEN** a required artifact, metric, config, or documented command is missing
- **THEN** the readiness command exits nonzero and lists the exact gaps to fix

### Requirement: Alpha artifact manifest
The alpha run SHALL produce an artifact manifest that names the git commit, config preset, VM shape, benchmark reports, profile summary, checkpoints, loss curves, throughput summaries, and known acceptance gaps.

#### Scenario: Alpha manifest is written
- **WHEN** an alpha smoke or candidate run finishes
- **THEN** the run directory contains a manifest that references every required alpha artifact by relative path

#### Scenario: Artifact is missing
- **WHEN** a manifest references a missing artifact
- **THEN** the readiness command reports the manifest as invalid

### Requirement: VM-only training runbook
The documentation SHALL clearly separate local-safe commands from VM-only CUDA/DDP commands and provide the exact command sequence for low-cost smoke testing and dual-GPU alpha training.

#### Scenario: User reads VM runbook
- **WHEN** the user opens the VM runbook
- **THEN** it includes setup, DDP smoke, CUDA microbenchmarks, two-rank benchmark, ten-minute smoke, one-hour alpha candidate, artifact download, and teardown steps

#### Scenario: Local README safety
- **WHEN** the user reads the local README
- **THEN** it states that GPU-heavy benchmark and training commands are VM-only and must not be run accidentally on the gaming PC

### Requirement: Alpha config presets
The project SHALL define explicit config presets for local contract tests, single-GPU smoke tests, dual-GPU DDP smoke, and `checkmate-alpha1`.

#### Scenario: Local preset
- **WHEN** the local preset is selected
- **THEN** it uses short CPU-safe checks and does not require torch GPU extras

#### Scenario: Alpha1 preset
- **WHEN** the alpha1 preset is selected
- **THEN** it requires DDP world size two, CUDA, BF16, peer access, GPU replay, and explicit VM execution

### Requirement: No-local-bloat safeguard
The implementation SHALL avoid persistent local services, watchers, auto-started processes, global tool installs, and accidental CUDA workloads on the development PC.

#### Scenario: Command exits
- **WHEN** a local-safe command finishes
- **THEN** no project-owned background process remains running

#### Scenario: CUDA command on local machine
- **WHEN** a GPU-heavy command is invoked without an explicit CUDA/DDP profile
- **THEN** the command refuses to run or prints the skipped hardware sections instead of using the GPU

### Requirement: Alpha tag readiness
The project SHALL only consider `checkmate-alpha1` ready to tag after the base implementation change and this optimization change satisfy their OpenSpec task gates.

#### Scenario: OpenSpec gates incomplete
- **WHEN** either required OpenSpec change has incomplete tasks
- **THEN** the readiness command reports that alpha1 cannot be tagged yet
