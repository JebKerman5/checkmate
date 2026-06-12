## ADDED Requirements

### Requirement: Per-rank GPU self-play
Each DDP rank SHALL generate self-play positions on its own GPU using its local active game batch and local labeler snapshots.

#### Scenario: Independent rank generation
- **WHEN** both DDP ranks enter steady-state training
- **THEN** each rank advances its own game batch and writes records to its own replay shard.

### Requirement: Q-guided move selection
The generator SHALL choose moves from legal moves using model Q scores, temperature, and a small uniform exploration mix.

#### Scenario: Legal action sample
- **WHEN** the generator samples an action
- **THEN** the selected action belongs to the legal mask for that position.

### Requirement: Sparse negamax labeling
The labeler SHALL build targets by evaluating the top-k Q moves plus at least one random legal move for each labeled position.

#### Scenario: Child target creation
- **WHEN** a non-terminal position is labeled
- **THEN** the record stores Q targets for expanded moves and a V target from the best negated child value.

### Requirement: GPU opening seeds
The system SHALL create opening seed positions from GPU random playouts and assign rank-local seed slices for game resets.

#### Scenario: Game reset from seed
- **WHEN** an active game ends
- **THEN** the rank resets it from a GPU-resident opening seed without loading a PGN or CPU dataset.

### Requirement: Outcome backfill
The generator SHALL retain enough per-game record metadata to backfill outcome-anchored WDL targets when a game terminates.

#### Scenario: Finished game
- **WHEN** a game reaches a terminal result
- **THEN** a GPU backfill path updates the open records for that game with the final outcome component.
