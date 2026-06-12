## 1. DDP Runtime Skeleton

- [x] 1.1 Update project docs and config defaults from single-GPU wording to dual-GPU DDP wording.
- [x] 1.2 Add a `torchrun` entrypoint that initializes one PyTorch DDP rank per GPU.
- [x] 1.3 Bind each rank to its CUDA device and log rank, local rank, world size, device name, and BF16 support.
- [x] 1.4 Add startup checks for two visible CUDA GPUs, NCCL initialization, peer access, and available VRAM.
- [x] 1.5 Add per-rank static buffer sizing config for active games, child positions, replay records, and learner batch.
- [x] 1.6 Add a smoke test command that launches two ranks and exits after runtime validation.

## 2. LATTICE Model And Synthetic DDP Learner

- [x] 2.1 Implement LATTICE-base config, board feature shape contracts, and model output contracts.
- [x] 2.2 Implement the axis-mixer block, per-square MLP, Q head, WDL head, value head, and moves-left head.
- [x] 2.3 Add unit tests for forward shapes, legal-move Q gathering, and loss input masks.
- [x] 2.4 Add synthetic GPU replay tensors so the DDP learner can run before chess generation exists.
- [x] 2.5 Implement DDP learner step with BF16 autocast, gradient clipping, AdamW, and NCCL all-reduce.
- [x] 2.6 Verify both ranks keep identical online model weights after optimizer steps.

## 3. GPU Chess Kernel Contract

- [x] 3.1 Define the batched board-state API for bitboards, occupancy, metadata, repetition history, and status.
- [x] 3.2 Add CPU reference adapters using `python-chess` for legal moves, make, and terminal checks.
- [x] 3.3 Implement a small-batch GPU movegen prototype for pawns, knights, kings, and occupancy updates.
- [x] 3.4 Add differential tests that report FEN, GPU moves, CPU moves, and failing stage on mismatch.
- [x] 3.5 Add slider attack tables and legal king-safety filtering.
- [x] 3.6 Add terminal detection for mate, stalemate, fifty-move, repetition, and insufficient material.
- [x] 3.7 Add a captured or capturable board-step benchmark that reports board-steps per second and host syncs.

## 4. VRAM Replay And Labeling

- [x] 4.1 Implement rank-local replay shard tensors for boards, sparse Q labels, V/WDL labels, moves-left labels, masks, and priorities.
- [x] 4.2 Implement GPU record writes from generated positions into circular replay slots.
- [x] 4.3 Implement GPU batch sampling from replay with priorities and importance weights.
- [x] 4.4 Implement local EMA labeler snapshots from synchronized online model weights.
- [x] 4.5 Implement sparse negamax labeling with top-k Q moves plus one random legal move.
- [x] 4.6 Implement terminal child overrides so exact game results replace network child values.
- [x] 4.7 Implement episode metadata and GPU outcome backfill for finished games.

## 5. Integrated Self-Play Training Loop

- [x] 5.1 Run one-rank generation into replay and verify replay records are trainable.
- [ ] 5.2 Run two-rank DDP learner on generated replay with generation disabled.
- [ ] 5.3 Integrate per-rank generation, labeling, replay sampling, and DDP learner in one process group.
- [x] 5.4 Add replay freshness accounting so record reuse and overwrite rate are reported.
- [x] 5.5 Add divergence tripwires for value drift, loss spikes, anchor regression, and replay starvation.
- [x] 5.6 Add checkpoint save and resume for model, optimizer, EMA labelers, scheduler, and counters.
- [ ] 5.7 Run a 10-minute dual-GPU smoke run and record labeled positions per second, learner steps per second, and loss curves.

## 6. Evaluation And Profiling

- [x] 6.1 Add searchless inference over legal moves from Q scores.
- [x] 6.2 Add checkpoint comparison hooks for fixed openings, tactical positions, and anchor positions.
- [x] 6.3 Add profiler reporting for SM busy, tensor-core utilization, CPU core usage, all-reduce time share, and host syncs.
- [ ] 6.4 Profile the integrated smoke run and identify the top GPU and CPU bottlenecks.
- [ ] 6.5 Move the highest-impact chess hot path from prototype code into Triton or CUDA.
- [ ] 6.6 Run a one-hour dual-GPU training run and save the profile summary with acceptance gaps.
