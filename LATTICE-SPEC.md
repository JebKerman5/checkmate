# LATTICE — Architectural Specification v1.0

A chess neural network system designed around one hard constraint: **the GPU must be the bottleneck.** Generation, labeling, and learning all live in VRAM; the CPU's only jobs are logging and UCI parsing.

This spec is fully standalone. It shares no components, formats, or assumptions with any prior spec. Written for an implementation agent: shapes, budgets, and acceptance gates throughout.

---

## 0. The bottleneck audit (why this design exists)

Where wall-clock actually goes in existing training pipelines:

| pipeline | label source | where time goes |
|---|---|---|
| AlphaZero / Lc0 style | MCTS self-play, ~800 net calls **per move** | CPU tree logic + GPU idling between tiny inference batches; GPU utilization in the generation phase is typically a few % |
| NNUE style | huge offline corpora labeled by CPU engine search | weeks of pure CPU farm time before the GPU even starts |
| Searchless (distillation) | a strong CPU engine labels every position | same CPU farm problem, relocated |

Three structural causes: (1) chess simulation runs on CPU, (2) the label generator is a *search*, which is sequential and pointer-chasing, (3) data travels CPU↔disk↔GPU. LATTICE removes all three:

1. **Chess itself runs on the GPU** — boards, move generation, terminal detection, all as batched tensor ops over ~65k games in lockstep (§2).
2. **No search anywhere in training.** Labels come from *negamax self-consistency*: the value function is trained to be a fixed point of the one-ply negamax operator, with checkmates/stalemates/draw rules (exactly detectable in-tensor) as ground-truth anchors that propagate backward through the state space. Cost per labeled position: ~5 batched forwards instead of ~800 sequential ones (§4).
3. **The replay buffer lives in VRAM.** No disk, no dataloader, no host round-trips. One process, one GPU, three CUDA streams (§4.1).

**The headline acceptance gate, taken directly from the requirement:** during steady-state training, GPU utilization ≥ 90% (SM busy), CPU usage ≤ 1 core. If the implementation misses this gate, the implementation is wrong, not the gate.

Precedent statement (one line, then M0 verifies): GPU-vectorized board-game environments exist (Pgx/JAX lineage), TD-style consistency training has ancestors (TreeStrap, DQN targets), and MLP-Mixer exists in vision. The chess-axis mixer architecture (§3), Q-guided sparse negamax labeling at megabatch scale as the *primary* label source (§4.3), and the fully VRAM-resident generate→label→train loop with the 90% gate are, to my knowledge, an unprecedented system. Milestone M0 is a literature check to confirm before building.

---

## 1. System overview

```
ONE PROCESS, ONE GPU
┌──────────────────────────────────────────────────────────────┐
│  stream G (generation): 65,536 games stepped in lockstep      │
│      └─ writes packed (board, targets) into the VRAM ring     │
│  stream L (learner): samples 4096-batches from the ring,      │
│      └─ trains online net; EMA target nets updated every k    │
│  stream M (metrics): tiny; copies scalars to host             │
└──────────────────────────────────────────────────────────────┘
CPU: UCI bridge (inference only), TensorBoard logging, checkpoints.
```

Framework: PyTorch 2.x with CUDA graphs + `torch.compile` (or JAX — agent's choice; the spec is framework-neutral, budgets assume an RTX 4070-class card, 12 GB).

---

## 2. Subsystem S0 — the GPU chess kernel

### 2.1 State tensors (struct-of-arrays, batch dim B first)

| tensor | shape/dtype | content |
|---|---|---|
| `bb` | [B, 12] int64 | piece bitboards (P,N,B,R,Q,K × white,black) |
| `occ` | [B, 2] int64 | per-color occupancy (derived, cached) |
| `meta` | [B] int32 | stm(1) castling(4) ep-file(4) halfmove(7) packed |
| `zhist` | [B, 104] int64 | Zobrist ring buffer for repetition detection |
| `status` | [B] int8 | ongoing / white-win / black-win / draw(reason) |

### 2.2 Batched move generation

- Knights/kings/pawns: bitwise shifts and masks — trivially vectorized.
- Sliders: **magic bitboards as GPU lookup tensors.** The standard magic tables (~800 KB) upload once; attack sets become `gather(table, (occ & mask) * magic >> shift)` — pure tensor ops.
- Legality (own-king safety): compute opponent attack maps in-tensor; pinned-piece and check-evasion masks via the standard between/line lookup tables (also resident gathers).
- Output: `moves [B, M=96] int16` move codes + `legal_mask [B, M]` bool. M=96 covers >99.99% of positions (true max is 218); overflow positions get a `wide` flag and a second padded pass — never silently dropped.
- `make(boards, chosen_move_idx)` is branchless masked bit-surgery; castling/en-passant/promotion handled by precomputed per-move-type delta masks.

### 2.3 Terminal detection (this is what anchors training)

All in-tensor, every step: checkmate (no legal moves ∧ in check), stalemate, 50-move, threefold (`zhist` equality count ≥ 2), insufficient material. These exact game-theoretic values are the *only* ground truth the system ever needs.

### 2.4 Acceptance (M1)

- **Correctness:** differential test vs a reference CPU implementation — perft(5) parity on 100k random positions; zero tolerance.
- **Throughput:** ≥ 1M full legal-movegen+make board-steps/s on the reference GPU (profile and report; stretch 5M/s). The kernel must be CUDA-graph-capturable (no host syncs in the step path).

---

## 3. Architecture — the chess-axis mixer

Attention is deliberately absent. Softmax attention is memory-bandwidth-hungry and mask-heavy; what tensor cores want is dense, static-shape GEMMs. The board is a 64-cell lattice whose geometry is *known* — so mix information along the axes chess pieces actually move along.

### 3.1 Input

`[B, 64, C0]` per-square features: piece one-hot (13 incl. empty) ⊕ side-to-move ⊕ castling(4) ⊕ ep-file(9) ⊕ halfmove bucket(8), broadcast scalars included per square. Linear → `[B, 64, d]`.

### 3.2 LATTICE block (×N)

```
x = x + AxisMix(RMSNorm(x))      # geometry mixing
x = x + SwiGLU_MLP(RMSNorm(x))   # per-square channel MLP, hidden 3d
```

**AxisMix:** four parallel branches, each a tiny shared dense matmul mixing the 8 cells of a line, applied to a quarter of the channels (d/4 each):
- **FileMix:** reshape to [B,8files,8ranks,d/4], GEMM over the rank dim (W ∈ R^{8×8}, per-channel-group).
- **RankMix:** same over files.
- **DiagMix / AntiDiagMix:** permute squares to diagonal-major layout via a precomputed index tensor (15 diagonals padded to 8×8 canvas), same 8×8 GEMM, permute back.
Branches are concatenated back to d and linearly fused (W_o ∈ R^{d×d}).

One block = static dense GEMMs + two free permutations. The whole network compiles into a single CUDA graph. **Acceptance: ≥ 70% tensor-core utilization in the learner's fwd+bwd at batch 4096** (profiler-verified).

Knights and king-distance patterns are not on mixed axes — they're learned through depth (two stacked axis hops reach every knight move); ablation A2 adds an optional fifth "KnightMix" branch (precomputed knight-graph gather + GEMM) to test whether it's worth the channels.

### 3.3 Heads

- **Q-head (central to training):** per-square `f_from = W_f·x`, `f_to = W_t·x` ∈ R^{64×32}; score(move a: s→t, promo p) = `f_from[s]·f_to[t] + b_promo[p]`. One forward yields Q for **all** legal moves (engine masks by `legal_mask`). 
- **V-head:** WDL 3-softmax from mean+max-pooled square features.
- **Moves-left head:** softplus scalar (time management, prefer faster mates).

### 3.4 Configs

| config | d | blocks | params | fwd FLOPs/board |
|---|---|---|---|---|
| LATTICE-base | 192 | 8 | ~3 M | ~0.34 G |
| LATTICE-big | 320 | 12 | ~12 M | ~1.4 G |

Build and gate everything on `base`; `big` is a config change once the loop saturates.

---

## 4. Subsystem S2 — the lockstep training loop

### 4.1 Stream layout

- **Stream G (generation + labeling):** steps the 65,536-game batch; every visited position is labeled (§4.3) and written into the VRAM ring buffer.
- **Stream L (learner):** continuously samples and trains the online net. Target/labeling nets are EMA copies (two of them, §4.4) refreshed every 500 steps.
- GPU budget split ≈ 30% G / 70% L, enforced by stream priorities; both streams run concurrently — the GPU never waits for chess.

### 4.2 Move selection during generation (no search)

`a ~ softmax(Q(s,·)/τ)` over legal moves, ε=0.03 uniform mix, τ annealed 1.2→0.35 over training and per-game (high early plies, low late). Openings: each new game seeds from a 2M-position opening tensor, generated once *on the GPU itself* by high-τ random playouts to ply 6–10, deduplicated in-tensor by Zobrist sort-unique. Draw-adjudication and resign rules with a 10% no-resign holdout, false-resign monitor < 3%.

### 4.3 Q-guided sparse negamax labeling (the core mechanism)

For each visited position s (in megabatch):

1. One forward of the **labeler** (EMA net): Q̂(s,·) over all legal moves, V̂(s).
2. Pick the top-k (k=4) moves by Q̂ **plus 1 uniformly random legal move** (blind-spot insurance).
3. `make` those 5 children in-tensor; one batched forward evaluates V̂(child) for all of them. Terminal children take their exact game-theoretic value instead.
4. Targets:
   - `y_Q(s, a) = −V̂(child_a)` for the 5 expanded moves (others unlabeled/masked in the loss)
   - `y_V(s) = max over expanded a of −V̂(child_a)` (negamax consistency)
   - `y_WDL(s)` additionally mixes the eventual real game outcome: `0.6·consistency + 0.4·outcome`, outcome written back when the game ends (the ring buffer keeps per-game open slots until termination).
5. Cost: ~5–6 net forwards per labeled position, all in batches of tens of thousands — versus ~800 sequential forwards per move for MCTS self-play. That is the source of the speedup, and every one of those forwards is a perfectly-shaped GEMM.

Why this can reach strength without search: terminal states inject exact values; one-ply negamax consistency propagates them backward through the visited state distribution (value iteration with function approximation). The Q-head makes the expansion *self-guided* — the net chooses which children are worth exact evaluation, and the random child keeps the guidance honest. Depth emerges in the weights instead of being recomputed at inference.

### 4.4 Stability kit (mandatory — this is the known failure mode)

Bootstrapped value learning can diverge (deadly triad). Required mitigations, each ablatable:
- **Double EMA labelers:** two EMA nets (decays 0.999 / 0.9995); child values use the elementwise `min` of their −V̂ before the max — counters negamax max-overestimation bias (Double-DQN logic adapted to negamax).
- **Outcome anchoring:** the 0.4 real-outcome mix in y_WDL ties the fixed point to reality.
- **Endgame anchor set:** 2M positions with ≤5 pieces labeled once, offline, with exact tablebase values (the only permitted offline CPU step; ~minutes). Mixed into every 20th learner batch at weight 0.5. Hard ground truth deep in the state space.
- **Prioritized replay:** VRAM ring of 8M records (~1.5 GB); priority = |y_V − V|; sampling by GPU multinomial; importance weights β: 0.4→1.0.
- **Divergence tripwire:** if mean |V| drifts > 0.15 over 50k steps or eval-vs-anchor Elo drops twice consecutively, auto-rollback to last gated checkpoint and halve lr.

### 4.5 Loss

`L = Huber(V, y_V) + Huber(Q_masked, y_Q) + CE(WDL, y_WDL) + 0.3·Huber(ML, plies_left) `, AdamW lr 3e-4 cosine, wd 0.05, grad-clip 1.0, batch 4096, bf16. Each record trained on ≈ 8 times before overwrite (ring sized to enforce this given measured G/L throughputs — assert at runtime).

### 4.6 Throughput budget (the arithmetic the agent must reproduce)

On the reference GPU (~20 TFLOPs effective bf16), LATTICE-base (0.34 GF/board):
- Labeling: 20e12 × 30% / (0.34e9 × 6) ≈ **~3k labeled positions/s** floor (target; measure).
- Learner: fwd+bwd ≈ 3× fwd → 20e12 × 70% / (0.34e9 × 3 × 4096) ≈ **~3.3 steps/s**, consuming ~1.7k fresh-equivalent records/s at 8× reuse. Generation ≥ consumption → balanced; tune the 30/70 split empirically.
- **Gates (M3):** GPU SM-busy ≥ 90%, CPU ≤ 1 core, ≥ 2.5k labeled positions/s sustained for 24h, zero host syncs in the steady-state loop (Nsight-verified).

---

## 5. Subsystem S3 — inference

Primary mode is **searchless**: encode position, play `argmax Q` over legal moves (temperature 0). The consistency training is what makes this viable — Q has lookahead baked in.

Optional strength mode, still tensor-only: **lattice beam** — fixed-shape beam search (width 16, depth 6) as batched expand→evaluate→top-k ops on GPU; ~100 board-evals per move, ~10 ms. No tree, no recursion, no CPU.

UCI bridge on CPU translates positions/moves only. (CPU int8 inference is explicitly out of scope — this system's home is the GPU.)

Evaluation harness: gating matches between checkpoints (SPRT elo0=0, elo1=5) using lattice beam at fixed evals; absolute anchor matches vs a fixed external engine level; 2k-position tactics suite.

---

## 6. Data preparation

Deliberately almost nothing — that is the point. The loop generates its own curriculum. The only offline artifacts (one-time, scripted):
1. opening seed tensor (GPU-generated, §4.2),
2. endgame anchor set with tablebase values (§4.4),
3. held-out eval suites (tactics, anchor openings).
No PGN ingestion, no corpora, no shards, no dataloaders.

## 7. Milestones

| # | deliverable | acceptance |
|---|---|---|
| M0 | literature check (GPU-vectorized chess, consistency-trained chess nets) | written diff vs closest prior work; adjust if scooped |
| M1 | GPU chess kernel | §2.4 correctness + throughput |
| M2 | net + loop, 1h smoke run | loss ↓, no divergence, tripwires functional |
| M3 | steady-state training | §4.6 gates — **the 90% GPU / ≤1 CPU core gate is the contract** |
| M4 | 7-day unattended run | monotone gating Elo; searchless mode beats material+PST baseline ≥ 95%; tactics ≥ 40% |
| M5 | LATTICE-big + lattice beam | ≥ +200 Elo over base searchless; budget re-verified |

Ablations: A1 double-min off (measure value blow-up), A2 KnightMix branch, A3 random-child off, A4 outcome-anchor off, A5 k ∈ {2,4,8}, A6 axis-mixer vs same-size ResNet (architecture headline claim).

## 8. Risks & fallbacks

| risk | detection | fallback |
|---|---|---|
| value iteration converges to a weak fixed point (positional blindness without search) | M4 anchor Elo plateau | raise k; add depth-2 targets on top-1 child (still tensor ops, ~2× label cost); last resort: shallow lattice-beam targets during labeling (search-lite, still GPU) |
| TD divergence | tripwire §4.4 | rollback machinery is specced; tighten EMA decays |
| movegen kernel correctness | M1 differential test | reference impl disagreement = hard stop |
| M=96 padding waste | profiler | dynamic bucketing by legal-move count |
| axis-mixer underperforms attention per-param | A6 | swap body for attention, keep everything else — the loop, kernel, and labeling are architecture-agnostic |

— end of spec —
