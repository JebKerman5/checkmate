# checkmate-alpha1 VM Profile Summary

VM: Thunder Compute `0`, 2x NVIDIA A100-SXM4-80GB, 8 vCPU, PyTorch 2.11.0+cu130.
Commit: `2be62e4`.

## Single-GPU CUDA Benchmark

- Model forward: 14,249.78 boards/s at batch 128.
- Learner step: 16.866 steps/s at batch 128.
- Replay write: 130,086.22 records/s.
- Replay priority sample: 74,486.90 records/s.
- Replay uniform sample: 150,254.46 records/s.
- Movegen reference hook: 1,027.71 board steps/s.
- Movegen vectorized tensor hook: 126.75 board steps/s with 1 host sync.
- Compiled learner: skipped because `torch.compile` could not locate `libcuda.so` in the VM linker cache.

## Two-GPU DDP Benchmark

- World size: 2.
- Global batch: 256.
- Min rank learner throughput: 11.9186 steps/s.
- Max rank learner throughput: 14.8666 steps/s.
- Rank skew: 2.9479 steps/s.
- Mean all-reduce share: 0.812%.
- Weight checksum max delta: 0.0.
- Host sync metric: 0.

## Bottleneck Selection

The first alpha hot path remains chess move generation/make. The vectorized tensor hook is slower than the CPU reference hook in this small fixture and still reports a host synchronization. Replay priority sampling is also nontrivial, but the uniform replay path is already about 2.02x faster than priority sampling and is available behind the `ReplayConfig.sampling_mode` preset.

## Remaining Acceptance Gaps

- Real integrated generation/labeling/replay training is not implemented yet, so the ten-minute integrated smoke and one-hour alpha candidate cannot be honestly completed.
- The final CUDA/Triton chess kernel is still not implemented.
- `torch.compile` needs VM linker/cache setup before compiled-mode speed can be compared.
