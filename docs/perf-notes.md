# Performance notes

A working ledger of where time is spent and what we have tried. Numbers are RTX 3060 / Chrome 122 / Linux unless noted.

## Forward pass, GPT-2 small, single token

| Stage                | Time (ms) | Share |
|----------------------|-----------|-------|
| Embedding + pos enc  |       0.3 |    3% |
| Attention (×12)      |       6.1 |   51% |
| MLP (×12)            |       4.2 |   35% |
| LayerNorm + residual |       0.8 |    7% |
| Unembed + sample     |       0.5 |    4% |
| **Total**            |    **12** |  100% |

Attention and MLP are 86% of budget. Everything else is out of scope.

## Open work

### Flash-style fused attention

The current attention kernel materialises the `(seq, seq)` pattern in GPU memory because the UI sometimes wants to read it. Profile shows ~40% of the attention kernel is the round-trip. Switch to an online-softmax fused kernel (Q→K^T→softmax→@V in registers); only materialise the pattern when a hook is subscribed.

Expected: 1.4–1.8× attention speedup. Tracking issue: TBD.

### MLP int8 quantisation

The 3072×768 and 768×3072 projections are the largest matmuls. Weight-only int8 with per-channel scale, fp32 accumulate. Calibration on WikiText-2 to set scales. Should keep PPL within 0.1 of fp16.

Expected: 2× MLP speedup, halves the weight blob. Tracking issue: TBD.

### Matmul autotune on first model load

Tile sizes baked at 16×16 today. On Apple Silicon, 32×8 matches CoreML's heuristic better. Add a 30 ms autotune that walks {16×16, 16×32, 32×16, 32×8, 8×32} for one full transformer block, picks the best, and persists the choice in `localStorage`.

Expected: 5–15% on M-series, 0–5% on discrete GPUs.

## Things that did not pan out

- **fp16 accumulation in matmul.** Saved 12% but PPL on WikiText-2 rose by 0.4. Reverted.
- **`@compute @workgroup_size(32, 32)` in matmul.** Slower than 16×16 on every adapter we tested. 16×16 hits the shared-memory sweet spot.
- **Shared-memory softmax with shared sum reduction across rows.** The cross-row barrier dominated; per-row workgroups are simpler and faster.

## CPU backend

The CPU backend lags the targets by ~10%. Profile shows the matmul triple loop is not autovectorising; trying `wide` for explicit SIMD next.
