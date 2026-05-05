# Performance notes

A working ledger of where time is spent and what we have tried. CPU baseline numbers are
measured on an Intel Core i7-9750H (6c/12t, 2.6 GHz base) with the rayon thread pool at its
default size; the WebGPU path targets RTX 3060 / Chrome 122 unless noted.

## CPU baseline — GPT-2 small, fp32, real weights

Measured via `cargo run --release --example cli_generate models/gpt2-small.glx "<prompt>" <n>`.
Forward pass recomputes the full sequence on every step — there is no KV cache yet, so
per-token wall time grows with the current sequence length.

| prompt tokens | new tokens | wall (ms) | tok/s | mean ms/forward |
|---------------|-----------:|----------:|------:|----------------:|
|             7 |          1 |       554 |   1.8 |             554 |
|             7 |          8 |     6 189 |   1.3 |             774 |
|             7 |         32 |    57 972 |   0.6 |           1 812 |
|             9 |         16 |    20 004 |   0.8 |           1 250 |
|            20 |         16 |    27 712 |   0.6 |           1 732 |

Forward at seq=7 is ~554 ms, at seq~22 it is ~1.8 s — consistent with attention's `O(n²)`
contribution stacked on the per-block MLP. Adding a KV cache collapses every column above
to a constant ~per-token cost; that is the next leverage point on the CPU path.

## Open work

### KV cache for the runner

Today `Gpt2Runner::forward` rebuilds Q, K, V for every token. Caching K and V per layer
(append-only along the sequence axis) flattens the curve above to a single per-token
cost equal to the seq=7 number. Same for the async runner.

Expected: 5–8× speedup on `max_new ≥ 32`. Tracking issue: TBD.

### Flash-style fused attention (GPU path)

The current attention kernel materialises the `(seq, seq)` pattern in GPU memory because
the UI sometimes wants to read it. Profile shows ~40% of the attention kernel is the
round-trip. Switch to an online-softmax fused kernel (Q→Kᵀ→softmax→@V in registers) and
only materialise the pattern when a hook is subscribed.

Expected: 1.4–1.8× attention speedup. Tracking issue: TBD.

### MLP int8 quantisation

The 3072×768 and 768×3072 projections are the largest matmuls. Weight-only int8 with
per-channel scale and fp32 accumulate. Calibrate on WikiText-2 to set scales. Should keep
PPL within 0.1 of fp32.

Expected: 2× MLP speedup, halves the weight blob. Tracking issue: TBD.

### Matmul autotune on first model load

Tile sizes baked at 16×16 today. On Apple Silicon, 32×8 matches CoreML's heuristic better.
Add a 30 ms autotune that walks {16×16, 16×32, 32×16, 32×8, 8×32} for one full transformer
block, picks the best, and persists the choice in `localStorage`.

Expected: 5–15% on M-series, 0–5% on discrete GPUs.

## Done

- **GPU-resident tensors.** `Backend::alloc/upload/download` and a buffer pool inside
  `WgpuBackend` mean ops no longer round-trip activations through the CPU. Weights upload
  once at model load; intermediates stay on the GPU between ops; only the final logits come
  back. Matmul-chain test confirms two consecutive matmuls with zero readback in between,
  parity-matched against the CPU reference.
- **Async readback.** `WgpuBackend::download_async` resolves through a `futures-channel`
  oneshot so the browser's JS event loop is never blocked while waiting for `mapAsync`.
  The wasm surface exposes `Glassbox.generateAsync(prompt, max_new, args) → Promise` for
  the WebGPU backend; the page picks `generateAsync` when `activeBackend === 'webgpu'`
  and the sync path on CPU.
- **End-to-end real-weight validation.** The CPU backend produces coherent English
  continuations from the published HF GPT-2 124M weights (see `cli_generate` example), so
  the BPE tokenizer, weight layout, attention mask, and unembed are all correct.

## Things that did not pan out

- **fp16 accumulation in matmul.** Saved 12% but PPL on WikiText-2 rose by 0.4. Reverted.
- **`@compute @workgroup_size(32, 32)` in matmul.** Slower than 16×16 on every adapter
  we tested. 16×16 hits the shared-memory sweet spot.
- **Shared-memory softmax with shared sum reduction across rows.** The cross-row barrier
  dominated; per-row workgroups are simpler and faster.

## CPU backend

The single-token forward at seq=7 lands at ~554 ms on a 6-core laptop CPU with rayon
saturating all threads. The MLP's two large matmuls (`768→3072`, `3072→768`) are the bulk
of that. `cargo asm` shows the inner triple loop is autovectorising into AVX2 fmas, so the
remaining headroom is algorithmic (KV cache) and quantisation, not microarchitectural.
