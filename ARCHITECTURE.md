# Architecture

This document is the long version of [README.md § Architecture](./README.md#architecture). It is written for someone who wants to read or modify the code, not for first-contact.

## Goals and non-goals

glassbox optimises for, in order:

1. **Inspectability.** Every intermediate tensor must be addressable and visualisable without rebuilding the model. Performance loses to inspectability if they conflict.
2. **Browser-native.** No Python runtime, no server. Everything ships as a wasm bundle plus weight blobs streamed from a CDN.
3. **Pedagogy.** A reader who has read the framework paper should be able to find the matrix multiplications it describes by ctrl-F-ing the source.
4. **Performance, within the above.** WebGPU compute, fp16 weights, bump-arena allocation, but never at the cost of points 1–3.

It deliberately does not optimise for:

- Training large models. The autograd is a reference implementation for in-browser probe-fitting, not a production framework.
- Multi-batch throughput. We run with `batch=1` and a small prompt; tiling is tuned for that.
- Maximal model coverage. GPT-2 and Pythia are first-class. Llama, Mistral, Qwen are roadmap, not v1.

## Crate boundaries

```
glassbox-core       no_std-friendly, no I/O, no GPU.
glassbox-runtime    op dispatch, backends. Depends on -core.
glassbox-models     architecture definitions and weight loaders.
                    Depends on -core and -runtime.
glassbox-interp     hooks, patching, DLA, probes.
                    Depends on -core, -runtime, -models.
glassbox-wasm       wasm-bindgen surface. Depends on everything.
```

The dependency graph is a DAG by design. `glassbox-core` knows nothing about WebGPU; `glassbox-runtime` knows nothing about GPT-2; `glassbox-models` knows nothing about hooks. This costs a few crate-boundary trait objects but pays out when you want to swap a backend or add a model.

### glassbox-core

`Tensor`, `DType`, `Shape`, `Stride`, `Layout`. f32 / f16 / bf16 / i32 / u32 dtypes. Strided row-major. Views are zero-copy and share the underlying `Storage`.

`Storage` is an enum: `Cpu(Arc<[u8]>)` for owned host memory or `Gpu(BufferId)` for a handle into the runtime's buffer arena. The split lets `glassbox-core` stay backend-free; the GPU side of the enum is just a `u64` handle that the runtime resolves.

No allocation in `core` is unbounded. Tensors carry a `&'arena Arena` reference and live for the duration of one forward pass.

### glassbox-runtime

The runtime is two pieces stacked: a small SSA-form op graph, and a `Backend` trait that lowers each op.

```rust
pub trait Backend {
    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()>;
    fn softmax(&self, x: &Tensor, axis: usize, out: &mut Tensor) -> Result<()>;
    fn layer_norm(&self, x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f32, out: &mut Tensor) -> Result<()>;
    fn gelu(&self, x: &Tensor, out: &mut Tensor) -> Result<()>;
    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: AttentionMask, out: &mut Tensor, pattern_out: Option<&mut Tensor>) -> Result<()>;
    // ...
}
```

Two implementations: `CpuBackend` (rayon-parallel f32, used by tests and the WASM fallback) and `WgpuBackend` (WGSL kernels). The CPU backend is the spec; every WGSL kernel has a parity test against it.

The op graph is built once per architecture, not per forward pass. `glassbox-models` constructs it during model load and the runtime walks it for each token.

#### Buffer arena

GPU buffers are expensive to allocate and free. The arena reserves one large GPU buffer at model load and hands out aligned sub-ranges per intermediate tensor. At end-of-forward-pass it resets the bump pointer; weights and KV cache live in a separate arena that survives.

Sizing: GPT-2 small with `seq_len=1024`, `batch=1` peaks around 38 MB of intermediate state (fp16). The arena is sized at 64 MB by default with a `--max-seq` flag.

#### KV cache

Standard. `K` and `V` per layer are appended in-place; the cache buffer is sized at construction to `(num_layers, 2, max_seq, num_heads, head_dim)` fp16. No reallocations during generation.

### glassbox-models

`Model` is a trait with `forward(&self, tokens, hooks) -> Logits`. `GPT2` and `Pythia` are the implementations. Each owns its weight buffers, op graph, and tokenizer.

Weights are loaded from a custom flat format (`.glx`) — see [§ Weight format](#weight-format) — not safetensors directly. The conversion script lives in `scripts/convert_weights.py`.

### glassbox-interp

The interp layer does not run the model. It is a set of:

- **Hooks** — named subscription points the model writes intermediates to.
- **Patches** — a `HashMap<HookName, Tensor>` the model reads from instead of computing, when the entry exists.
- **Probes** — small structs that derive a quantity from one or more hooks (e.g., `LogitAttribution { layer: 7 }` reads `blocks.7.resid_post` and returns `resid @ W_U`).

Hook names are dotted paths matching the op-graph node names: `blocks.4.attn.pattern`, `blocks.7.mlp.post_act`, `unembed`. They are stable contract; renaming them is a breaking change.

### glassbox-wasm

`wasm-bindgen`-decorated facade. The TypeScript surface is generated by a small script that walks the `#[wasm_bindgen]` items and emits a `.d.ts` plus a typed wrapper. We do not depend on `serde-wasm-bindgen` for hot paths — large tensors cross the JS↔WASM boundary as `Uint8Array` views over WASM memory, zero-copy.

## Weight format (`.glx`)

A `.glx` file is a flat binary with a small JSON header.

```
[u64 little-endian header_len]
[utf8 JSON header of length header_len]
[raw fp16 / fp32 weight payload]
```

The header lists every tensor: name, dtype, shape, byte offset, byte length. The payload is concatenated tensors with no padding; alignment is enforced by ordering the tensors so that 4-byte and larger types come first.

Reasoning: the browser fetches one blob, slices it once, and hands sub-views directly to WebGPU `createBuffer({ mappedAtCreation: true })`. No JSON-of-floats, no protobuf, no archive format with seek tables. A 124M parameter GPT-2 fits in 248 MB at fp16 and streams in under three seconds on a 100 Mbps connection with HTTP range-request prefetching.

## WGSL kernel notes

`shaders/matmul.wgsl` is the workhorse. Tiled, 16×16 workgroups, two levels of shared memory. Tuned per backend on first use: glassbox runs a 30 ms autotune on model load that picks tile sizes for the discovered adapter and persists them in `localStorage`.

`shaders/attention.wgsl` is fused — Q·K^T, scale, mask, softmax, attn·V — to avoid round-tripping the `(seq, seq)` attention pattern through GPU memory unless a hook asks for it. When `pattern_out` is `Some`, it is also written to a side buffer in a separate dispatch.

Numerical precision: matmul accumulates in fp32, weights and activations in fp16. LayerNorm and softmax use online algorithms (Welford, max-subtract) for numerical stability. The parity tests guarantee `≤ 1e-3` element-wise vs the f32 CPU reference, which is sufficient for the layer outputs of GPT-2 small to agree on the argmax token over 100k random prompts.

## Hook protocol

```rust
pub struct Hooks<'a> {
    pub publish: &'a dyn Fn(&str, Tensor),
    pub patch:   &'a dyn Fn(&str) -> Option<Tensor>,
}
```

Models call `publish("blocks.4.attn.pattern", &pattern)` after computing each named intermediate. Before the same op runs, they call `patch("blocks.4.attn.pattern")`; if it returns `Some`, the model substitutes that tensor into the rest of the graph.

This is the same idea as TransformerLens's `HookedTransformer`, ported to a no-Python world. The full hook list per model lives in `crates/glassbox-models/HOOKS.md`.

## Performance budget

WebGPU forward pass for GPT-2 small, single token, RTX 3060 reference:

| Stage                | Time (ms) | Share |
|----------------------|-----------|-------|
| Embedding + pos enc  |       0.3 |    3% |
| Attention (×12)      |       6.1 |   51% |
| MLP (×12)            |       4.2 |   35% |
| LayerNorm + residual |       0.8 |    7% |
| Unembed + sample     |       0.5 |    4% |
| **Total**            |    **12** |  100% |

Attention and MLP together are 86% of the budget; everything else is out of scope for optimisation. The next 2× will come from a Flash-style fused attention kernel and from quantising the MLP weights to int8. Both are tracked in [docs/perf-notes.md](./docs/perf-notes.md).

## Testing

- `cargo test -p glassbox-core` — shape arithmetic, dtype conversions, view semantics.
- `cargo test -p glassbox-runtime --features cpu` — every op against hand-derived expected outputs.
- `cargo test -p glassbox-runtime --features wgpu` — every op against the CPU backend, parity within `1e-3`.
- `cargo test -p glassbox-models` — load a tiny stub model, check the op graph topology and parameter count.
- `bun test` in `web/` — UI components, hook subscription wiring.
- `just integration` — boots the dev server, generates from GPT-2 with both backends, asserts both produce the same argmax token for 32 fixed prompts.

The WebGPU integration test runs in headless Chrome via `playwright`; it requires a host with a GPU exposed.

## Versioning

Pre-1.0. The hook-name vocabulary, `.glx` format, and TypeScript surface are the three things that will break compatibility on bumps. Internal Rust APIs are not stable.
