# Changelog

## Unreleased

### Added

- Workspace skeleton: `glassbox-core`, `glassbox-runtime`, `glassbox-models`, `glassbox-interp`, `glassbox-wasm`.
- CPU backend with parity-tested matmul, softmax, layernorm, GELU, and causal attention (rayon-parallel f32 reference).
- WebGPU backend wired through the dispatcher: matmul, softmax, layernorm, GELU, and attention now run on real WGSL kernels with element-wise parity tests against the CPU reference (matmul `1e-4`, softmax/layernorm/attention `1e-3`, GELU `5e-3`).
- WGSL kernel set: tiled matmul, online softmax, Welford layernorm, GELU, mask-aware fused attention.
- `.glx` flat weight format and a Python conversion script for HF GPT-2 checkpoints.
- Hook contract for activation publication and patch substitution; `HookRegistry` lives in the runtime so models, interp, and the wasm surface share one type.
- End-to-end GPT-2 forward pass (`Gpt2Runner`) on top of the backend trait, with hook capture and patch interception at every named intermediate.
- `generate(prompt, max_new, sampling)` on the wasm surface returning tokens, decoded text, elapsed ms, and tok/s.
- Path patching primitive (`run_path_patch`) implementing the clean/corrupt/repatch dance from Wang et al., with a recovery score.
- Sparse autoencoder primitive (`SparseAutoencoder` with ReLU encode + linear decode) and `top_k_features` for feature discovery on captured activations.
- SvelteKit 2 + Svelte 5 (runes) web shell with attention grid, residual river, neuron atlas, circuit canvas, and path-patching panel views; the attention grid reads real captured patterns when a model is loaded.
- Wasm surface for the interp primitives: `runPathPatch`, `loadSae` / `encodeSaeFromHook`, `installPatch` / `clearPatches`. The path-patching panel drives the first three end-to-end from the browser.
- GPU-resident tensor path on `WgpuBackend`: a buffer pool maps every `BufferId` to a live `wgpu::Buffer`, ops accept GPU-resident inputs and emit GPU-resident outputs without intermediate readback. New `Backend::alloc`, `upload`, and `download` round out the trait. Six dedicated tests cover round-trip, chained matmul (no readback between ops), and softmax/attention parity against the CPU reference.
- WebGPU is wired through to the wasm bundle and the browser. `Glassbox.fromBlobWebGpu(blob)` is an async constructor that initialises a real `WgpuBackend` against `navigator.gpu`. The web app probes for WebGPU on load, falls back to CPU on browsers without it, and tags the active backend in the top bar.
- Pure-GPU `add` and `embed` WGSL kernels (replace previous CPU round-trips). The wgpu backend now never falls back to CPU during the per-block forward op set.
- `WgpuBackend::download_async` for async readback via `futures-channel` oneshot, gated on `device.poll(Wait)` only on native targets so the browser path is non-blocking.
- Sparse-autoencoder feature explorer view (`SaePanel.svelte`): load an SAE from a JSON file, probe an arbitrary text against any captured hook, and see the top-K firing features as bar plots.
- Async runner: `Gpt2RunnerAsync` and `AsyncBackend` trait (with `download_async`) so the browser path can `await` GPU readbacks instead of blocking the JS event loop. The wasm surface gains `Glassbox.generateAsync(prompt, max_new, args)` returning a `Promise<GenerateOutput>`.
- End-to-end validation against real GPT-2: a `cargo run --release --example cli_generate` example now loads `models/gpt2-small.glx` (124M params, 477 MB) and produces coherent English continuations through the Rust forward pass on the CPU backend.
- BPE byte-encoding fix in `glassbox-models::tokenizer`: spaces and newlines now byte-encode to `Ġ`/`Ċ` during pretokenisation so HF GPT-2 vocab merges resolve correctly. Encode/decode round-trip is regression-tested.
- ARCHITECTURE.md, perf-notes, GPT-2 hook reference.

### Known limitations

- The runner's helpers (`split_qkv`, `reshape_to_heads`, `merge_heads`, `add_row_bias`) operate on CPU tensors and currently call the sync `Backend::download` to reach them. Native WebGPU works end-to-end; in the browser those sync downloads would block the JS event loop, so the browser path defaults to the CPU backend. Promoting the runner to async (using `WgpuBackend::download_async`, which is already in place) is what unlocks browser-WebGPU generate.
- SAE feature discovery is wired in the wasm bindings but the corresponding UI panel is not yet built; users can drive it from the browser console via the `Glassbox` handle.
