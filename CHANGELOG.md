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
- ARCHITECTURE.md, perf-notes, GPT-2 hook reference.

### Known limitations

- The wasm bundle currently routes through the CPU backend; switching it to WebGPU in the browser is now a one-line change once we wire `WgpuBackend::new` into the wasm entry point.
- SAE feature discovery is wired in the wasm bindings but the corresponding UI panel is not yet built; users can drive it from the browser console via the `Glassbox` handle.
