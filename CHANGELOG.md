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
- SvelteKit 2 + Svelte 5 (runes) web shell with attention grid, residual river, neuron atlas, and circuit canvas views; the attention grid reads real captured patterns when a model is loaded.
- ARCHITECTURE.md, perf-notes, GPT-2 hook reference.

### Known limitations

- Tensors are CPU-resident around each WebGPU call (upload + dispatch + download per op). GPU-resident weights and intermediates are next on the perf list; the kernels themselves are already there.
- Generate runs on the CPU backend in WASM; routing the runner through the WebGPU backend in the browser is the natural follow-up.
- The web app ships UI scaffolding for path patching and SAE feature discovery, but the wasm bindings for those primitives are not yet exposed (only the Rust-side `glassbox-interp` API).
