# Changelog

## Unreleased

### Added

- Workspace skeleton: `glassbox-core`, `glassbox-runtime`, `glassbox-models`, `glassbox-interp`, `glassbox-wasm`.
- CPU backend with parity-tested matmul, softmax, layernorm, GELU, and causal attention.
- WGSL kernel set: matmul (tiled), softmax (online, two-pass), layernorm (Welford), GELU, attention (mask-aware).
- `.glx` flat weight format and a Python conversion script for HF GPT-2 checkpoints.
- Hook contract for activation publication and patch substitution; `HookRegistry` implements the runtime side.
- SvelteKit 2 + Svelte 5 web shell with attention grid, residual river, neuron atlas, and circuit canvas views.
- ARCHITECTURE.md, perf-notes, and the GPT-2 hook reference.

### Known limitations

- WebGPU backend present but not yet wired to the dispatcher; ops fall back to CPU.
- Inference end-to-end runs only on stubbed activations in the UI.
- No path-patching implementation yet.
