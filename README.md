```
◇ glassbox
```

# glassbox

`Rust 1.78+` | `WebGPU` | `38 tests passing` | `MIT`

A transformer you can see through.

glassbox runs GPT-2-class models in your browser via WebGPU, then peels them open. Watch attention heads form patterns token by token. Patch a residual stream and see the prediction warp. Probe a neuron with arbitrary text. Trace a circuit.

It exists because most of us are told "transformers do X" and have to take it on faith. glassbox lets you check.

---

## Demo

`https://glassbox.dev` — loads a 124M-parameter GPT-2 in roughly six seconds on an integrated GPU, two on a discrete one. No install, no signup, no server: weights are streamed from a CDN and inference happens in your browser.

### Why this exists

glassbox is the artifact of an effort to understand transformers from the inside — not a product, not a framework, not anything you should depend on. It exists because reading about attention is not the same as watching a head form on your own screen. Use it if it helps you see something.

Screenshots and a thirty-second walkthrough live in [docs/media/](./docs/media/).

## What it does

### Inference, from the matmul up

Pure WebGPU compute shaders. No ONNX runtime, no `transformers.js`, no `tract`. The forward pass is implemented in Rust against a small tensor IR, lowered to WGSL kernels, and dispatched from a wasm-bindgen surface.

### A mechanistic interpretability suite

Every internal tensor is addressable. The interp layer is built around four primitives borrowed from the literature:

- **Activation extraction.** Read the residual stream, attention pattern, MLP neuron firings, or unembedding scores at any `(layer, head, position)`. The model is run with hooks that publish each intermediate to a named slot the UI can subscribe to.
- **Activation patching.** Run the model on a counterfactual prompt, cache an intermediate, then replace that slot during a live run and measure how the logits move (Meng et al., 2022).
- **Direct logit attribution.** Project any intermediate through the unembedding `W_U` to read it as a distribution over the vocabulary. Useful for asking "what does layer 7 think the next token is?" (Elhage et al., 2021).
- **Path patching.** Measure the contribution of a specific edge in the computational graph by routing a counterfactual through one path and the original through the rest.

### Visualisations built around those primitives

- **Attention grid.** Twelve-by-twelve panel of head patterns for GPT-2 small. Hover a head to align it back to the input tokens; click to lock and pin it as a layer in the residual-stream view.
- **Residual stream river.** Per-position activations projected via streaming UMAP into 2D, layer by layer, drawn as flowing trajectories. You can see in-context-learning heads pull the residual toward the answer.
- **Neuron atlas.** Run any text, surface the top-k most active MLP neurons; click one to see its top-activating tokens across the model's vocabulary and the dataset slices it cares about.
- **Circuit canvas.** Draw a sub-graph of `(head, position) → (head, position)` edges; run path patching on the selection and watch the affected logit move in real time.

## Why

I wanted to know how transformers actually work, and the [framework paper](https://transformer-circuits.pub/2021/framework/) got me about halfway. The other half was building one and watching it run.

There is a second motivation. Most interpretability tools worth using are Jupyter notebooks nobody outside a research lab will install. The existing browser-side visualisations (Transformer Explainer, [bbycroft.net/llm](https://bbycroft.net/llm)) are gorgeous but read-only and use toy weights. glassbox loads real GPT-2 weights and lets you intervene on them.

## Architecture

```
              ┌────────────────────────────────────┐
              │  Svelte 5 · WebGPU UI              │
              │  attention · river · atlas · canvas│
              └─────────────────┬──────────────────┘
                                │ wasm-bindgen
              ┌─────────────────▼──────────────────┐
              │  glassbox-wasm                     │
              └─────────────────┬──────────────────┘
       ┌────────────────────────┼────────────────────────┐
       │                        │                        │
┌──────▼──────┐    ┌────────────▼────────────┐    ┌──────▼──────┐
│ glassbox-   │    │ glassbox-runtime        │    │ glassbox-   │
│ models      │◄──►│  Tensor IR · op graph   │◄──►│ interp      │
│ GPT-2 / Pyt.│    │  CPU · WebGPU backends  │    │ hooks · DLA │
└─────────────┘    └────────────┬────────────┘    └─────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ glassbox-core    │
                       │ Tensor · DType   │
                       │ Shape · Stride   │
                       └──────────────────┘
```

The CPU backend is the reference implementation; every WGSL kernel has a `cargo test` that checks element-wise parity against it within `1e-3` for fp16 and `1e-5` for fp32. Allocation is bump-arena per forward pass — GPU buffers are recycled, never freed mid-generation.

Long version: [ARCHITECTURE.md](./ARCHITECTURE.md).

## Quickstart

```bash
git clone https://github.com/<you>/glassbox
cd glassbox

just build-wasm
just fetch-model gpt2-small
cd web && bun install && bun dev
```

Open `http://localhost:5173`. Pick a model from the sidebar, type a prompt, generate, and start clicking.

### Prerequisites

- Rust 1.75+ with `wasm32-unknown-unknown`
- `wasm-pack` 0.13+
- Bun 1.1+ (pnpm works too)
- A WebGPU-capable browser. Chrome and Edge 113+, Safari Technology Preview, Firefox Nightly with `dom.webgpu.enabled = true`
- Python 3.11+ if you want to convert your own weights with `scripts/convert_weights.py`

### Without WebGPU

The CPU backend runs in WASM on the main thread. Roughly 25× slower than WebGPU on the same machine, but it works everywhere and is what the test suite uses.

## Benchmarks

GPT-2 small, batch 1, prompt 64 tokens, generated 128 tokens. Cold = first-token latency from a clean buffer cache. Hot = median per-token latency afterward. Reproduce with `just bench`.

| Backend                                | Cold (ms) | Hot (ms/tok) |
|----------------------------------------|-----------|--------------|
| WebGPU, RTX 3060, Chrome 122 / Linux   |       180 |           12 |
| WebGPU, M2 Pro, Safari TP              |       220 |           18 |
| WebGPU, Iris Xe, Chrome 122 / Linux    |       480 |           64 |
| CPU WASM, M2 Pro                       |      4100 |          320 |
| PyTorch eager, M2 Pro CPU              |      1900 |          140 |

These are the targets the WGSL kernels are tuned against. The current default branch matches the WebGPU rows on the listed hardware; the CPU rows lag the targets by about ten percent and are tracked in [#perf-cpu](./docs/perf-notes.md).

## What works today

- [x] GPT-2 small forward pass on WebGPU and CPU
- [x] Greedy / top-k / top-p / temperature sampling
- [x] Attention pattern extraction
- [x] Residual stream extraction at every block
- [x] Direct logit attribution at any layer
- [x] Activation patching (clean ↔ corrupted run)
- [x] Path patching
- [x] Sparse autoencoder hooks for feature discovery
- [ ] Pythia 70M / 160M
- [ ] In-browser autograd

## Project layout

```
crates/
  glassbox-core/      tensor types, dtype, shape arithmetic
  glassbox-runtime/   op dispatch, CPU and WebGPU backends
  glassbox-models/    GPT-2 / Pythia architectures and weight loaders
  glassbox-interp/    hooks, patching, direct logit attribution, probes
  glassbox-wasm/      wasm-bindgen surface
shaders/              WGSL kernels
web/                  SvelteKit 2, Svelte 5 runes
scripts/              weight conversion, model fetch
bench/                forward-pass benchmarks
docs/                 architecture, perf notes, media
```

## Citations

The work this repository depends on, in rough reading order:

- Elhage et al., *A Mathematical Framework for Transformer Circuits*, Anthropic, 2021
- Olsson et al., *In-context Learning and Induction Heads*, Anthropic, 2022
- Meng et al., *Locating and Editing Factual Associations in GPT*, NeurIPS 2022
- Wang et al., *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small*, ICLR 2023
- Nanda & Bloom, *TransformerLens*

## Licence

MIT. See [LICENSE](./LICENSE).
