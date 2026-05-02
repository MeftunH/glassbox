# Hooks

The full vocabulary of named intermediate states a glassbox model emits. Hook names are dotted paths and form a stable contract between the runtime and the UI; renaming any of them is a breaking change.

## GPT-2

For each block `l` in `0..n_layer`:

| Hook name                       | Shape                            | Description                                                                                  |
|---------------------------------|----------------------------------|----------------------------------------------------------------------------------------------|
| `blocks.{l}.resid_pre`          | `(seq, n_embd)`                  | Residual stream entering block `l`.                                                          |
| `blocks.{l}.attn.q`             | `(seq, n_head, head_dim)`        | Query projections after the c_attn split.                                                    |
| `blocks.{l}.attn.k`             | `(seq, n_head, head_dim)`        | Key projections.                                                                             |
| `blocks.{l}.attn.v`             | `(seq, n_head, head_dim)`        | Value projections.                                                                           |
| `blocks.{l}.attn.pattern`       | `(n_head, seq, seq)`             | Softmaxed attention pattern. Causal mask is already applied; entries above the diagonal are zero. |
| `blocks.{l}.attn.z`             | `(seq, n_head, head_dim)`        | Per-head attention output before the c_proj merge.                                           |
| `blocks.{l}.attn.out`           | `(seq, n_embd)`                  | Attention output after c_proj.                                                               |
| `blocks.{l}.resid_mid`          | `(seq, n_embd)`                  | Residual stream after attention, before MLP.                                                 |
| `blocks.{l}.mlp.pre`            | `(seq, 4 * n_embd)`              | Output of c_fc, before GELU.                                                                 |
| `blocks.{l}.mlp.post`           | `(seq, 4 * n_embd)`              | Output of GELU, the canonical "neuron activations" for the layer.                            |
| `blocks.{l}.mlp.out`            | `(seq, n_embd)`                  | Output of c_proj.                                                                            |
| `blocks.{l}.resid_post`         | `(seq, n_embd)`                  | Residual stream leaving block `l`.                                                           |

Globals:

| Hook name        | Shape              | Description                                                |
|------------------|--------------------|------------------------------------------------------------|
| `embed`          | `(seq, n_embd)`    | Token + position embedding sum, before block 0.            |
| `final_ln`       | `(seq, n_embd)`    | Output of the final LayerNorm.                             |
| `unembed`        | `(seq, vocab)`     | Logits, equal to `final_ln @ wte^T`.                       |

## Conventions

- Subscriptions are explicit: a hook that nobody subscribes to does not allocate a capture buffer.
- Patches are read once per forward pass at the matching emit point. A patch can be installed for any hook above; the model will substitute the patch tensor and continue with it as the input to subsequent ops.
- Shapes are written without the leading `batch` dimension because the browser-side runtime is `batch=1` everywhere. The Rust types do carry the batch axis.

## Why this exact set

It mirrors the TransformerLens `HookedTransformer` convention so existing literature and notebooks translate directly. The two minor extras (`resid_pre`, `resid_mid`, `resid_post` separately) are kept because mech-interp work routinely needs the stream both before and after a sublayer adds into it; reconstructing one from the other is cheap but error-prone.
