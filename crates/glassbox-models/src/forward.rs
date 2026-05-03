use std::sync::Arc;

use glassbox_core::{CoreError, Shape, Tensor};
use glassbox_runtime::{AttentionMask, Backend, HookRegistry};

use crate::error::{ModelError, Result};
use crate::gpt2::{hooks as gpt2_hooks, Gpt2};

pub struct Gpt2Runner<'a> {
    model: &'a Gpt2,
    backend: &'a dyn Backend,
    hooks: Arc<HookRegistry>,
    wte_t: Tensor,
}

impl std::fmt::Debug for Gpt2Runner<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gpt2Runner")
            .field("backend", &self.backend.name())
            .finish_non_exhaustive()
    }
}

impl<'a> Gpt2Runner<'a> {
    pub fn new(model: &'a Gpt2, backend: &'a dyn Backend, hooks: Arc<HookRegistry>) -> Result<Self> {
        let wte_t = transpose_2d(&model.wte)?;
        Ok(Self { model, backend, hooks, wte_t })
    }

    pub fn forward(&self, ids: &[u32]) -> Result<Tensor> {
        let cfg = &self.model.config;
        let seq = ids.len();
        let d = cfg.n_embd;
        let n_head = cfg.n_head;
        let head_dim = cfg.head_dim();

        let mut x = self.embed(ids)?;
        x = self.intercept("embed", x);

        for layer in 0..cfg.n_layer {
            let block = &self.model.blocks[layer];
            x = self.intercept(&gpt2_hooks::block_resid_pre(layer), x);

            let n1 = self.layer_norm(&x, &block.ln_1_g, &block.ln_1_b, cfg.layer_norm_epsilon)?;

            let qkv = self.linear(&n1, &block.attn_c_attn_w, &block.attn_c_attn_b)?;
            let qkv_cpu = ensure_cpu(self.backend, &qkv)?;
            let (q, k, v) = split_qkv(&qkv_cpu, d)?;
            let q4 = reshape_to_heads(&q, seq, n_head, head_dim)?;
            let k4 = reshape_to_heads(&k, seq, n_head, head_dim)?;
            let v4 = reshape_to_heads(&v, seq, n_head, head_dim)?;

            let mut attn_z =
                Tensor::from_f32(&vec![0.0; seq * n_head * head_dim], Shape::from([1, n_head, seq, head_dim]))
                    .map_err(ModelError::from)?;
            let mut pattern_buf =
                Tensor::from_f32(&vec![0.0; n_head * seq * seq], Shape::from([1, n_head, seq, seq]))
                    .map_err(ModelError::from)?;
            self.backend
                .attention(&q4, &k4, &v4, AttentionMask::Causal, &mut attn_z, Some(&mut pattern_buf))
                .map_err(ModelError::from)?;
            let _pattern = self.intercept(&gpt2_hooks::block_attn_pattern(layer), pattern_buf);
            let attn_z = self.intercept(&gpt2_hooks::block_attn_z(layer), attn_z);

            let attn_z_cpu = ensure_cpu(self.backend, &attn_z)?;
            let attn_merged = merge_heads(&attn_z_cpu, seq, n_head, head_dim)?;
            let attn_out = self.linear(&attn_merged, &block.attn_c_proj_w, &block.attn_c_proj_b)?;

            x = self.add(&x, &attn_out)?;
            x = self.intercept(&gpt2_hooks::block_resid_mid(layer), x);

            let n2 = self.layer_norm(&x, &block.ln_2_g, &block.ln_2_b, cfg.layer_norm_epsilon)?;
            let mlp_pre = self.linear(&n2, &block.mlp_c_fc_w, &block.mlp_c_fc_b)?;
            let _ = self.intercept(&format!("blocks.{layer}.mlp.pre"), mlp_pre.clone());
            let mlp_post = self.gelu(&mlp_pre)?;
            let mlp_post = self.intercept(&gpt2_hooks::block_mlp_post(layer), mlp_post);
            let mlp_out = self.linear(&mlp_post, &block.mlp_c_proj_w, &block.mlp_c_proj_b)?;

            x = self.add(&x, &mlp_out)?;
            x = self.intercept(&gpt2_hooks::block_resid_post(layer), x);
        }

        let final_ln = self.layer_norm(&x, &self.model.ln_f_g, &self.model.ln_f_b, cfg.layer_norm_epsilon)?;
        let final_ln = self.intercept("final_ln", final_ln);

        let logits = self.matmul(&final_ln, &self.wte_t)?;
        let logits = self.intercept(gpt2_hooks::UNEMBED, logits);
        Ok(logits)
    }

    pub fn last_position_logits(&self, logits: &Tensor) -> Result<Vec<f32>> {
        let cfg = &self.model.config;
        let vocab = cfg.vocab_size;
        let logits_cpu = ensure_cpu(self.backend, logits)?;
        let data = logits_cpu.as_f32().map_err(ModelError::from)?;
        let seq = data.len() / vocab;
        let off = (seq - 1) * vocab;
        Ok(data[off..off + vocab].to_vec())
    }

    fn embed(&self, ids: &[u32]) -> Result<Tensor> {
        let cfg = &self.model.config;
        let d = cfg.n_embd;
        let mut tok =
            Tensor::from_f32(&vec![0.0; ids.len() * d], Shape::from([ids.len(), d])).map_err(ModelError::from)?;
        self.backend.embed(&self.model.wte, ids, &mut tok).map_err(ModelError::from)?;
        let pos_ids: Vec<u32> = (0..ids.len() as u32).collect();
        let mut pos =
            Tensor::from_f32(&vec![0.0; ids.len() * d], Shape::from([ids.len(), d])).map_err(ModelError::from)?;
        self.backend
            .embed(&self.model.wpe, &pos_ids, &mut pos)
            .map_err(ModelError::from)?;
        self.add(&tok, &pos)
    }

    fn linear(&self, x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
        let m = x.shape().dim(0).map_err(ModelError::from)?;
        let n = w.shape().dim(1).map_err(ModelError::from)?;
        let mut y = Tensor::from_f32(&vec![0.0; m * n], Shape::from([m, n])).map_err(ModelError::from)?;
        self.backend.matmul(x, w, &mut y).map_err(ModelError::from)?;
        let y_cpu = ensure_cpu(self.backend, &y)?;
        let mut y_cpu = y_cpu;
        add_row_bias(&mut y_cpu, b)?;
        Ok(y_cpu)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let m = a.shape().dim(0).map_err(ModelError::from)?;
        let n = b.shape().dim(1).map_err(ModelError::from)?;
        let mut out = Tensor::from_f32(&vec![0.0; m * n], Shape::from([m, n])).map_err(ModelError::from)?;
        self.backend.matmul(a, b, &mut out).map_err(ModelError::from)?;
        Ok(out)
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let mut out =
            Tensor::from_f32(&vec![0.0; a.numel()], a.shape().clone()).map_err(ModelError::from)?;
        self.backend.add(a, b, &mut out).map_err(ModelError::from)?;
        Ok(out)
    }

    fn gelu(&self, x: &Tensor) -> Result<Tensor> {
        let mut out =
            Tensor::from_f32(&vec![0.0; x.numel()], x.shape().clone()).map_err(ModelError::from)?;
        self.backend.gelu(x, &mut out).map_err(ModelError::from)?;
        Ok(out)
    }

    fn layer_norm(&self, x: &Tensor, g: &Tensor, b: &Tensor, eps: f32) -> Result<Tensor> {
        let mut out =
            Tensor::from_f32(&vec![0.0; x.numel()], x.shape().clone()).map_err(ModelError::from)?;
        self.backend.layer_norm(x, g, b, eps, &mut out).map_err(ModelError::from)?;
        Ok(out)
    }

    fn publish(&self, name: &str, t: &Tensor) {
        if self.hooks.is_subscribed(name) {
            self.hooks.publish(name, t.clone());
        }
    }

    fn intercept(&self, name: &str, computed: Tensor) -> Tensor {
        if let Some(patched) = self.hooks.patch(name) {
            self.publish(name, &patched);
            patched
        } else {
            self.publish(name, &computed);
            computed
        }
    }
}

fn ensure_cpu(backend: &dyn Backend, t: &Tensor) -> Result<Tensor> {
    if matches!(t.storage(), glassbox_core::Storage::Cpu(_)) {
        return Ok(t.clone());
    }
    backend.download(t).map_err(ModelError::from)
}

fn transpose_2d(t: &Tensor) -> Result<Tensor> {
    let m = t.shape().dim(0).map_err(ModelError::from)?;
    let n = t.shape().dim(1).map_err(ModelError::from)?;
    let src = t.as_f32().map_err(ModelError::from)?;
    let mut dst = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            dst[j * m + i] = src[i * n + j];
        }
    }
    Tensor::from_f32(&dst, Shape::from([n, m])).map_err(ModelError::from)
}

fn split_qkv(qkv: &Tensor, d: usize) -> Result<(Tensor, Tensor, Tensor)> {
    let seq = qkv.shape().dim(0).map_err(ModelError::from)?;
    if qkv.shape().dim(1).map_err(ModelError::from)? != 3 * d {
        return Err(ModelError::Core(CoreError::ShapeMismatch {
            expected: Shape::from([seq, 3 * d]),
            got: qkv.shape().clone(),
        }));
    }
    let data = qkv.as_f32().map_err(ModelError::from)?;
    let mut q = vec![0.0f32; seq * d];
    let mut k = vec![0.0f32; seq * d];
    let mut v = vec![0.0f32; seq * d];
    for i in 0..seq {
        let base = i * 3 * d;
        q[i * d..(i + 1) * d].copy_from_slice(&data[base..base + d]);
        k[i * d..(i + 1) * d].copy_from_slice(&data[base + d..base + 2 * d]);
        v[i * d..(i + 1) * d].copy_from_slice(&data[base + 2 * d..base + 3 * d]);
    }
    Ok((
        Tensor::from_f32(&q, Shape::from([seq, d])).map_err(ModelError::from)?,
        Tensor::from_f32(&k, Shape::from([seq, d])).map_err(ModelError::from)?,
        Tensor::from_f32(&v, Shape::from([seq, d])).map_err(ModelError::from)?,
    ))
}

fn reshape_to_heads(x: &Tensor, seq: usize, n_head: usize, head_dim: usize) -> Result<Tensor> {
    let src = x.as_f32().map_err(ModelError::from)?;
    let mut dst = vec![0.0f32; n_head * seq * head_dim];
    for s in 0..seq {
        for h in 0..n_head {
            let src_off = s * n_head * head_dim + h * head_dim;
            let dst_off = h * seq * head_dim + s * head_dim;
            dst[dst_off..dst_off + head_dim].copy_from_slice(&src[src_off..src_off + head_dim]);
        }
    }
    Tensor::from_f32(&dst, Shape::from([1, n_head, seq, head_dim])).map_err(ModelError::from)
}

fn merge_heads(x: &Tensor, seq: usize, n_head: usize, head_dim: usize) -> Result<Tensor> {
    let src = x.as_f32().map_err(ModelError::from)?;
    let d = n_head * head_dim;
    let mut dst = vec![0.0f32; seq * d];
    for h in 0..n_head {
        for s in 0..seq {
            let src_off = h * seq * head_dim + s * head_dim;
            let dst_off = s * d + h * head_dim;
            dst[dst_off..dst_off + head_dim].copy_from_slice(&src[src_off..src_off + head_dim]);
        }
    }
    Tensor::from_f32(&dst, Shape::from([seq, d])).map_err(ModelError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::gpt2::{Gpt2, Gpt2Block};
    use glassbox_runtime::CpuBackend;

    fn tiny_tensor(shape: Shape, fill: f32) -> Tensor {
        let n = shape.numel();
        Tensor::from_f32(&vec![fill; n], shape).unwrap()
    }

    fn deterministic_tensor(shape: Shape, seed: f32) -> Tensor {
        let n = shape.numel();
        let data: Vec<f32> = (0..n).map(|i| ((i as f32 + seed) * 0.07).sin() * 0.05).collect();
        Tensor::from_f32(&data, shape).unwrap()
    }

    fn synth_gpt2() -> Gpt2 {
        let cfg = ModelConfig {
            architecture: "gpt2".into(),
            vocab_size: 16,
            n_positions: 8,
            n_embd: 8,
            n_layer: 2,
            n_head: 2,
            layer_norm_epsilon: 1e-5,
        };
        let d = cfg.n_embd;
        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for layer in 0..cfg.n_layer {
            let s = layer as f32 + 1.0;
            blocks.push(Gpt2Block {
                ln_1_g: tiny_tensor(Shape::from([d]), 1.0),
                ln_1_b: tiny_tensor(Shape::from([d]), 0.0),
                attn_c_attn_w: deterministic_tensor(Shape::from([d, 3 * d]), s + 0.1),
                attn_c_attn_b: tiny_tensor(Shape::from([3 * d]), 0.0),
                attn_c_proj_w: deterministic_tensor(Shape::from([d, d]), s + 0.2),
                attn_c_proj_b: tiny_tensor(Shape::from([d]), 0.0),
                ln_2_g: tiny_tensor(Shape::from([d]), 1.0),
                ln_2_b: tiny_tensor(Shape::from([d]), 0.0),
                mlp_c_fc_w: deterministic_tensor(Shape::from([d, 4 * d]), s + 0.3),
                mlp_c_fc_b: tiny_tensor(Shape::from([4 * d]), 0.0),
                mlp_c_proj_w: deterministic_tensor(Shape::from([4 * d, d]), s + 0.4),
                mlp_c_proj_b: tiny_tensor(Shape::from([d]), 0.0),
            });
        }
        Gpt2 {
            config: cfg.clone(),
            wte: deterministic_tensor(Shape::from([cfg.vocab_size, d]), 0.5),
            wpe: deterministic_tensor(Shape::from([cfg.n_positions, d]), 1.5),
            blocks,
            ln_f_g: tiny_tensor(Shape::from([d]), 1.0),
            ln_f_b: tiny_tensor(Shape::from([d]), 0.0),
        }
    }

    #[test]
    fn forward_produces_finite_logits() {
        let model = synth_gpt2();
        let runner = Gpt2Runner::new(&model, &CpuBackend, HookRegistry::new()).unwrap();
        let logits = runner.forward(&[1, 2, 3, 4]).unwrap();
        assert_eq!(logits.shape().dims(), &[4, 16]);
        for &v in logits.as_f32().unwrap() {
            assert!(v.is_finite(), "logit was not finite: {v}");
        }
    }

    #[test]
    fn last_position_logits_is_correct_slice() {
        let model = synth_gpt2();
        let runner = Gpt2Runner::new(&model, &CpuBackend, HookRegistry::new()).unwrap();
        let logits = runner.forward(&[5, 6, 7]).unwrap();
        let last = runner.last_position_logits(&logits).unwrap();
        assert_eq!(last.len(), 16);
        let full = logits.as_f32().unwrap();
        assert_eq!(&last[..], &full[2 * 16..3 * 16]);
    }

    #[test]
    fn hooks_capture_attention_pattern() {
        let model = synth_gpt2();
        let hooks = HookRegistry::new();
        hooks.subscribe("blocks.0.attn.pattern");
        hooks.subscribe("blocks.1.resid_post");
        let runner = Gpt2Runner::new(&model, &CpuBackend, Arc::clone(&hooks)).unwrap();
        let _ = runner.forward(&[1, 2, 3]).unwrap();
        let snap = hooks.snapshot();
        assert!(snap.contains_key("blocks.0.attn.pattern"));
        assert!(snap.contains_key("blocks.1.resid_post"));

        let pattern = snap.get("blocks.0.attn.pattern").unwrap();
        assert_eq!(pattern.shape().dims(), &[1, 2, 3, 3]);
        let p_data = pattern.as_f32().unwrap();
        for h in 0..2 {
            for q in 0..3 {
                let off = (h * 3 + q) * 3;
                let row_sum: f32 = p_data[off..off + q + 1].iter().sum();
                assert!((row_sum - 1.0).abs() < 1e-3, "row {q} of head {h} sums to {row_sum}");
            }
        }
    }
}

fn add_row_bias(y: &mut Tensor, bias: &Tensor) -> Result<()> {
    let m = y.shape().dim(0).map_err(ModelError::from)?;
    let n = y.shape().dim(1).map_err(ModelError::from)?;
    if bias.numel() != n {
        return Err(ModelError::Core(CoreError::ShapeMismatch {
            expected: Shape::from([n]),
            got: bias.shape().clone(),
        }));
    }
    let y_data = y.as_f32().map_err(ModelError::from)?;
    let b_data = bias.as_f32().map_err(ModelError::from)?;
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = y_data[i * n + j] + b_data[j];
        }
    }
    *y = Tensor::from_f32(&out, Shape::from([m, n])).map_err(ModelError::from)?;
    Ok(())
}
