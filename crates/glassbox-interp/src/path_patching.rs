use std::sync::Arc;

use ahash::AHashMap;
use glassbox_core::Tensor;
use glassbox_models::{Gpt2, Gpt2Runner};
use glassbox_runtime::{Backend, HookRegistry};

use crate::error::{InterpError, Result};

#[derive(Debug, Clone)]
pub struct PathPatchSpec {
    pub clean_ids: Vec<u32>,
    pub corrupt_ids: Vec<u32>,
    pub sender_hook: String,
    pub receiver_hooks: Vec<String>,
    pub target_token: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct PathPatchResult {
    pub clean_logit: f32,
    pub corrupt_logit: f32,
    pub patched_logit: f32,
    pub recovery: f32,
}

pub fn run_path_patch(
    model: &Gpt2,
    backend: &dyn Backend,
    spec: &PathPatchSpec,
) -> Result<PathPatchResult> {
    let clean_cache = capture_all(model, backend, &spec.clean_ids)?;
    let corrupt_cache = capture_all(model, backend, &spec.corrupt_ids)?;

    let target = match spec.target_token {
        Some(t) => t as usize,
        None => {
            let clean_logits = clean_cache
                .get(glassbox_models::gpt2::hooks::UNEMBED)
                .ok_or_else(|| InterpError::MissingHook(glassbox_models::gpt2::hooks::UNEMBED.into()))?;
            argmax_last_position(clean_logits, model.config.vocab_size)?
        }
    };

    let clean_logit = read_logit(&clean_cache, model.config.vocab_size, target)?;
    let corrupt_logit = read_logit(&corrupt_cache, model.config.vocab_size, target)?;

    let hooks = HookRegistry::new();
    let sender_clean = clean_cache
        .get(&spec.sender_hook)
        .ok_or_else(|| InterpError::MissingHook(spec.sender_hook.clone()))?
        .clone();
    hooks.install_patch(spec.sender_hook.clone(), sender_clean);
    for receiver in &spec.receiver_hooks {
        let v = corrupt_cache
            .get(receiver)
            .ok_or_else(|| InterpError::MissingHook(receiver.clone()))?
            .clone();
        hooks.install_patch(receiver.clone(), v);
    }
    hooks.subscribe(glassbox_models::gpt2::hooks::UNEMBED);

    let runner = Gpt2Runner::new(model, backend, Arc::clone(&hooks))
        .map_err(|e| InterpError::Forward(e.to_string()))?;
    runner.forward(&spec.corrupt_ids).map_err(|e| InterpError::Forward(e.to_string()))?;
    let snap = hooks.snapshot();
    let patched_logit = read_logit(&snap, model.config.vocab_size, target)?;

    let denom = clean_logit - corrupt_logit;
    let recovery = if denom.abs() < f32::EPSILON {
        0.0
    } else {
        (patched_logit - corrupt_logit) / denom
    };

    Ok(PathPatchResult { clean_logit, corrupt_logit, patched_logit, recovery })
}

fn capture_all(
    model: &Gpt2,
    backend: &dyn Backend,
    ids: &[u32],
) -> Result<AHashMap<String, Tensor>> {
    let hooks = HookRegistry::new();
    let names = enumerate_hooks(model);
    for n in &names {
        hooks.subscribe(n.clone());
    }
    let runner = Gpt2Runner::new(model, backend, Arc::clone(&hooks))
        .map_err(|e| InterpError::Forward(e.to_string()))?;
    runner.forward(ids).map_err(|e| InterpError::Forward(e.to_string()))?;
    Ok(hooks.snapshot())
}

fn enumerate_hooks(model: &Gpt2) -> Vec<String> {
    use glassbox_models::gpt2::hooks as h;
    let mut out = vec!["embed".to_string(), "final_ln".to_string(), h::UNEMBED.to_string()];
    for l in 0..model.config.n_layer {
        out.push(h::block_resid_pre(l));
        out.push(h::block_resid_mid(l));
        out.push(h::block_resid_post(l));
        out.push(h::block_attn_pattern(l));
        out.push(h::block_attn_z(l));
        out.push(h::block_mlp_post(l));
    }
    out
}

fn argmax_last_position(logits: &Tensor, vocab: usize) -> Result<usize> {
    let data = logits.as_f32().map_err(|e| InterpError::Forward(e.to_string()))?;
    let seq = data.len() / vocab;
    if seq == 0 {
        return Err(InterpError::Forward("empty logits".into()));
    }
    let off = (seq - 1) * vocab;
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for i in 0..vocab {
        let v = data[off + i];
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    Ok(best_i)
}

fn read_logit(
    cache: &AHashMap<String, Tensor>,
    vocab: usize,
    target: usize,
) -> Result<f32> {
    let logits = cache
        .get(glassbox_models::gpt2::hooks::UNEMBED)
        .ok_or_else(|| InterpError::MissingHook(glassbox_models::gpt2::hooks::UNEMBED.into()))?;
    let data = logits.as_f32().map_err(|e| InterpError::Forward(e.to_string()))?;
    let seq = data.len() / vocab;
    if seq == 0 || target >= vocab {
        return Err(InterpError::Forward(format!("bad logit slice seq={seq} target={target}")));
    }
    Ok(data[(seq - 1) * vocab + target])
}

#[cfg(test)]
mod tests {
    use super::*;
    use glassbox_core::Shape;
    use glassbox_models::config::ModelConfig;
    use glassbox_models::gpt2::Gpt2Block;
    use glassbox_runtime::CpuBackend;

    fn tiny(shape: Shape, fill: f32) -> Tensor {
        Tensor::from_f32(&vec![fill; shape.numel()], shape).unwrap()
    }

    fn det(shape: Shape, seed: f32) -> Tensor {
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
                ln_1_g: tiny(Shape::from([d]), 1.0),
                ln_1_b: tiny(Shape::from([d]), 0.0),
                attn_c_attn_w: det(Shape::from([d, 3 * d]), s + 0.1),
                attn_c_attn_b: tiny(Shape::from([3 * d]), 0.0),
                attn_c_proj_w: det(Shape::from([d, d]), s + 0.2),
                attn_c_proj_b: tiny(Shape::from([d]), 0.0),
                ln_2_g: tiny(Shape::from([d]), 1.0),
                ln_2_b: tiny(Shape::from([d]), 0.0),
                mlp_c_fc_w: det(Shape::from([d, 4 * d]), s + 0.3),
                mlp_c_fc_b: tiny(Shape::from([4 * d]), 0.0),
                mlp_c_proj_w: det(Shape::from([4 * d, d]), s + 0.4),
                mlp_c_proj_b: tiny(Shape::from([d]), 0.0),
            });
        }
        Gpt2 {
            config: cfg.clone(),
            wte: det(Shape::from([cfg.vocab_size, d]), 0.5),
            wpe: det(Shape::from([cfg.n_positions, d]), 1.5),
            blocks,
            ln_f_g: tiny(Shape::from([d]), 1.0),
            ln_f_b: tiny(Shape::from([d]), 0.0),
        }
    }

    #[test]
    fn path_patch_runs_and_returns_finite_recovery() {
        let model = synth_gpt2();
        let spec = PathPatchSpec {
            clean_ids: vec![1, 2, 3, 4],
            corrupt_ids: vec![5, 6, 7, 8],
            sender_hook: glassbox_models::gpt2::hooks::block_resid_post(0),
            receiver_hooks: vec![glassbox_models::gpt2::hooks::block_resid_post(1)],
            target_token: None,
        };
        let result = run_path_patch(&model, &CpuBackend, &spec).unwrap();
        assert!(result.clean_logit.is_finite());
        assert!(result.corrupt_logit.is_finite());
        assert!(result.patched_logit.is_finite());
        assert!(result.recovery.is_finite());
    }

    #[test]
    fn full_patch_recovers_clean_completely() {
        let model = synth_gpt2();
        let spec = PathPatchSpec {
            clean_ids: vec![1, 2, 3, 4],
            corrupt_ids: vec![1, 2, 3, 4],
            sender_hook: glassbox_models::gpt2::hooks::block_resid_post(0),
            receiver_hooks: vec![],
            target_token: Some(7),
        };
        let result = run_path_patch(&model, &CpuBackend, &spec).unwrap();
        assert!((result.patched_logit - result.clean_logit).abs() < 1e-3);
    }
}
