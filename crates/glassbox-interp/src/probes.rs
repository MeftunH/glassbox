use glassbox_core::Tensor;

use crate::hooks::HookRegistry;

#[derive(Debug, Clone)]
pub struct NeuronProbe {
    pub layer: usize,
    pub neuron: usize,
}

impl NeuronProbe {
    pub fn new(layer: usize, neuron: usize) -> Self {
        Self { layer, neuron }
    }
}

#[derive(Debug, Clone)]
pub struct NeuronProbeResult {
    pub layer: usize,
    pub neuron: usize,
    pub activations: Vec<f32>,
}

impl NeuronProbeResult {
    pub fn from_post_act(probe: &NeuronProbe, post_act: &Tensor, dim: usize) -> Option<Self> {
        let data = post_act.as_f32().ok()?;
        let positions = data.len() / dim;
        let mut acts = Vec::with_capacity(positions);
        for p in 0..positions {
            acts.push(data[p * dim + probe.neuron]);
        }
        Some(Self { layer: probe.layer, neuron: probe.neuron, activations: acts })
    }
}

pub fn top_k_neurons(post_act: &Tensor, layer: usize, dim: usize, k: usize) -> Vec<(usize, f32)> {
    let Ok(data) = post_act.as_f32() else { return Vec::new() };
    let positions = data.len() / dim;
    let mut summed = vec![0.0f32; dim];
    for p in 0..positions {
        for n in 0..dim {
            summed[n] += data[p * dim + n].max(0.0);
        }
    }
    let _ = layer;
    let mut indexed: Vec<(usize, f32)> = summed.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
}

pub fn snapshot_into_results(
    reg: &HookRegistry,
    probe: &NeuronProbe,
    hook: &str,
    dim: usize,
) -> Option<NeuronProbeResult> {
    let captured = reg.snapshot();
    let post_act = captured.get(hook)?;
    NeuronProbeResult::from_post_act(probe, post_act, dim)
}
