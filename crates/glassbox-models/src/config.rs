use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub architecture: String,
    pub vocab_size: usize,
    pub n_positions: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub layer_norm_epsilon: f32,
}

impl ModelConfig {
    pub fn gpt2_small() -> Self {
        Self {
            architecture: "gpt2".into(),
            vocab_size: 50_257,
            n_positions: 1_024,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            layer_norm_epsilon: 1e-5,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt2_small_shape() {
        let c = ModelConfig::gpt2_small();
        assert_eq!(c.head_dim(), 64);
        assert_eq!(c.n_embd, c.n_head * c.head_dim());
    }
}
