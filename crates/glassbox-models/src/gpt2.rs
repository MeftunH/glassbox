use glassbox_core::Tensor;

use crate::config::ModelConfig;
use crate::error::{ModelError, Result};
use crate::glx::GlxFile;

#[derive(Debug)]
pub struct Gpt2Block {
    pub ln_1_g: Tensor,
    pub ln_1_b: Tensor,
    pub attn_c_attn_w: Tensor,
    pub attn_c_attn_b: Tensor,
    pub attn_c_proj_w: Tensor,
    pub attn_c_proj_b: Tensor,
    pub ln_2_g: Tensor,
    pub ln_2_b: Tensor,
    pub mlp_c_fc_w: Tensor,
    pub mlp_c_fc_b: Tensor,
    pub mlp_c_proj_w: Tensor,
    pub mlp_c_proj_b: Tensor,
}

#[derive(Debug)]
pub struct Gpt2 {
    pub config: ModelConfig,
    pub wte: Tensor,
    pub wpe: Tensor,
    pub blocks: Vec<Gpt2Block>,
    pub ln_f_g: Tensor,
    pub ln_f_b: Tensor,
}

impl Gpt2 {
    pub fn from_glx(file: &GlxFile) -> Result<Self> {
        let config: ModelConfig = serde_json::from_value(file.header.config.clone())
            .map_err(|e| ModelError::BadGlx(format!("config: {e}")))?;
        if config.architecture != "gpt2" {
            return Err(ModelError::UnknownArchitecture(config.architecture));
        }

        let wte = file.tensor("wte")?;
        let wpe = file.tensor("wpe")?;
        let ln_f_g = file.tensor("ln_f.g")?;
        let ln_f_b = file.tensor("ln_f.b")?;

        let mut blocks = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            blocks.push(Gpt2Block {
                ln_1_g: file.tensor(&format!("h.{i}.ln_1.g"))?,
                ln_1_b: file.tensor(&format!("h.{i}.ln_1.b"))?,
                attn_c_attn_w: file.tensor(&format!("h.{i}.attn.c_attn.w"))?,
                attn_c_attn_b: file.tensor(&format!("h.{i}.attn.c_attn.b"))?,
                attn_c_proj_w: file.tensor(&format!("h.{i}.attn.c_proj.w"))?,
                attn_c_proj_b: file.tensor(&format!("h.{i}.attn.c_proj.b"))?,
                ln_2_g: file.tensor(&format!("h.{i}.ln_2.g"))?,
                ln_2_b: file.tensor(&format!("h.{i}.ln_2.b"))?,
                mlp_c_fc_w: file.tensor(&format!("h.{i}.mlp.c_fc.w"))?,
                mlp_c_fc_b: file.tensor(&format!("h.{i}.mlp.c_fc.b"))?,
                mlp_c_proj_w: file.tensor(&format!("h.{i}.mlp.c_proj.w"))?,
                mlp_c_proj_b: file.tensor(&format!("h.{i}.mlp.c_proj.b"))?,
            });
        }

        Ok(Self { config, wte, wpe, blocks, ln_f_g, ln_f_b })
    }

    pub fn parameter_count(&self) -> usize {
        let mut count = self.wte.numel() + self.wpe.numel();
        count += self.ln_f_g.numel() + self.ln_f_b.numel();
        for block in &self.blocks {
            count += block.ln_1_g.numel()
                + block.ln_1_b.numel()
                + block.attn_c_attn_w.numel()
                + block.attn_c_attn_b.numel()
                + block.attn_c_proj_w.numel()
                + block.attn_c_proj_b.numel()
                + block.ln_2_g.numel()
                + block.ln_2_b.numel()
                + block.mlp_c_fc_w.numel()
                + block.mlp_c_fc_b.numel()
                + block.mlp_c_proj_w.numel()
                + block.mlp_c_proj_b.numel();
        }
        count
    }
}

pub mod hooks {
    pub fn block_resid_pre(layer: usize) -> String {
        format!("blocks.{layer}.resid_pre")
    }
    pub fn block_resid_mid(layer: usize) -> String {
        format!("blocks.{layer}.resid_mid")
    }
    pub fn block_resid_post(layer: usize) -> String {
        format!("blocks.{layer}.resid_post")
    }
    pub fn block_attn_pattern(layer: usize) -> String {
        format!("blocks.{layer}.attn.pattern")
    }
    pub fn block_attn_z(layer: usize) -> String {
        format!("blocks.{layer}.attn.z")
    }
    pub fn block_mlp_pre(layer: usize) -> String {
        format!("blocks.{layer}.mlp.pre")
    }
    pub fn block_mlp_post(layer: usize) -> String {
        format!("blocks.{layer}.mlp.post")
    }
    pub const UNEMBED: &str = "unembed";

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn names_are_stable() {
            assert_eq!(block_attn_pattern(7), "blocks.7.attn.pattern");
        }
    }
}
