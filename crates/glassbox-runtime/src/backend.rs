use glassbox_core::Tensor;

use crate::error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionMask {
    None,
    Causal,
}

pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;

    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()>;

    fn add(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()>;

    fn layer_norm(
        &self,
        x: &Tensor,
        gamma: &Tensor,
        beta: &Tensor,
        eps: f32,
        out: &mut Tensor,
    ) -> Result<()>;

    fn gelu(&self, x: &Tensor, out: &mut Tensor) -> Result<()>;

    fn softmax(&self, x: &Tensor, axis: isize, out: &mut Tensor) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: AttentionMask,
        out: &mut Tensor,
        pattern_out: Option<&mut Tensor>,
    ) -> Result<()>;

    fn embed(&self, table: &Tensor, ids: &[u32], out: &mut Tensor) -> Result<()>;
}
