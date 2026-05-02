use glassbox_core::Tensor;
use glassbox_runtime::{Backend, Result};

#[derive(Debug, Clone, Copy)]
pub struct DirectLogitAttribution {
    pub layer: usize,
    pub position: Option<usize>,
}

impl DirectLogitAttribution {
    pub fn at(layer: usize) -> Self {
        Self { layer, position: None }
    }

    pub fn at_position(layer: usize, position: usize) -> Self {
        Self { layer, position: Some(position) }
    }

    pub fn project(
        &self,
        backend: &dyn Backend,
        residual: &Tensor,
        unembed_w: &Tensor,
        out: &mut Tensor,
    ) -> Result<()> {
        backend.matmul(residual, unembed_w, out)
    }
}
