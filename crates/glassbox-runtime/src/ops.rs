use glassbox_core::Tensor;

use crate::backend::{AttentionMask, Backend};
use crate::error::Result;

pub fn matmul(backend: &dyn Backend, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    backend.matmul(a, b, out)
}

pub fn add(backend: &dyn Backend, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    backend.add(a, b, out)
}

pub fn layer_norm(
    backend: &dyn Backend,
    x: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    eps: f32,
    out: &mut Tensor,
) -> Result<()> {
    backend.layer_norm(x, gamma, beta, eps, out)
}

pub fn gelu(backend: &dyn Backend, x: &Tensor, out: &mut Tensor) -> Result<()> {
    backend.gelu(x, out)
}

pub fn softmax(backend: &dyn Backend, x: &Tensor, axis: isize, out: &mut Tensor) -> Result<()> {
    backend.softmax(x, axis, out)
}

#[allow(clippy::too_many_arguments)]
pub fn attention(
    backend: &dyn Backend,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: AttentionMask,
    out: &mut Tensor,
    pattern_out: Option<&mut Tensor>,
) -> Result<()> {
    backend.attention(q, k, v, mask, out, pattern_out)
}
