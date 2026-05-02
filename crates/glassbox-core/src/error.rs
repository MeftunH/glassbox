use crate::{DType, Shape};

pub type Result<T, E = CoreError> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Shape, got: Shape },

    #[error("rank mismatch: expected rank {expected}, got rank {got}")]
    RankMismatch { expected: usize, got: usize },

    #[error("dtype mismatch: expected {expected:?}, got {got:?}")]
    DTypeMismatch { expected: DType, got: DType },

    #[error("non-contiguous tensor where contiguous required")]
    NotContiguous,

    #[error("axis {axis} out of bounds for rank {rank}")]
    AxisOutOfBounds { axis: usize, rank: usize },

    #[error("storage on wrong backend: expected {expected}, got {got}")]
    WrongBackend { expected: &'static str, got: &'static str },

    #[error("byte length {got} not aligned for dtype {dtype:?} (element size {elem})")]
    Misaligned { got: usize, dtype: DType, elem: usize },

    #[error("element count {got} does not match shape {shape:?} ({expected} elements)")]
    ElementCountMismatch {
        got: usize,
        expected: usize,
        shape: Shape,
    },
}
