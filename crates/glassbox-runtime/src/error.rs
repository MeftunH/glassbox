use glassbox_core::CoreError;

pub type Result<T, E = RuntimeError> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("core: {0}")]
    Core(#[from] CoreError),

    #[error("backend feature `{0}` not compiled in")]
    BackendUnavailable(&'static str),

    #[error("op `{op}` does not support dtype {dtype}")]
    UnsupportedDType { op: &'static str, dtype: &'static str },

    #[error("op `{op}` requires rank-{expected} input, got rank {got}")]
    BadRank { op: &'static str, expected: usize, got: usize },

    #[error("matmul shape mismatch: a={a:?}, b={b:?}")]
    MatmulShapeMismatch { a: Vec<usize>, b: Vec<usize> },

    #[error("attention shape mismatch: q={q:?}, k={k:?}, v={v:?}")]
    AttentionShapeMismatch {
        q: Vec<usize>,
        k: Vec<usize>,
        v: Vec<usize>,
    },

    #[error("arena out of memory: requested {requested} bytes, capacity {capacity}")]
    ArenaOom { requested: usize, capacity: usize },

    #[error("wgpu: {0}")]
    Wgpu(String),
}
