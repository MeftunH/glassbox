use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub u64);

#[derive(Debug, Clone)]
pub enum Storage {
    Cpu(Arc<[u8]>),
    Gpu(BufferId),
}

impl Storage {
    pub fn cpu(bytes: impl Into<Arc<[u8]>>) -> Self {
        Self::Cpu(bytes.into())
    }

    pub fn backend_name(&self) -> &'static str {
        match self {
            Self::Cpu(_) => "cpu",
            Self::Gpu(_) => "gpu",
        }
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu(_))
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Gpu(_))
    }

    pub fn as_cpu(&self) -> Option<&[u8]> {
        match self {
            Self::Cpu(arc) => Some(arc),
            Self::Gpu(_) => None,
        }
    }

    pub fn gpu_id(&self) -> Option<BufferId> {
        match self {
            Self::Gpu(id) => Some(*id),
            Self::Cpu(_) => None,
        }
    }
}
