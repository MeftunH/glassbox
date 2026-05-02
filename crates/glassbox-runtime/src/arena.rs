use std::sync::Mutex;

use glassbox_core::{BufferId, DType, Shape, Storage, Tensor};

use crate::error::{Result, RuntimeError};

#[derive(Debug)]
pub struct CpuArena {
    bump: Mutex<Vec<u8>>,
    capacity: usize,
}

impl CpuArena {
    pub fn with_capacity(capacity: usize) -> Self {
        Self { bump: Mutex::new(Vec::with_capacity(capacity)), capacity }
    }

    pub fn reset(&self) {
        if let Ok(mut buf) = self.bump.lock() {
            buf.clear();
        }
    }

    pub fn alloc(&self, shape: Shape, dtype: DType) -> Result<Tensor> {
        let bytes_needed = shape.numel() * dtype.size();
        let mut buf = self
            .bump
            .lock()
            .map_err(|_| RuntimeError::ArenaOom { requested: bytes_needed, capacity: self.capacity })?;
        if buf.len() + bytes_needed > self.capacity {
            return Err(RuntimeError::ArenaOom {
                requested: bytes_needed,
                capacity: self.capacity,
            });
        }
        let start = buf.len();
        buf.resize(start + bytes_needed, 0);
        let bytes: std::sync::Arc<[u8]> = std::sync::Arc::from(buf[start..start + bytes_needed].to_vec());
        Ok(Tensor::new(Storage::cpu(bytes), shape, dtype))
    }
}

#[derive(Debug, Default)]
pub struct GpuArena {
    next_id: std::sync::atomic::AtomicU64,
}

impl GpuArena {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn alloc(&self, shape: Shape, dtype: DType) -> Tensor {
        let id = BufferId(self.next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
        Tensor::from_gpu(id, shape, dtype)
    }
}
