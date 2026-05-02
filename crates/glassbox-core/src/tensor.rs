use std::sync::Arc;

use bytemuck::Pod;
use half::f16;

use crate::{
    dtype::DType,
    error::{CoreError, Result},
    shape::{Shape, Stride},
    storage::{BufferId, Storage},
};

#[derive(Debug, Clone)]
pub struct Tensor {
    storage: Storage,
    shape: Shape,
    stride: Stride,
    dtype: DType,
    offset: usize,
}

impl Tensor {
    pub fn new(storage: Storage, shape: Shape, dtype: DType) -> Self {
        let stride = shape.contiguous_strides();
        Self { storage, shape, stride, dtype, offset: 0 }
    }

    pub fn from_slice<T: Pod>(values: &[T], shape: Shape, dtype: DType) -> Result<Self> {
        let elem = dtype.size();
        if std::mem::size_of::<T>() != elem {
            return Err(CoreError::Misaligned {
                got: std::mem::size_of::<T>(),
                dtype,
                elem,
            });
        }
        if values.len() != shape.numel() {
            return Err(CoreError::ElementCountMismatch {
                got: values.len(),
                expected: shape.numel(),
                shape: shape.clone(),
            });
        }
        let bytes: Vec<u8> = bytemuck::cast_slice(values).to_vec();
        Ok(Self::new(Storage::cpu(Arc::<[u8]>::from(bytes)), shape, dtype))
    }

    pub fn from_f32(values: &[f32], shape: Shape) -> Result<Self> {
        Self::from_slice(values, shape, DType::F32)
    }

    pub fn from_f16(values: &[f16], shape: Shape) -> Result<Self> {
        Self::from_slice(values, shape, DType::F16)
    }

    pub fn from_gpu(id: BufferId, shape: Shape, dtype: DType) -> Self {
        Self::new(Storage::Gpu(id), shape, dtype)
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn stride(&self) -> &Stride {
        &self.stride
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn is_contiguous(&self) -> bool {
        self.stride.is_contiguous_for(&self.shape)
    }

    pub fn as_cpu_bytes(&self) -> Result<&[u8]> {
        let bytes = self
            .storage
            .as_cpu()
            .ok_or(CoreError::WrongBackend { expected: "cpu", got: self.storage.backend_name() })?;
        let elem = self.dtype.size();
        let start = self.offset * elem;
        let end = start + self.numel() * elem;
        Ok(&bytes[start..end])
    }

    pub fn as_f32(&self) -> Result<&[f32]> {
        if self.dtype != DType::F32 {
            return Err(CoreError::DTypeMismatch { expected: DType::F32, got: self.dtype });
        }
        if !self.is_contiguous() {
            return Err(CoreError::NotContiguous);
        }
        Ok(bytemuck::cast_slice(self.as_cpu_bytes()?))
    }

    pub fn as_f16(&self) -> Result<&[f16]> {
        if self.dtype != DType::F16 {
            return Err(CoreError::DTypeMismatch { expected: DType::F16, got: self.dtype });
        }
        if !self.is_contiguous() {
            return Err(CoreError::NotContiguous);
        }
        Ok(bytemuck::cast_slice(self.as_cpu_bytes()?))
    }

    pub fn reshape(&self, new_shape: impl Into<Shape>) -> Result<Self> {
        let new_shape = new_shape.into();
        if new_shape.numel() != self.numel() {
            return Err(CoreError::ElementCountMismatch {
                got: new_shape.numel(),
                expected: self.numel(),
                shape: new_shape,
            });
        }
        if !self.is_contiguous() {
            return Err(CoreError::NotContiguous);
        }
        let stride = new_shape.contiguous_strides();
        Ok(Self {
            storage: self.storage.clone(),
            shape: new_shape,
            stride,
            dtype: self.dtype,
            offset: self.offset,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_f32_slice() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from([2, 3])).unwrap();
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.numel(), 6);
        assert!(t.is_contiguous());
        assert_eq!(t.as_f32().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn reshape_preserves_data() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from([2, 3])).unwrap();
        let r = t.reshape([3, 2]).unwrap();
        assert_eq!(r.shape().dims(), &[3, 2]);
        assert_eq!(r.as_f32().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn reshape_rejects_wrong_size() {
        let t = Tensor::from_f32(&[1.0; 6], Shape::from([2, 3])).unwrap();
        assert!(t.reshape([3, 3]).is_err());
    }

    #[test]
    fn dtype_mismatch_on_view() {
        let t = Tensor::from_f32(&[1.0; 4], Shape::from([4])).unwrap();
        assert!(t.as_f16().is_err());
    }
}
