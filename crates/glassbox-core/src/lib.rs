#![doc = include_str!("../README.md")]

pub mod dtype;
pub mod error;
pub mod shape;
pub mod storage;
pub mod tensor;

pub use dtype::DType;
pub use error::{CoreError, Result};
pub use shape::{Shape, Stride};
pub use storage::{BufferId, Storage};
pub use tensor::Tensor;
