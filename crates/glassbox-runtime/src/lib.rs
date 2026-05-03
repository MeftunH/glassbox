#![doc = include_str!("../README.md")]

pub mod arena;
pub mod async_backend;
pub mod backend;
pub mod error;
pub mod hooks;
pub mod ops;
pub mod sampling;

#[cfg(feature = "cpu")]
pub mod cpu;

#[cfg(feature = "wgpu")]
pub mod wgpu_backend;

pub use async_backend::AsyncBackend;
pub use backend::{AttentionMask, Backend};
pub use error::{Result, RuntimeError};
pub use hooks::{HookName, HookRegistry};
pub use sampling::{SamplingConfig, Sampler};

#[cfg(feature = "cpu")]
pub use cpu::CpuBackend;
