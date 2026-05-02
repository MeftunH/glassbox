use std::sync::Arc;

use glassbox_core::Tensor;
use wgpu::util::DeviceExt;

use crate::backend::{AttentionMask, Backend};
use crate::error::{Result, RuntimeError};

pub const SHADER_MATMUL: &str = include_str!("../../../shaders/matmul.wgsl");
pub const SHADER_SOFTMAX: &str = include_str!("../../../shaders/softmax.wgsl");
pub const SHADER_LAYERNORM: &str = include_str!("../../../shaders/layernorm.wgsl");
pub const SHADER_GELU: &str = include_str!("../../../shaders/gelu.wgsl");
pub const SHADER_ATTENTION: &str = include_str!("../../../shaders/attention.wgsl");

#[derive(Debug)]
pub struct WgpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

impl WgpuBackend {
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| RuntimeError::Wgpu("no adapter".into()))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("glassbox"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| RuntimeError::Wgpu(format!("{e}")))?;

        let _ = SHADER_MATMUL;
        let _ = SHADER_SOFTMAX;
        let _ = SHADER_LAYERNORM;
        let _ = SHADER_GELU;
        let _ = SHADER_ATTENTION;
        let _ = wgpu::util::DeviceExt::create_buffer_init;

        Ok(Self { device: Arc::new(device), queue: Arc::new(queue) })
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

impl Backend for WgpuBackend {
    fn name(&self) -> &'static str {
        "wgpu"
    }

    fn matmul(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> Result<()> {
        Err(RuntimeError::Wgpu("matmul not yet wired through dispatcher".into()))
    }
    fn add(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> Result<()> {
        Err(RuntimeError::Wgpu("add not yet wired".into()))
    }
    fn layer_norm(
        &self,
        _x: &Tensor,
        _gamma: &Tensor,
        _beta: &Tensor,
        _eps: f32,
        _out: &mut Tensor,
    ) -> Result<()> {
        Err(RuntimeError::Wgpu("layer_norm not yet wired".into()))
    }
    fn gelu(&self, _x: &Tensor, _out: &mut Tensor) -> Result<()> {
        Err(RuntimeError::Wgpu("gelu not yet wired".into()))
    }
    fn softmax(&self, _x: &Tensor, _axis: isize, _out: &mut Tensor) -> Result<()> {
        Err(RuntimeError::Wgpu("softmax not yet wired".into()))
    }
    fn attention(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _mask: AttentionMask,
        _out: &mut Tensor,
        _pattern_out: Option<&mut Tensor>,
    ) -> Result<()> {
        Err(RuntimeError::Wgpu("attention not yet wired".into()))
    }
    fn embed(&self, _table: &Tensor, _ids: &[u32], _out: &mut Tensor) -> Result<()> {
        Err(RuntimeError::Wgpu("embed not yet wired".into()))
    }
}
