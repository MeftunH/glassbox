use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

use ahash::AHashMap;
use bytemuck::{cast_slice, Pod};
use glassbox_core::{BufferId, CoreError, DType, Shape, Storage, Tensor};
use wgpu::util::DeviceExt;

use crate::backend::{AttentionMask, Backend};
use crate::error::{Result, RuntimeError};

pub const SHADER_MATMUL: &str = include_str!("../../../shaders/matmul.wgsl");
pub const SHADER_SOFTMAX: &str = include_str!("../../../shaders/softmax.wgsl");
pub const SHADER_LAYERNORM: &str = include_str!("../../../shaders/layernorm.wgsl");
pub const SHADER_GELU: &str = include_str!("../../../shaders/gelu.wgsl");
pub const SHADER_ATTENTION: &str = include_str!("../../../shaders/attention.wgsl");
pub const SHADER_ADD: &str = include_str!("../../../shaders/add.wgsl");
pub const SHADER_EMBED: &str = include_str!("../../../shaders/embed.wgsl");

#[derive(Debug)]
pub struct WgpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipelines: Mutex<AHashMap<&'static str, Arc<wgpu::ComputePipeline>>>,
    buffers: Mutex<AHashMap<BufferId, Arc<wgpu::Buffer>>>,
    next_id: AtomicU64,
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

        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_compute_invocations_per_workgroup = 256;
        limits.max_compute_workgroup_size_x = 256;
        limits.max_compute_workgroup_size_y = 256;
        limits.max_storage_buffers_per_shader_stage = 8;
        limits.max_buffer_size = 1 << 30;
        limits.max_storage_buffer_binding_size = 1 << 30;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("glassbox"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| RuntimeError::Wgpu(format!("{e}")))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            pipelines: Mutex::new(AHashMap::new()),
            buffers: Mutex::new(AHashMap::new()),
            next_id: AtomicU64::new(1),
        })
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn release(&self, id: BufferId) {
        if let Ok(mut map) = self.buffers.lock() {
            map.remove(&id);
        }
    }

    pub fn live_buffer_count(&self) -> usize {
        self.buffers.lock().map(|m| m.len()).unwrap_or(0)
    }

    fn fresh_id(&self) -> BufferId {
        BufferId(self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    fn register(&self, buf: wgpu::Buffer) -> BufferId {
        let id = self.fresh_id();
        if let Ok(mut map) = self.buffers.lock() {
            map.insert(id, Arc::new(buf));
        }
        id
    }

    fn get(&self, id: BufferId) -> Result<Arc<wgpu::Buffer>> {
        self.buffers
            .lock()
            .ok()
            .and_then(|m| m.get(&id).cloned())
            .ok_or_else(|| RuntimeError::Wgpu(format!("buffer id {} not registered", id.0)))
    }

    fn buffer_of(&self, t: &Tensor) -> Result<Arc<wgpu::Buffer>> {
        match t.storage() {
            Storage::Gpu(id) => self.get(*id),
            Storage::Cpu(_) => Err(RuntimeError::Core(CoreError::WrongBackend {
                expected: "gpu",
                got: "cpu",
            })),
        }
    }

    fn pipeline(&self, key: &'static str, source: &str, entry: &str) -> Result<Arc<wgpu::ComputePipeline>> {
        if let Ok(map) = self.pipelines.lock() {
            if let Some(p) = map.get(key) {
                return Ok(Arc::clone(p));
            }
        }
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(key),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = Arc::new(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(key),
            layout: None,
            module: &module,
            entry_point: entry,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        }));
        if let Ok(mut map) = self.pipelines.lock() {
            map.insert(key, Arc::clone(&pipeline));
        }
        Ok(pipeline)
    }

    fn upload_storage_bytes(&self, label: &str, bytes: &[u8]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        })
    }

    fn upload_uniform<T: Pod>(&self, label: &str, data: &T) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn alloc_storage_buffer(&self, label: &str, byte_len: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_len.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn alloc_gpu_tensor(&self, shape: Shape, dtype: DType, label: &str) -> Tensor {
        let byte_len = (shape.numel() * dtype.size()) as u64;
        let buf = self.alloc_storage_buffer(label, byte_len);
        let id = self.register(buf);
        Tensor::from_gpu(id, shape, dtype)
    }

    pub async fn download_async(&self, tensor: &Tensor) -> Result<Tensor> {
        match tensor.storage() {
            Storage::Cpu(_) => Ok(tensor.clone()),
            Storage::Gpu(id) => {
                let buf = self.get(*id)?;
                let byte_len = (tensor.numel() * tensor.dtype().size()) as u64;
                let data = self.read_back_f32_bytes_async(&buf, byte_len).await?;
                Tensor::from_f32(&data, tensor.shape().clone()).map_err(RuntimeError::from)
            }
        }
    }

    async fn read_back_f32_bytes_async(&self, buf: &wgpu::Buffer, byte_len: u64) -> Result<Vec<f32>> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("glassbox/staging-async"),
            size: byte_len,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("glassbox/readback-async") });
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, byte_len);
        self.queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });

        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::Maintain::Wait);

        rx.await
            .map_err(|e| RuntimeError::Wgpu(format!("readback channel: {e}")))?
            .map_err(|e| RuntimeError::Wgpu(format!("map_async: {e}")))?;

        let data = slice.get_mapped_range();
        let out: Vec<f32> = cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }

    fn read_back_f32_bytes(&self, buf: &wgpu::Buffer, byte_len: u64) -> Result<Vec<f32>> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("glassbox/staging"),
            size: byte_len,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("glassbox/readback") });
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, byte_len);
        self.queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| RuntimeError::Wgpu(format!("readback channel: {e}")))?
            .map_err(|e| RuntimeError::Wgpu(format!("map_async: {e}")))?;

        let data = slice.get_mapped_range();
        let out: Vec<f32> = cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }
}

fn check_f32(t: &Tensor, op: &'static str) -> Result<()> {
    if t.dtype() != DType::F32 {
        return Err(RuntimeError::UnsupportedDType { op, dtype: t.dtype().name() });
    }
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulDims {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxDims {
    rows: u32,
    cols: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LayernormDims {
    rows: u32,
    cols: u32,
    eps: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EmbedDims {
    n_ids: u32,
    dim: u32,
    vocab: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AttentionDims {
    batch: u32,
    heads: u32,
    seq_q: u32,
    seq_k: u32,
    head_dim: u32,
    causal: u32,
    write_pattern: u32,
    _pad: u32,
}

impl Backend for WgpuBackend {
    fn name(&self) -> &'static str {
        "wgpu"
    }

    fn alloc(&self, shape: Shape, dtype: DType) -> Result<Tensor> {
        if dtype != DType::F32 {
            return Err(RuntimeError::UnsupportedDType { op: "alloc", dtype: dtype.name() });
        }
        Ok(self.alloc_gpu_tensor(shape, dtype, "alloc"))
    }

    fn upload(&self, tensor: &Tensor) -> Result<Tensor> {
        if matches!(tensor.storage(), Storage::Gpu(_)) {
            return Ok(tensor.clone());
        }
        check_f32(tensor, "upload")?;
        let bytes: Vec<u8> = cast_slice(tensor.as_f32()?).to_vec();
        let buf = self.upload_storage_bytes("upload", &bytes);
        let id = self.register(buf);
        Ok(Tensor::from_gpu(id, tensor.shape().clone(), tensor.dtype()))
    }

    fn download(&self, tensor: &Tensor) -> Result<Tensor> {
        match tensor.storage() {
            Storage::Cpu(_) => Ok(tensor.clone()),
            Storage::Gpu(id) => {
                let buf = self.get(*id)?;
                let byte_len = (tensor.numel() * tensor.dtype().size()) as u64;
                let data = self.read_back_f32_bytes(&buf, byte_len)?;
                Tensor::from_f32(&data, tensor.shape().clone()).map_err(RuntimeError::from)
            }
        }
    }

    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        check_f32(a, "matmul")?;
        check_f32(b, "matmul")?;
        check_f32(out, "matmul")?;
        if a.rank() != 2 || b.rank() != 2 {
            return Err(RuntimeError::MatmulShapeMismatch {
                a: a.shape().dims().to_vec(),
                b: b.shape().dims().to_vec(),
            });
        }
        let m = a.shape().dim(0)?;
        let k = a.shape().dim(1)?;
        let kb = b.shape().dim(0)?;
        let n = b.shape().dim(1)?;
        if k != kb || out.shape().dim(0)? != m || out.shape().dim(1)? != n {
            return Err(RuntimeError::MatmulShapeMismatch {
                a: a.shape().dims().to_vec(),
                b: b.shape().dims().to_vec(),
            });
        }

        let pipeline = self.pipeline("matmul", SHADER_MATMUL, "matmul")?;
        let dims = MatmulDims { m: m as u32, k: k as u32, n: n as u32, _pad: 0 };
        let dims_buf = self.upload_uniform("matmul/dims", &dims);

        let a_gpu = ensure_gpu(self, a, "matmul/a")?;
        let b_gpu = ensure_gpu(self, b, "matmul/b")?;
        let (out_gpu, out_buf) = ensure_gpu_writable(self, out, "matmul/out")?;

        let bg_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul/bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a_gpu.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_gpu.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
            ],
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("matmul/enc") });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul/pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (n as u32 + 15) / 16;
            let wg_y = (m as u32 + 15) / 16;
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        self.queue.submit(Some(enc.finish()));

        *out = out_gpu;
        Ok(())
    }

    fn add(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        check_f32(a, "add")?;
        check_f32(b, "add")?;
        check_f32(out, "add")?;
        if a.shape() != b.shape() || a.shape() != out.shape() {
            return Err(RuntimeError::Core(CoreError::ShapeMismatch {
                expected: a.shape().clone(),
                got: b.shape().clone(),
            }));
        }

        let pipeline = self.pipeline("add", SHADER_ADD, "add")?;
        let a_gpu = ensure_gpu(self, a, "add/a")?;
        let b_gpu = ensure_gpu(self, b, "add/b")?;
        let (out_gpu, out_buf) = ensure_gpu_writable(self, out, "add/out")?;

        let bg_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("add/bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_gpu.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_gpu.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
            ],
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("add/enc") });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("add/pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let wg = (a.numel() as u32 + 255) / 256;
            cpass.dispatch_workgroups(wg, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));

        *out = out_gpu;
        Ok(())
    }

    fn layer_norm(
        &self,
        x: &Tensor,
        gamma: &Tensor,
        beta: &Tensor,
        eps: f32,
        out: &mut Tensor,
    ) -> Result<()> {
        check_f32(x, "layer_norm")?;
        check_f32(gamma, "layer_norm")?;
        check_f32(beta, "layer_norm")?;
        check_f32(out, "layer_norm")?;
        let dims_v = x.shape().dims();
        if dims_v.is_empty() {
            return Err(RuntimeError::BadRank { op: "layer_norm", expected: 1, got: 0 });
        }
        let last = *dims_v.last().expect("non-empty");
        let rows = x.numel() / last;

        let pipeline = self.pipeline("layernorm", SHADER_LAYERNORM, "layernorm_row")?;
        let dims = LayernormDims { rows: rows as u32, cols: last as u32, eps, _pad: 0 };
        let dims_buf = self.upload_uniform("ln/dims", &dims);

        let x_gpu = ensure_gpu(self, x, "ln/x")?;
        let g_gpu = ensure_gpu(self, gamma, "ln/g")?;
        let b_gpu = ensure_gpu(self, beta, "ln/b")?;
        let (out_gpu, out_buf) = ensure_gpu_writable(self, out, "ln/out")?;

        let bg_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ln/bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_gpu.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: g_gpu.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: b_gpu.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: out_buf.as_entire_binding() },
            ],
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ln/enc") });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ln/pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(rows as u32, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));

        *out = out_gpu;
        Ok(())
    }

    fn gelu(&self, x: &Tensor, out: &mut Tensor) -> Result<()> {
        check_f32(x, "gelu")?;
        check_f32(out, "gelu")?;
        let pipeline = self.pipeline("gelu", SHADER_GELU, "gelu")?;
        let x_gpu = ensure_gpu(self, x, "gelu/x")?;
        let (out_gpu, out_buf) = ensure_gpu_writable(self, out, "gelu/out")?;

        let bg_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gelu/bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x_gpu.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
            ],
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gelu/enc") });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gelu/pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let wg = (x.numel() as u32 + 255) / 256;
            cpass.dispatch_workgroups(wg, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));

        *out = out_gpu;
        Ok(())
    }

    fn softmax(&self, x: &Tensor, axis: isize, out: &mut Tensor) -> Result<()> {
        check_f32(x, "softmax")?;
        check_f32(out, "softmax")?;
        let rank = x.rank() as isize;
        let axis = if axis < 0 { axis + rank } else { axis } as usize;
        let dims_v = x.shape().dims();
        if axis >= dims_v.len() || axis != dims_v.len() - 1 {
            return Err(RuntimeError::UnsupportedDType { op: "softmax", dtype: "non-last-axis" });
        }
        let last = dims_v[axis];
        let rows = x.numel() / last;

        let pipeline = self.pipeline("softmax", SHADER_SOFTMAX, "softmax_row")?;
        let dims = SoftmaxDims { rows: rows as u32, cols: last as u32 };
        let dims_buf = self.upload_uniform("softmax/dims", &dims);

        let x_gpu = ensure_gpu(self, x, "softmax/x")?;
        let (out_gpu, out_buf) = ensure_gpu_writable(self, out, "softmax/out")?;

        let bg_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax/bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_gpu.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
            ],
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("softmax/enc") });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("softmax/pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(rows as u32, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));

        *out = out_gpu;
        Ok(())
    }

    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: AttentionMask,
        out: &mut Tensor,
        pattern_out: Option<&mut Tensor>,
    ) -> Result<()> {
        check_f32(q, "attention")?;
        check_f32(k, "attention")?;
        check_f32(v, "attention")?;
        check_f32(out, "attention")?;
        let bs = q.shape().dim(0)?;
        let heads = q.shape().dim(1)?;
        let sq = q.shape().dim(2)?;
        let dh = q.shape().dim(3)?;
        let sk = k.shape().dim(2)?;

        let pipeline = self.pipeline("attention", SHADER_ATTENTION, "attention")?;
        let dims = AttentionDims {
            batch: bs as u32,
            heads: heads as u32,
            seq_q: sq as u32,
            seq_k: sk as u32,
            head_dim: dh as u32,
            causal: matches!(mask, AttentionMask::Causal) as u32,
            write_pattern: pattern_out.is_some() as u32,
            _pad: 0,
        };
        let dims_buf = self.upload_uniform("attn/dims", &dims);

        let q_gpu = ensure_gpu(self, q, "attn/q")?;
        let k_gpu = ensure_gpu(self, k, "attn/k")?;
        let v_gpu = ensure_gpu(self, v, "attn/v")?;
        let (out_gpu, out_buf) = ensure_gpu_writable(self, out, "attn/out")?;
        let pat_shape = Shape::from([bs, heads, sq, sk]);
        let pat_tensor = self.alloc_gpu_tensor(pat_shape.clone(), DType::F32, "attn/pat");
        let pat_buf = self.buffer_of(&pat_tensor)?;

        let bg_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("attn/bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: q_gpu.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: k_gpu.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: v_gpu.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: pat_buf.as_entire_binding() },
            ],
        });

        let total_rows = (bs * heads * sq) as u32;
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("attn/enc") });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("attn/pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let wg = (total_rows + 63) / 64;
            cpass.dispatch_workgroups(wg, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));

        *out = out_gpu;
        if let Some(p) = pattern_out {
            *p = pat_tensor;
        } else if let Some(id) = pat_tensor.storage().gpu_id() {
            self.release(id);
        }
        Ok(())
    }

    fn embed(&self, table: &Tensor, ids: &[u32], out: &mut Tensor) -> Result<()> {
        check_f32(table, "embed")?;
        check_f32(out, "embed")?;
        let dim = table.shape().dim(1)?;
        let vocab = table.shape().dim(0)?;

        let pipeline = self.pipeline("embed", SHADER_EMBED, "embed")?;
        let dims = EmbedDims {
            n_ids: ids.len() as u32,
            dim: dim as u32,
            vocab: vocab as u32,
            _pad: 0,
        };
        let dims_buf = self.upload_uniform("embed/dims", &dims);
        let table_gpu = ensure_gpu(self, table, "embed/table")?;
        let ids_buf = Arc::new(self.upload_storage_bytes("embed/ids", cast_slice(ids)));
        let (out_gpu, out_buf) = ensure_gpu_writable(self, out, "embed/out")?;

        let bg_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("embed/bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: table_gpu.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: ids_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
            ],
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("embed/enc") });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("embed/pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let wg = (ids.len() as u32 + 63) / 64;
            cpass.dispatch_workgroups(wg, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));

        *out = out_gpu;
        Ok(())
    }
}

fn ensure_gpu(backend: &WgpuBackend, t: &Tensor, label: &str) -> Result<Arc<wgpu::Buffer>> {
    match t.storage() {
        Storage::Gpu(id) => backend.get(*id),
        Storage::Cpu(_) => {
            let bytes: Vec<u8> = cast_slice(t.as_f32()?).to_vec();
            Ok(Arc::new(backend.upload_storage_bytes(label, &bytes)))
        }
    }
}

fn ensure_gpu_writable(
    backend: &WgpuBackend,
    out: &Tensor,
    label: &str,
) -> Result<(Tensor, Arc<wgpu::Buffer>)> {
    match out.storage() {
        Storage::Gpu(id) => {
            let buf = backend.get(*id)?;
            Ok((out.clone(), buf))
        }
        Storage::Cpu(_) => {
            let new_t = backend.alloc_gpu_tensor(out.shape().clone(), out.dtype(), label);
            let buf = backend.buffer_of(&new_t)?;
            Ok((new_t, buf))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuBackend;

    fn try_backend() -> Option<WgpuBackend> {
        pollster::block_on(WgpuBackend::new()).ok()
    }

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    #[test]
    fn download_async_roundtrips() {
        let Some(gpu) = try_backend() else { return };
        let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.5).collect();
        let cpu_t = Tensor::from_f32(&data, Shape::from([8, 4])).unwrap();
        let gpu_t = gpu.upload(&cpu_t).unwrap();
        let back = pollster::block_on(gpu.download_async(&gpu_t)).unwrap();
        assert!(back.storage().is_cpu());
        assert!(approx_eq(back.as_f32().unwrap(), &data, 1e-6));
    }

    #[test]
    fn upload_then_download_roundtrips() {
        let Some(gpu) = try_backend() else { return };
        let data: Vec<f32> = (0..64).map(|i| (i as f32 * 0.31).sin()).collect();
        let cpu_t = Tensor::from_f32(&data, Shape::from([8, 8])).unwrap();
        let gpu_t = gpu.upload(&cpu_t).unwrap();
        assert!(gpu_t.storage().is_gpu());
        let back = gpu.download(&gpu_t).unwrap();
        assert!(back.storage().is_cpu());
        assert!(approx_eq(back.as_f32().unwrap(), &data, 1e-6));
    }

    #[test]
    fn matmul_resident_matches_cpu() {
        let Some(gpu) = try_backend() else { return };
        let m = 16;
        let k = 24;
        let n = 8;
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.13).sin()).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.07).cos()).collect();
        let a = Tensor::from_f32(&a_data, Shape::from([m, k])).unwrap();
        let b = Tensor::from_f32(&b_data, Shape::from([k, n])).unwrap();

        let mut out_cpu = Tensor::from_f32(&vec![0.0; m * n], Shape::from([m, n])).unwrap();
        CpuBackend.matmul(&a, &b, &mut out_cpu).unwrap();

        let a_g = gpu.upload(&a).unwrap();
        let b_g = gpu.upload(&b).unwrap();
        let mut out_g = gpu.alloc(Shape::from([m, n]), DType::F32).unwrap();
        gpu.matmul(&a_g, &b_g, &mut out_g).unwrap();
        assert!(out_g.storage().is_gpu());
        let down = gpu.download(&out_g).unwrap();
        assert!(approx_eq(out_cpu.as_f32().unwrap(), down.as_f32().unwrap(), 1e-3));
    }

    #[test]
    fn chained_matmul_stays_gpu() {
        let Some(gpu) = try_backend() else { return };
        let n = 8;
        let a = Tensor::from_f32(
            &(0..n * n).map(|i| (i as f32 * 0.07).sin()).collect::<Vec<_>>(),
            Shape::from([n, n]),
        )
        .unwrap();
        let b = Tensor::from_f32(
            &(0..n * n).map(|i| (i as f32 * 0.11).cos()).collect::<Vec<_>>(),
            Shape::from([n, n]),
        )
        .unwrap();
        let c = Tensor::from_f32(
            &(0..n * n).map(|i| (i as f32 * 0.13).sin()).collect::<Vec<_>>(),
            Shape::from([n, n]),
        )
        .unwrap();

        let a_g = gpu.upload(&a).unwrap();
        let b_g = gpu.upload(&b).unwrap();
        let c_g = gpu.upload(&c).unwrap();
        let mut ab = gpu.alloc(Shape::from([n, n]), DType::F32).unwrap();
        gpu.matmul(&a_g, &b_g, &mut ab).unwrap();
        assert!(ab.storage().is_gpu());
        let mut abc = gpu.alloc(Shape::from([n, n]), DType::F32).unwrap();
        gpu.matmul(&ab, &c_g, &mut abc).unwrap();
        assert!(abc.storage().is_gpu());

        let mut ab_cpu = Tensor::from_f32(&vec![0.0; n * n], Shape::from([n, n])).unwrap();
        CpuBackend.matmul(&a, &b, &mut ab_cpu).unwrap();
        let mut abc_cpu = Tensor::from_f32(&vec![0.0; n * n], Shape::from([n, n])).unwrap();
        CpuBackend.matmul(&ab_cpu, &c, &mut abc_cpu).unwrap();

        let down = gpu.download(&abc).unwrap();
        assert!(approx_eq(abc_cpu.as_f32().unwrap(), down.as_f32().unwrap(), 5e-3));
    }

    #[test]
    fn release_drops_buffer() {
        let Some(gpu) = try_backend() else { return };
        let t = gpu.alloc(Shape::from([16]), DType::F32).unwrap();
        let id = t.storage().gpu_id().unwrap();
        let before = gpu.live_buffer_count();
        gpu.release(id);
        let after = gpu.live_buffer_count();
        assert_eq!(after + 1, before);
    }

    #[test]
    fn softmax_resident_matches_cpu() {
        let Some(gpu) = try_backend() else { return };
        let rows = 4;
        let cols = 32;
        let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.21).sin()).collect();
        let x = Tensor::from_f32(&data, Shape::from([rows, cols])).unwrap();

        let mut out_cpu = Tensor::from_f32(&vec![0.0; rows * cols], Shape::from([rows, cols])).unwrap();
        CpuBackend.softmax(&x, -1, &mut out_cpu).unwrap();

        let x_g = gpu.upload(&x).unwrap();
        let mut out_g = gpu.alloc(Shape::from([rows, cols]), DType::F32).unwrap();
        gpu.softmax(&x_g, -1, &mut out_g).unwrap();
        let down = gpu.download(&out_g).unwrap();
        assert!(approx_eq(out_cpu.as_f32().unwrap(), down.as_f32().unwrap(), 1e-5));
    }

    #[test]
    fn add_kernel_matches_cpu() {
        let Some(gpu) = try_backend() else { return };
        let n = 137;
        let a_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.13).sin()).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.21).cos()).collect();
        let a = Tensor::from_f32(&a_data, Shape::from([n])).unwrap();
        let b = Tensor::from_f32(&b_data, Shape::from([n])).unwrap();

        let mut out_cpu = Tensor::from_f32(&vec![0.0; n], Shape::from([n])).unwrap();
        CpuBackend.add(&a, &b, &mut out_cpu).unwrap();

        let a_g = gpu.upload(&a).unwrap();
        let b_g = gpu.upload(&b).unwrap();
        let mut out_g = gpu.alloc(Shape::from([n]), DType::F32).unwrap();
        gpu.add(&a_g, &b_g, &mut out_g).unwrap();
        assert!(out_g.storage().is_gpu());
        let down = gpu.download(&out_g).unwrap();
        assert!(approx_eq(out_cpu.as_f32().unwrap(), down.as_f32().unwrap(), 1e-6));
    }

    #[test]
    fn embed_kernel_matches_cpu() {
        let Some(gpu) = try_backend() else { return };
        let vocab = 32;
        let dim = 8;
        let table_data: Vec<f32> = (0..vocab * dim).map(|i| i as f32 * 0.01).collect();
        let table = Tensor::from_f32(&table_data, Shape::from([vocab, dim])).unwrap();
        let ids: Vec<u32> = vec![0, 5, 17, 31, 12];

        let mut out_cpu = Tensor::from_f32(&vec![0.0; ids.len() * dim], Shape::from([ids.len(), dim])).unwrap();
        CpuBackend.embed(&table, &ids, &mut out_cpu).unwrap();

        let table_g = gpu.upload(&table).unwrap();
        let mut out_g = gpu.alloc(Shape::from([ids.len(), dim]), DType::F32).unwrap();
        gpu.embed(&table_g, &ids, &mut out_g).unwrap();
        assert!(out_g.storage().is_gpu());
        let down = gpu.download(&out_g).unwrap();
        assert!(approx_eq(out_cpu.as_f32().unwrap(), down.as_f32().unwrap(), 1e-6));
    }

    #[test]
    fn attention_resident_matches_cpu() {
        let Some(gpu) = try_backend() else { return };
        let bs = 1;
        let heads = 2;
        let sq = 4;
        let dh = 8;
        let sk = 4;
        let q_data: Vec<f32> = (0..bs * heads * sq * dh).map(|i| (i as f32 * 0.17).sin()).collect();
        let k_data: Vec<f32> = (0..bs * heads * sk * dh).map(|i| (i as f32 * 0.19).cos()).collect();
        let v_data: Vec<f32> = (0..bs * heads * sk * dh).map(|i| (i as f32 * 0.23).sin()).collect();
        let q = Tensor::from_f32(&q_data, Shape::from([bs, heads, sq, dh])).unwrap();
        let k = Tensor::from_f32(&k_data, Shape::from([bs, heads, sk, dh])).unwrap();
        let v = Tensor::from_f32(&v_data, Shape::from([bs, heads, sk, dh])).unwrap();

        let mut out_cpu = Tensor::from_f32(&vec![0.0; bs * heads * sq * dh], Shape::from([bs, heads, sq, dh])).unwrap();
        CpuBackend.attention(&q, &k, &v, AttentionMask::Causal, &mut out_cpu, None).unwrap();

        let q_g = gpu.upload(&q).unwrap();
        let k_g = gpu.upload(&k).unwrap();
        let v_g = gpu.upload(&v).unwrap();
        let mut out_g = gpu.alloc(Shape::from([bs, heads, sq, dh]), DType::F32).unwrap();
        gpu.attention(&q_g, &k_g, &v_g, AttentionMask::Causal, &mut out_g, None).unwrap();
        let down = gpu.download(&out_g).unwrap();
        assert!(approx_eq(out_cpu.as_f32().unwrap(), down.as_f32().unwrap(), 1e-3));
    }
}
