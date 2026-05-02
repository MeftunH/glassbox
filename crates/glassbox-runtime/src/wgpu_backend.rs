use std::sync::{Arc, Mutex};

use ahash::AHashMap;
use bytemuck::{cast_slice, Pod};
use glassbox_core::{CoreError, DType, Storage, Tensor};
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
    pipelines: Mutex<AHashMap<&'static str, Arc<wgpu::ComputePipeline>>>,
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
        })
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
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

    fn upload_storage<T: Pod>(&self, label: &str, data: &[T]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: cast_slice(data),
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

    fn alloc_storage(&self, label: &str, byte_len: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_len.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn read_back_f32(&self, buf: &wgpu::Buffer, byte_len: u64) -> Result<Vec<f32>> {
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

fn write_f32(out: &mut Tensor, data: Vec<f32>) -> Result<()> {
    if data.len() != out.numel() {
        return Err(RuntimeError::Core(CoreError::ElementCountMismatch {
            got: data.len(),
            expected: out.numel(),
            shape: out.shape().clone(),
        }));
    }
    let bytes: Vec<u8> = cast_slice(&data).to_vec();
    *out = Tensor::new(Storage::cpu(Arc::<[u8]>::from(bytes)), out.shape().clone(), out.dtype());
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
        let a_buf = self.upload_storage("matmul/a", a.as_f32()?);
        let b_buf = self.upload_storage("matmul/b", b.as_f32()?);
        let out_byte_len = (m * n * 4) as u64;
        let out_buf = self.alloc_storage("matmul/out", out_byte_len);

        let bg_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul/bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_buf.as_entire_binding() },
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

        let result = self.read_back_f32(&out_buf, out_byte_len)?;
        write_f32(out, result)
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
        let result: Vec<f32> = a
            .as_f32()?
            .iter()
            .zip(b.as_f32()?.iter())
            .map(|(x, y)| x + y)
            .collect();
        write_f32(out, result)
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
        let x_buf = self.upload_storage("ln/x", x.as_f32()?);
        let g_buf = self.upload_storage("ln/g", gamma.as_f32()?);
        let b_buf = self.upload_storage("ln/b", beta.as_f32()?);
        let out_byte_len = (x.numel() * 4) as u64;
        let out_buf = self.alloc_storage("ln/out", out_byte_len);

        let bg_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ln/bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: g_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: b_buf.as_entire_binding() },
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

        let result = self.read_back_f32(&out_buf, out_byte_len)?;
        write_f32(out, result)
    }

    fn gelu(&self, x: &Tensor, out: &mut Tensor) -> Result<()> {
        check_f32(x, "gelu")?;
        check_f32(out, "gelu")?;
        let pipeline = self.pipeline("gelu", SHADER_GELU, "gelu")?;
        let x_buf = self.upload_storage("gelu/x", x.as_f32()?);
        let out_byte_len = (x.numel() * 4) as u64;
        let out_buf = self.alloc_storage("gelu/out", out_byte_len);

        let bg_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gelu/bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
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

        let result = self.read_back_f32(&out_buf, out_byte_len)?;
        write_f32(out, result)
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
        let x_buf = self.upload_storage("softmax/x", x.as_f32()?);
        let out_byte_len = (x.numel() * 4) as u64;
        let out_buf = self.alloc_storage("softmax/out", out_byte_len);

        let bg_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax/bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
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

        let result = self.read_back_f32(&out_buf, out_byte_len)?;
        write_f32(out, result)
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
        let q_buf = self.upload_storage("attn/q", q.as_f32()?);
        let k_buf = self.upload_storage("attn/k", k.as_f32()?);
        let v_buf = self.upload_storage("attn/v", v.as_f32()?);
        let out_byte_len = (bs * heads * sq * dh * 4) as u64;
        let out_buf = self.alloc_storage("attn/out", out_byte_len);
        let pat_byte_len = (bs * heads * sq * sk * 4) as u64;
        let pat_buf = self.alloc_storage("attn/pat", pat_byte_len);

        let bg_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("attn/bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: k_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: v_buf.as_entire_binding() },
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

        let out_result = self.read_back_f32(&out_buf, out_byte_len)?;
        write_f32(out, out_result)?;
        if let Some(p) = pattern_out {
            let pat_result = self.read_back_f32(&pat_buf, pat_byte_len)?;
            write_f32(p, pat_result)?;
        }
        Ok(())
    }

    fn embed(&self, table: &Tensor, ids: &[u32], out: &mut Tensor) -> Result<()> {
        check_f32(table, "embed")?;
        check_f32(out, "embed")?;
        let dim = table.shape().dim(1)?;
        let vocab = table.shape().dim(0)?;
        let table_data = table.as_f32()?;
        let mut result = vec![0.0f32; ids.len() * dim];
        for (i, &id) in ids.iter().enumerate() {
            let id = id as usize;
            if id >= vocab {
                return Err(RuntimeError::Core(CoreError::AxisOutOfBounds { axis: id, rank: vocab }));
            }
            result[i * dim..(i + 1) * dim].copy_from_slice(&table_data[id * dim..(id + 1) * dim]);
        }
        write_f32(out, result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuBackend;
    use glassbox_core::Shape;

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
    fn matmul_parity_with_cpu() {
        let Some(gpu) = try_backend() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let m = 8;
        let k = 16;
        let n = 12;
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.13).sin()).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.07).cos()).collect();
        let a = Tensor::from_f32(&a_data, Shape::from([m, k])).unwrap();
        let b = Tensor::from_f32(&b_data, Shape::from([k, n])).unwrap();

        let mut out_cpu = Tensor::from_f32(&vec![0.0; m * n], Shape::from([m, n])).unwrap();
        CpuBackend.matmul(&a, &b, &mut out_cpu).unwrap();

        let mut out_gpu = Tensor::from_f32(&vec![0.0; m * n], Shape::from([m, n])).unwrap();
        gpu.matmul(&a, &b, &mut out_gpu).unwrap();

        assert!(approx_eq(out_cpu.as_f32().unwrap(), out_gpu.as_f32().unwrap(), 1e-4));
    }

    #[test]
    fn softmax_parity_with_cpu() {
        let Some(gpu) = try_backend() else { return };
        let rows = 4;
        let cols = 32;
        let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.21).sin()).collect();
        let x = Tensor::from_f32(&data, Shape::from([rows, cols])).unwrap();

        let mut out_cpu = Tensor::from_f32(&vec![0.0; rows * cols], Shape::from([rows, cols])).unwrap();
        CpuBackend.softmax(&x, -1, &mut out_cpu).unwrap();

        let mut out_gpu = Tensor::from_f32(&vec![0.0; rows * cols], Shape::from([rows, cols])).unwrap();
        gpu.softmax(&x, -1, &mut out_gpu).unwrap();

        assert!(approx_eq(out_cpu.as_f32().unwrap(), out_gpu.as_f32().unwrap(), 1e-5));
    }

    #[test]
    fn layer_norm_parity_with_cpu() {
        let Some(gpu) = try_backend() else { return };
        let rows = 4;
        let cols = 32;
        let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.11).cos()).collect();
        let x = Tensor::from_f32(&data, Shape::from([rows, cols])).unwrap();
        let g = Tensor::from_f32(&vec![1.0; cols], Shape::from([cols])).unwrap();
        let b = Tensor::from_f32(&vec![0.0; cols], Shape::from([cols])).unwrap();

        let mut out_cpu = Tensor::from_f32(&vec![0.0; rows * cols], Shape::from([rows, cols])).unwrap();
        CpuBackend.layer_norm(&x, &g, &b, 1e-5, &mut out_cpu).unwrap();

        let mut out_gpu = Tensor::from_f32(&vec![0.0; rows * cols], Shape::from([rows, cols])).unwrap();
        gpu.layer_norm(&x, &g, &b, 1e-5, &mut out_gpu).unwrap();

        assert!(approx_eq(out_cpu.as_f32().unwrap(), out_gpu.as_f32().unwrap(), 1e-3));
    }

    #[test]
    fn gelu_parity_with_cpu() {
        let Some(gpu) = try_backend() else { return };
        let n = 256;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.05).collect();
        let x = Tensor::from_f32(&data, Shape::from([n])).unwrap();

        let mut out_cpu = Tensor::from_f32(&vec![0.0; n], Shape::from([n])).unwrap();
        CpuBackend.gelu(&x, &mut out_cpu).unwrap();

        let mut out_gpu = Tensor::from_f32(&vec![0.0; n], Shape::from([n])).unwrap();
        gpu.gelu(&x, &mut out_gpu).unwrap();

        assert!(approx_eq(out_cpu.as_f32().unwrap(), out_gpu.as_f32().unwrap(), 5e-3));
    }

    #[test]
    fn attention_parity_with_cpu() {
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

        let mut out_gpu = Tensor::from_f32(&vec![0.0; bs * heads * sq * dh], Shape::from([bs, heads, sq, dh])).unwrap();
        gpu.attention(&q, &k, &v, AttentionMask::Causal, &mut out_gpu, None).unwrap();

        assert!(approx_eq(out_cpu.as_f32().unwrap(), out_gpu.as_f32().unwrap(), 1e-3));
    }
}
