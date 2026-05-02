use std::sync::Arc;

use bytemuck::cast_slice;
use glassbox_core::{CoreError, DType, Shape, Storage, Tensor};
use rayon::prelude::*;

use crate::backend::{AttentionMask, Backend};
use crate::error::{Result, RuntimeError};

#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

fn check_f32(t: &Tensor, op: &'static str) -> Result<()> {
    if t.dtype() != DType::F32 {
        return Err(RuntimeError::UnsupportedDType { op, dtype: t.dtype().name() });
    }
    Ok(())
}

fn write_f32(out: &mut Tensor, data: &[f32]) -> Result<()> {
    if data.len() != out.numel() {
        return Err(RuntimeError::Core(CoreError::ElementCountMismatch {
            got: data.len(),
            expected: out.numel(),
            shape: out.shape().clone(),
        }));
    }
    let bytes: Vec<u8> = cast_slice(data).to_vec();
    *out = Tensor::new(Storage::cpu(Arc::<[u8]>::from(bytes)), out.shape().clone(), out.dtype());
    Ok(())
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn alloc(&self, shape: Shape, dtype: DType) -> Result<Tensor> {
        if dtype != DType::F32 {
            return Err(RuntimeError::UnsupportedDType { op: "alloc", dtype: dtype.name() });
        }
        Tensor::from_f32(&vec![0.0; shape.numel()], shape).map_err(RuntimeError::from)
    }

    fn upload(&self, tensor: &Tensor) -> Result<Tensor> {
        Ok(tensor.clone())
    }

    fn download(&self, tensor: &Tensor) -> Result<Tensor> {
        Ok(tensor.clone())
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

        let a_data = a.as_f32()?;
        let b_data = b.as_f32()?;
        let mut tmp = vec![0.0f32; m * n];

        tmp.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a_data[i * k + kk] * b_data[kk * n + j];
                }
                row[j] = acc;
            }
        });

        write_f32(out, &tmp)
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
        let a_data = a.as_f32()?;
        let b_data = b.as_f32()?;
        let result: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(x, y)| x + y).collect();
        write_f32(out, &result)
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
        let dims = x.shape().dims();
        if dims.is_empty() {
            return Err(RuntimeError::BadRank { op: "layer_norm", expected: 1, got: 0 });
        }
        let last = *dims.last().expect("non-empty");
        if gamma.numel() != last || beta.numel() != last {
            return Err(RuntimeError::Core(CoreError::ShapeMismatch {
                expected: x.shape().clone(),
                got: gamma.shape().clone(),
            }));
        }
        let xs = x.as_f32()?;
        let g = gamma.as_f32()?;
        let bvec = beta.as_f32()?;
        let mut result = vec![0.0f32; xs.len()];

        result.par_chunks_mut(last).enumerate().for_each(|(row, dst)| {
            let src = &xs[row * last..(row + 1) * last];
            let mean: f32 = src.iter().sum::<f32>() / last as f32;
            let var: f32 = src.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / last as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for i in 0..last {
                dst[i] = (src[i] - mean) * inv * g[i] + bvec[i];
            }
        });

        write_f32(out, &result)
    }

    fn gelu(&self, x: &Tensor, out: &mut Tensor) -> Result<()> {
        check_f32(x, "gelu")?;
        check_f32(out, "gelu")?;
        let xs = x.as_f32()?;
        let result: Vec<f32> = xs.iter().map(|&v| gelu_exact(v)).collect();
        write_f32(out, &result)
    }

    fn softmax(&self, x: &Tensor, axis: isize, out: &mut Tensor) -> Result<()> {
        check_f32(x, "softmax")?;
        check_f32(out, "softmax")?;
        let rank = x.rank() as isize;
        let axis = if axis < 0 { axis + rank } else { axis } as usize;
        let dims = x.shape().dims();
        if axis >= dims.len() {
            return Err(RuntimeError::Core(CoreError::AxisOutOfBounds { axis, rank: dims.len() }));
        }
        if axis != dims.len() - 1 {
            return Err(RuntimeError::UnsupportedDType { op: "softmax", dtype: "non-last-axis" });
        }
        let last = dims[axis];
        let xs = x.as_f32()?;
        let mut result = vec![0.0f32; xs.len()];

        result.par_chunks_mut(last).enumerate().for_each(|(row, dst)| {
            let src = &xs[row * last..(row + 1) * last];
            let mut max = f32::NEG_INFINITY;
            for &v in src {
                if v > max {
                    max = v;
                }
            }
            let mut sum = 0.0f32;
            for (i, &v) in src.iter().enumerate() {
                let e = (v - max).exp();
                dst[i] = e;
                sum += e;
            }
            if sum > 0.0 {
                for v in dst.iter_mut() {
                    *v /= sum;
                }
            }
        });

        write_f32(out, &result)
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
        if q.rank() != 4 || k.rank() != 4 || v.rank() != 4 {
            return Err(RuntimeError::AttentionShapeMismatch {
                q: q.shape().dims().to_vec(),
                k: k.shape().dims().to_vec(),
                v: v.shape().dims().to_vec(),
            });
        }
        let bs = q.shape().dim(0)?;
        let heads = q.shape().dim(1)?;
        let sq = q.shape().dim(2)?;
        let dh = q.shape().dim(3)?;
        let sk = k.shape().dim(2)?;
        if k.shape().dims() != [bs, heads, sk, dh] || v.shape().dims() != [bs, heads, sk, dh] {
            return Err(RuntimeError::AttentionShapeMismatch {
                q: q.shape().dims().to_vec(),
                k: k.shape().dims().to_vec(),
                v: v.shape().dims().to_vec(),
            });
        }

        let scale = 1.0 / (dh as f32).sqrt();
        let qs = q.as_f32()?;
        let ks = k.as_f32()?;
        let vs = v.as_f32()?;
        let mut out_buf = vec![0.0f32; bs * heads * sq * dh];
        let mut pat_buf = vec![0.0f32; bs * heads * sq * sk];

        for b in 0..bs {
            for h in 0..heads {
                for i in 0..sq {
                    let q_row = &qs[((b * heads + h) * sq + i) * dh..][..dh];
                    let mut scores = vec![0.0f32; sk];
                    for j in 0..sk {
                        if matches!(mask, AttentionMask::Causal) && j > i {
                            scores[j] = f32::NEG_INFINITY;
                            continue;
                        }
                        let k_row = &ks[((b * heads + h) * sk + j) * dh..][..dh];
                        let mut s = 0.0f32;
                        for d in 0..dh {
                            s += q_row[d] * k_row[d];
                        }
                        scores[j] = s * scale;
                    }
                    let mut max = f32::NEG_INFINITY;
                    for &s in &scores {
                        if s > max {
                            max = s;
                        }
                    }
                    let mut sum = 0.0f32;
                    for s in scores.iter_mut() {
                        *s = (*s - max).exp();
                        sum += *s;
                    }
                    if sum > 0.0 {
                        for s in scores.iter_mut() {
                            *s /= sum;
                        }
                    }
                    let pat_off = ((b * heads + h) * sq + i) * sk;
                    pat_buf[pat_off..pat_off + sk].copy_from_slice(&scores);

                    let out_off = ((b * heads + h) * sq + i) * dh;
                    for d in 0..dh {
                        let mut acc = 0.0f32;
                        for j in 0..sk {
                            let v_row = &vs[((b * heads + h) * sk + j) * dh..][..dh];
                            acc += scores[j] * v_row[d];
                        }
                        out_buf[out_off + d] = acc;
                    }
                }
            }
        }

        write_f32(out, &out_buf)?;
        if let Some(p) = pattern_out {
            write_f32(p, &pat_buf)?;
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
        write_f32(out, &result)
    }
}

fn gelu_exact(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

fn erf(x: f32) -> f32 {
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p = 0.327_591_1;

    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use glassbox_core::Shape;

    #[test]
    fn matmul_2x3_3x2() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from([2, 3])).unwrap();
        let b = Tensor::from_f32(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], Shape::from([3, 2])).unwrap();
        let mut out = Tensor::from_f32(&[0.0; 4], Shape::from([2, 2])).unwrap();
        CpuBackend.matmul(&a, &b, &mut out).unwrap();
        assert_eq!(out.as_f32().unwrap(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn layer_norm_zero_mean_unit_var() {
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], Shape::from([1, 4])).unwrap();
        let g = Tensor::from_f32(&[1.0; 4], Shape::from([4])).unwrap();
        let b = Tensor::from_f32(&[0.0; 4], Shape::from([4])).unwrap();
        let mut out = Tensor::from_f32(&[0.0; 4], Shape::from([1, 4])).unwrap();
        CpuBackend.layer_norm(&x, &g, &b, 1e-5, &mut out).unwrap();
        let r = out.as_f32().unwrap();
        let mean: f32 = r.iter().sum::<f32>() / 4.0;
        let var: f32 = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-4, "mean = {mean}");
        assert!((var - 1.0).abs() < 1e-3, "var = {var}");
    }

    #[test]
    fn softmax_last_axis() {
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], Shape::from([2, 3])).unwrap();
        let mut out = Tensor::from_f32(&[0.0; 6], Shape::from([2, 3])).unwrap();
        CpuBackend.softmax(&x, -1, &mut out).unwrap();
        let r = out.as_f32().unwrap();
        let s0: f32 = r[..3].iter().sum();
        let s1: f32 = r[3..].iter().sum();
        assert!((s0 - 1.0).abs() < 1e-5);
        assert!((s1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn gelu_known_values() {
        let x = Tensor::from_f32(&[0.0, 1.0, -1.0], Shape::from([3])).unwrap();
        let mut out = Tensor::from_f32(&[0.0; 3], Shape::from([3])).unwrap();
        CpuBackend.gelu(&x, &mut out).unwrap();
        let r = out.as_f32().unwrap();
        assert!(r[0].abs() < 1e-6);
        assert!((r[1] - 0.8413).abs() < 1e-3);
        assert!((r[2] - (-0.1587)).abs() < 1e-3);
    }

    #[test]
    fn attention_causal_smoke() {
        let bs = 1;
        let heads = 1;
        let sq = 3;
        let dh = 2;
        let q = Tensor::from_f32(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], Shape::from([bs, heads, sq, dh])).unwrap();
        let k = q.clone();
        let v = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from([bs, heads, sq, dh])).unwrap();
        let mut out = Tensor::from_f32(&[0.0; 6], Shape::from([bs, heads, sq, dh])).unwrap();
        CpuBackend.attention(&q, &k, &v, AttentionMask::Causal, &mut out, None).unwrap();
        let r = out.as_f32().unwrap();
        assert!((r[0] - 1.0).abs() < 1e-5);
        assert!((r[1] - 2.0).abs() < 1e-5);
    }
}
