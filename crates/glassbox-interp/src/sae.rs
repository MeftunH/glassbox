use glassbox_core::{Shape, Tensor};
use glassbox_runtime::Backend;

use crate::error::{InterpError, Result};

#[derive(Debug, Clone)]
pub struct SparseAutoencoder {
    pub d_in: usize,
    pub d_features: usize,
    pub w_enc: Tensor,
    pub b_enc: Tensor,
    pub w_dec: Tensor,
    pub b_dec: Tensor,
}

impl SparseAutoencoder {
    pub fn new(
        d_in: usize,
        d_features: usize,
        w_enc: Tensor,
        b_enc: Tensor,
        w_dec: Tensor,
        b_dec: Tensor,
    ) -> Result<Self> {
        if w_enc.shape().dims() != [d_in, d_features] {
            return Err(InterpError::ShapeMismatch {
                what: "w_enc",
                expected: format!("[{d_in}, {d_features}]"),
                got: format!("{:?}", w_enc.shape()),
            });
        }
        if b_enc.numel() != d_features {
            return Err(InterpError::ShapeMismatch {
                what: "b_enc",
                expected: format!("[{d_features}]"),
                got: format!("{:?}", b_enc.shape()),
            });
        }
        if w_dec.shape().dims() != [d_features, d_in] {
            return Err(InterpError::ShapeMismatch {
                what: "w_dec",
                expected: format!("[{d_features}, {d_in}]"),
                got: format!("{:?}", w_dec.shape()),
            });
        }
        if b_dec.numel() != d_in {
            return Err(InterpError::ShapeMismatch {
                what: "b_dec",
                expected: format!("[{d_in}]"),
                got: format!("{:?}", b_dec.shape()),
            });
        }
        Ok(Self { d_in, d_features, w_enc, b_enc, w_dec, b_dec })
    }

    pub fn encode(&self, backend: &dyn Backend, x: &Tensor) -> Result<Tensor> {
        let seq = x.shape().dim(0).map_err(|e| InterpError::Forward(e.to_string()))?;
        if x.shape().dim(1).map_err(|e| InterpError::Forward(e.to_string()))? != self.d_in {
            return Err(InterpError::ShapeMismatch {
                what: "encode input",
                expected: format!("[seq, {}]", self.d_in),
                got: format!("{:?}", x.shape()),
            });
        }
        let mut z = Tensor::from_f32(&vec![0.0; seq * self.d_features], Shape::from([seq, self.d_features]))
            .map_err(|e| InterpError::Forward(e.to_string()))?;
        backend.matmul(x, &self.w_enc, &mut z).map_err(|e| InterpError::Forward(e.to_string()))?;
        let z_data = z.as_f32().map_err(|e| InterpError::Forward(e.to_string()))?;
        let b_data = self.b_enc.as_f32().map_err(|e| InterpError::Forward(e.to_string()))?;
        let mut out = vec![0.0f32; seq * self.d_features];
        for i in 0..seq {
            for j in 0..self.d_features {
                let v = z_data[i * self.d_features + j] + b_data[j];
                out[i * self.d_features + j] = if v > 0.0 { v } else { 0.0 };
            }
        }
        Tensor::from_f32(&out, Shape::from([seq, self.d_features]))
            .map_err(|e| InterpError::Forward(e.to_string()))
    }

    pub fn decode(&self, backend: &dyn Backend, features: &Tensor) -> Result<Tensor> {
        let seq = features.shape().dim(0).map_err(|e| InterpError::Forward(e.to_string()))?;
        let mut z = Tensor::from_f32(&vec![0.0; seq * self.d_in], Shape::from([seq, self.d_in]))
            .map_err(|e| InterpError::Forward(e.to_string()))?;
        backend.matmul(features, &self.w_dec, &mut z).map_err(|e| InterpError::Forward(e.to_string()))?;
        let z_data = z.as_f32().map_err(|e| InterpError::Forward(e.to_string()))?;
        let b_data = self.b_dec.as_f32().map_err(|e| InterpError::Forward(e.to_string()))?;
        let mut out = vec![0.0f32; seq * self.d_in];
        for i in 0..seq {
            for j in 0..self.d_in {
                out[i * self.d_in + j] = z_data[i * self.d_in + j] + b_data[j];
            }
        }
        Tensor::from_f32(&out, Shape::from([seq, self.d_in]))
            .map_err(|e| InterpError::Forward(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glassbox_runtime::CpuBackend;

    fn t(data: &[f32], shape: Shape) -> Tensor {
        Tensor::from_f32(data, shape).unwrap()
    }

    #[test]
    fn encode_relu_zeros_negative() {
        let d_in = 2;
        let d_features = 3;
        let w_enc = t(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], Shape::from([d_in, d_features]));
        let b_enc = t(&[0.0, 0.0, -10.0], Shape::from([d_features]));
        let w_dec = t(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], Shape::from([d_features, d_in]));
        let b_dec = t(&[0.0, 0.0], Shape::from([d_in]));
        let sae = SparseAutoencoder::new(d_in, d_features, w_enc, b_enc, w_dec, b_dec).unwrap();

        let x = t(&[1.0, 2.0], Shape::from([1, d_in]));
        let z = sae.encode(&CpuBackend, &x).unwrap();
        let z_data = z.as_f32().unwrap();
        assert!((z_data[0] - 1.0).abs() < 1e-5);
        assert!((z_data[1] - 2.0).abs() < 1e-5);
        assert_eq!(z_data[2], 0.0);
    }

    #[test]
    fn rejects_wrong_shape() {
        let bad_w = Tensor::from_f32(&[1.0, 2.0, 3.0], Shape::from([1, 3])).unwrap();
        let b_enc = Tensor::from_f32(&[0.0, 0.0, 0.0], Shape::from([3])).unwrap();
        let w_dec = Tensor::from_f32(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], Shape::from([3, 2])).unwrap();
        let b_dec = Tensor::from_f32(&[0.0, 0.0], Shape::from([2])).unwrap();
        let r = SparseAutoencoder::new(2, 3, bad_w, b_enc, w_dec, b_dec);
        assert!(r.is_err());
    }
}

pub fn top_k_features(features: &Tensor, k: usize) -> Result<Vec<(usize, f32)>> {
    let data = features.as_f32().map_err(|e| InterpError::Forward(e.to_string()))?;
    let d = features.shape().dim(features.rank() - 1).map_err(|e| InterpError::Forward(e.to_string()))?;
    let positions = data.len() / d;
    let mut summed = vec![0.0f32; d];
    for p in 0..positions {
        for i in 0..d {
            summed[i] += data[p * d + i];
        }
    }
    let mut indexed: Vec<(usize, f32)> = summed.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    Ok(indexed)
}
