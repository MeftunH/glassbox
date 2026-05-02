use std::io::Read;

use glassbox_core::{DType, Shape, Tensor};
use serde::{Deserialize, Serialize};

use crate::error::{ModelError, Result};

pub const GLX_MAGIC: &[u8; 4] = b"GLX1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlxTensorEntry {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub byte_len: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlxHeader {
    pub magic: String,
    pub version: u32,
    pub architecture: String,
    pub config: serde_json::Value,
    pub tokenizer_blob: Option<String>,
    pub tensors: Vec<GlxTensorEntry>,
}

#[derive(Debug)]
pub struct GlxFile {
    pub header: GlxHeader,
    pub payload: Vec<u8>,
}

impl GlxFile {
    pub fn read(mut reader: impl Read) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != GLX_MAGIC {
            return Err(ModelError::BadGlx(format!("bad magic: {magic:?}")));
        }

        let mut header_len_buf = [0u8; 8];
        reader.read_exact(&mut header_len_buf)?;
        let header_len = u64::from_le_bytes(header_len_buf) as usize;

        let mut header_bytes = vec![0u8; header_len];
        reader.read_exact(&mut header_bytes)?;
        let header: GlxHeader = serde_json::from_slice(&header_bytes)?;

        let mut payload = Vec::new();
        reader.read_to_end(&mut payload)?;

        Ok(Self { header, payload })
    }

    pub fn tensor(&self, name: &str) -> Result<Tensor> {
        let entry = self
            .header
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| ModelError::MissingTensor(name.into()))?;

        let start = entry.offset as usize;
        let end = start + entry.byte_len as usize;
        if end > self.payload.len() {
            return Err(ModelError::BadGlx(format!(
                "tensor `{name}` extends past payload: {end} > {}",
                self.payload.len()
            )));
        }
        let bytes = self.payload[start..end].to_vec();
        let storage = glassbox_core::Storage::cpu(std::sync::Arc::<[u8]>::from(bytes));
        Ok(Tensor::new(storage, Shape::from(entry.shape.clone()), entry.dtype))
    }
}
