use std::sync::Arc;

use glassbox_interp::HookRegistry;
use glassbox_models::{Bpe, GlxFile, Gpt2};
use glassbox_runtime::{Backend, CpuBackend, SamplingConfig, Sampler};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub architecture: String,
    pub vocab_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub parameter_count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SamplingArgs {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub seed: u64,
}

impl From<SamplingArgs> for SamplingConfig {
    fn from(a: SamplingArgs) -> Self {
        Self { temperature: a.temperature, top_k: a.top_k, top_p: a.top_p, seed: a.seed }
    }
}

#[wasm_bindgen]
pub struct Glassbox {
    model: Arc<Gpt2>,
    tokenizer: Bpe,
    hooks: Arc<HookRegistry>,
    backend: CpuBackend,
}

#[wasm_bindgen]
impl Glassbox {
    #[wasm_bindgen(js_name = fromBlob)]
    pub fn from_blob(blob: &[u8]) -> Result<Glassbox, JsValue> {
        let file = GlxFile::read(blob).map_err(jserr)?;
        let model = Gpt2::from_glx(&file).map_err(jserr)?;
        let tokenizer_json = file.header.tokenizer_blob.clone().unwrap_or_default();
        let tokenizer = Bpe::from_json(&tokenizer_json)
            .unwrap_or_else(|_| Bpe::from_blob(glassbox_models::tokenizer::BpeBlob {
                vocab: ahash::AHashMap::new(),
                merges: Vec::new(),
            }));
        Ok(Self {
            model: Arc::new(model),
            tokenizer,
            hooks: HookRegistry::new(),
            backend: CpuBackend,
        })
    }

    #[wasm_bindgen(js_name = modelInfo)]
    pub fn model_info(&self) -> Result<JsValue, JsValue> {
        let info = ModelInfo {
            architecture: self.model.config.architecture.clone(),
            vocab_size: self.model.config.vocab_size,
            n_layer: self.model.config.n_layer,
            n_head: self.model.config.n_head,
            n_embd: self.model.config.n_embd,
            parameter_count: self.model.parameter_count(),
        };
        serde_wasm_bindgen::to_value(&info).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids)
    }

    pub fn subscribe(&self, hook: &str) {
        self.hooks.subscribe(hook);
    }

    pub fn unsubscribe(&self, hook: &str) {
        self.hooks.unsubscribe(hook);
    }

    #[wasm_bindgen(js_name = readHook)]
    pub fn read_hook(&self, hook: &str) -> Option<Vec<f32>> {
        let snap = self.hooks.snapshot();
        let t = snap.get(hook)?;
        t.as_f32().ok().map(<[f32]>::to_vec)
    }

    pub fn sample(&self, logits: &[f32], args: JsValue) -> Result<u32, JsValue> {
        let args: SamplingArgs = serde_wasm_bindgen::from_value(args)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let mut sampler = Sampler::new(args.into());
        Ok(sampler.sample(logits))
    }

    #[wasm_bindgen(js_name = backendName)]
    pub fn backend_name(&self) -> String {
        Backend::name(&self.backend).to_string()
    }
}

fn jserr(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}
