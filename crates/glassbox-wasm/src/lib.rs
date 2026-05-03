use std::sync::Arc;

use glassbox_core::{Shape, Tensor};
use glassbox_interp::{run_path_patch, top_k_features, PathPatchResult, PathPatchSpec, SparseAutoencoder};
use glassbox_models::{Bpe, GlxFile, Gpt2, Gpt2Runner, Gpt2RunnerAsync};
use glassbox_runtime::{AsyncBackend, Backend, CpuBackend, HookRegistry, SamplingConfig, Sampler};
use glassbox_runtime::wgpu_backend::WgpuBackend;
use wasm_bindgen_futures::future_to_promise;
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

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateOutput {
    pub tokens: Vec<u32>,
    pub text: String,
    pub elapsed_ms: f64,
    pub tokens_per_second: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PathPatchArgs {
    pub clean_prompt: String,
    pub corrupt_prompt: String,
    pub sender_hook: String,
    pub receiver_hooks: Vec<String>,
    pub target_token: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PathPatchOut {
    pub clean_logit: f32,
    pub corrupt_logit: f32,
    pub patched_logit: f32,
    pub recovery: f32,
    pub elapsed_ms: f64,
}

impl From<PathPatchResult> for PathPatchOut {
    fn from(r: PathPatchResult) -> Self {
        Self {
            clean_logit: r.clean_logit,
            corrupt_logit: r.corrupt_logit,
            patched_logit: r.patched_logit,
            recovery: r.recovery,
            elapsed_ms: 0.0,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SaeShape {
    pub d_in: usize,
    pub d_features: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SaeFeatureSpike {
    pub feature: usize,
    pub activation: f32,
}

enum BackendChoice {
    Cpu(CpuBackend),
    Wgpu(WgpuBackend),
}

impl BackendChoice {
    fn as_backend(&self) -> &dyn Backend {
        match self {
            BackendChoice::Cpu(b) => b,
            BackendChoice::Wgpu(b) => b,
        }
    }

    fn as_async_backend(&self) -> &dyn AsyncBackend {
        match self {
            BackendChoice::Cpu(b) => b,
            BackendChoice::Wgpu(b) => b,
        }
    }
}

impl std::fmt::Debug for BackendChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BackendChoice({})", self.as_backend().name())
    }
}

#[wasm_bindgen]
#[derive(Debug)]
pub struct Glassbox {
    model: Arc<Gpt2>,
    tokenizer: Bpe,
    hooks: Arc<HookRegistry>,
    backend: Arc<BackendChoice>,
    saes: Arc<std::sync::Mutex<ahash::AHashMap<String, SparseAutoencoder>>>,
}

#[wasm_bindgen]
impl Glassbox {
    #[wasm_bindgen(js_name = fromBlob)]
    pub fn from_blob(blob: &[u8]) -> Result<Glassbox, JsValue> {
        let (model, tokenizer) = parse_glx(blob)?;
        Ok(Self {
            model: Arc::new(model),
            tokenizer,
            hooks: HookRegistry::new(),
            backend: Arc::new(BackendChoice::Cpu(CpuBackend)),
            saes: Arc::new(std::sync::Mutex::new(ahash::AHashMap::new())),
        })
    }

    #[wasm_bindgen(js_name = fromBlobWebGpu)]
    pub fn from_blob_webgpu(blob: js_sys::Uint8Array) -> js_sys::Promise {
        let bytes = blob.to_vec();
        future_to_promise(async move {
            let (model, tokenizer) = parse_glx(&bytes)?;
            let backend = WgpuBackend::new().await.map_err(jserr)?;
            let glass = Self {
                model: Arc::new(model),
                tokenizer,
                hooks: HookRegistry::new(),
                backend: Arc::new(BackendChoice::Wgpu(backend)),
                saes: Arc::new(std::sync::Mutex::new(ahash::AHashMap::new())),
            };
            Ok(JsValue::from(glass))
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

    #[wasm_bindgen(js_name = clearHooks)]
    pub fn clear_hooks(&self) {
        self.hooks.clear();
    }

    pub fn forward(&self, ids: &[u32]) -> Result<Vec<f32>, JsValue> {
        let runner = Gpt2Runner::new(&self.model, self.backend.as_backend(), Arc::clone(&self.hooks)).map_err(jserr)?;
        let logits = runner.forward(ids).map_err(jserr)?;
        runner.last_position_logits(&logits).map_err(jserr)
    }

    pub fn generate(&self, prompt: &str, max_new: u32, args: JsValue) -> Result<JsValue, JsValue> {
        let sampling: SamplingArgs = serde_wasm_bindgen::from_value(args)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let mut sampler = Sampler::new(sampling.into());
        let runner = Gpt2Runner::new(&self.model, self.backend.as_backend(), Arc::clone(&self.hooks)).map_err(jserr)?;

        let start = now_ms();
        let mut ids: Vec<u32> = self.tokenizer.encode(prompt);
        let mut generated: Vec<u32> = Vec::with_capacity(max_new as usize);
        for _ in 0..max_new {
            let logits = runner.forward(&ids).map_err(jserr)?;
            let last = runner.last_position_logits(&logits).map_err(jserr)?;
            let next = sampler.sample(&last);
            ids.push(next);
            generated.push(next);
        }
        let elapsed = now_ms() - start;
        let tps = if elapsed > 0.0 { (generated.len() as f64) / (elapsed / 1000.0) } else { 0.0 };

        let text = self.tokenizer.decode(&generated);
        let out = GenerateOutput { tokens: generated, text, elapsed_ms: elapsed, tokens_per_second: tps };
        serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = backendName)]
    pub fn backend_name(&self) -> String {
        self.backend.as_backend().name().to_string()
    }

    #[wasm_bindgen(js_name = generateAsync)]
    pub fn generate_async(&self, prompt: String, max_new: u32, args: JsValue) -> js_sys::Promise {
        let model = Arc::clone(&self.model);
        let tokenizer = self.tokenizer.clone();
        let hooks = Arc::clone(&self.hooks);
        let backend = Arc::clone(&self.backend);
        let sampling_args: SamplingArgs = match serde_wasm_bindgen::from_value(args) {
            Ok(v) => v,
            Err(e) => return js_sys::Promise::reject(&JsValue::from_str(&e.to_string())),
        };

        future_to_promise(async move {
            let mut sampler = Sampler::new(sampling_args.into());
            let runner = Gpt2RunnerAsync::new(&model, backend.as_async_backend(), hooks).map_err(jserr)?;

            let start = now_ms();
            let mut ids: Vec<u32> = tokenizer.encode(&prompt);
            let mut generated: Vec<u32> = Vec::with_capacity(max_new as usize);
            for _ in 0..max_new {
                let logits = runner.forward(&ids).await.map_err(jserr)?;
                let last = runner.last_position_logits(&logits).await.map_err(jserr)?;
                let next = sampler.sample(&last);
                ids.push(next);
                generated.push(next);
            }
            let elapsed = now_ms() - start;
            let tps = if elapsed > 0.0 { (generated.len() as f64) / (elapsed / 1000.0) } else { 0.0 };

            let text = tokenizer.decode(&generated);
            let out = GenerateOutput { tokens: generated, text, elapsed_ms: elapsed, tokens_per_second: tps };
            serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }

    #[wasm_bindgen(js_name = runPathPatch)]
    pub fn run_path_patch(&self, args: JsValue) -> Result<JsValue, JsValue> {
        let args: PathPatchArgs = serde_wasm_bindgen::from_value(args)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let clean_ids = self.tokenizer.encode(&args.clean_prompt);
        let corrupt_ids = self.tokenizer.encode(&args.corrupt_prompt);
        if clean_ids.len() != corrupt_ids.len() {
            return Err(JsValue::from_str(&format!(
                "clean and corrupt prompts must tokenise to the same length (got {} and {})",
                clean_ids.len(),
                corrupt_ids.len()
            )));
        }
        let spec = PathPatchSpec {
            clean_ids,
            corrupt_ids,
            sender_hook: args.sender_hook,
            receiver_hooks: args.receiver_hooks,
            target_token: args.target_token,
        };
        let start = now_ms();
        let result = run_path_patch(&self.model, self.backend.as_backend(), &spec).map_err(jserr)?;
        let mut out: PathPatchOut = result.into();
        out.elapsed_ms = now_ms() - start;
        serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = loadSae)]
    pub fn load_sae(
        &self,
        key: &str,
        d_in: usize,
        d_features: usize,
        w_enc: &[f32],
        b_enc: &[f32],
        w_dec: &[f32],
        b_dec: &[f32],
    ) -> Result<(), JsValue> {
        let w_enc = Tensor::from_f32(w_enc, Shape::from([d_in, d_features])).map_err(jserr)?;
        let b_enc = Tensor::from_f32(b_enc, Shape::from([d_features])).map_err(jserr)?;
        let w_dec = Tensor::from_f32(w_dec, Shape::from([d_features, d_in])).map_err(jserr)?;
        let b_dec = Tensor::from_f32(b_dec, Shape::from([d_in])).map_err(jserr)?;
        let sae = SparseAutoencoder::new(d_in, d_features, w_enc, b_enc, w_dec, b_dec).map_err(jserr)?;
        if let Ok(mut map) = self.saes.lock() {
            map.insert(key.into(), sae);
        }
        Ok(())
    }

    #[wasm_bindgen(js_name = encodeSaeFromHook)]
    pub fn encode_sae_from_hook(&self, sae_key: &str, hook: &str, top_k: usize) -> Result<JsValue, JsValue> {
        let snap = self.hooks.snapshot();
        let activation = snap
            .get(hook)
            .ok_or_else(|| JsValue::from_str(&format!("hook `{hook}` not captured")))?;

        let map = self.saes.lock().map_err(|_| JsValue::from_str("sae mutex poisoned"))?;
        let sae = map
            .get(sae_key)
            .ok_or_else(|| JsValue::from_str(&format!("sae `{sae_key}` not loaded")))?;

        let activation_2d = if activation.rank() == 2 {
            activation.clone()
        } else if activation.rank() == 3 {
            let dims = activation.shape().dims();
            activation.reshape([dims[0] * dims[1], dims[2]]).map_err(jserr)?
        } else {
            return Err(JsValue::from_str(&format!(
                "unsupported rank {} for sae input",
                activation.rank()
            )));
        };

        let features = sae.encode(self.backend.as_backend(), &activation_2d).map_err(jserr)?;
        let top = top_k_features(&features, top_k).map_err(jserr)?;
        let result: Vec<SaeFeatureSpike> = top
            .into_iter()
            .map(|(feature, activation)| SaeFeatureSpike { feature, activation })
            .collect();
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = installPatch)]
    pub fn install_patch(&self, hook: &str, data: &[f32], shape: &[u32]) -> Result<(), JsValue> {
        let shape: Vec<usize> = shape.iter().map(|&v| v as usize).collect();
        let tensor = Tensor::from_f32(data, Shape::from(shape)).map_err(jserr)?;
        self.hooks.install_patch(hook.to_string(), tensor);
        Ok(())
    }

    #[wasm_bindgen(js_name = clearPatches)]
    pub fn clear_patches(&self) {
        self.hooks.clear_patches();
    }
}

fn jserr(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

fn parse_glx(blob: &[u8]) -> Result<(Gpt2, Bpe), JsValue> {
    let file = GlxFile::read(blob).map_err(jserr)?;
    let model = Gpt2::from_glx(&file).map_err(jserr)?;
    let tokenizer_json = file.header.tokenizer_blob.clone().unwrap_or_default();
    let tokenizer = Bpe::from_json(&tokenizer_json).unwrap_or_else(|_| {
        Bpe::from_blob(glassbox_models::tokenizer::BpeBlob {
            vocab: ahash::AHashMap::new(),
            merges: Vec::new(),
        })
    });
    Ok((model, tokenizer))
}

fn now_ms() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0)
}
