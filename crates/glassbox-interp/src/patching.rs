use ahash::AHashMap;
use glassbox_core::Tensor;

#[derive(Debug, Clone)]
pub struct PatchTarget {
    pub hook: String,
    pub tensor: Tensor,
}

#[derive(Debug, Default, Clone)]
pub struct PatchSet {
    patches: AHashMap<String, Tensor>,
}

impl PatchSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, hook: impl Into<String>, tensor: Tensor) -> Self {
        self.patches.insert(hook.into(), tensor);
        self
    }

    pub fn add(&mut self, hook: impl Into<String>, tensor: Tensor) {
        self.patches.insert(hook.into(), tensor);
    }

    pub fn remove(&mut self, hook: &str) -> Option<Tensor> {
        self.patches.remove(hook)
    }

    pub fn get(&self, hook: &str) -> Option<&Tensor> {
        self.patches.get(hook)
    }

    pub fn len(&self) -> usize {
        self.patches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.patches.is_empty()
    }
}
