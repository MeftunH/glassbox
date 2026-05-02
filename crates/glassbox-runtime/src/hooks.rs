use std::sync::{Arc, Mutex};

use ahash::AHashMap;
use glassbox_core::Tensor;

pub type HookName = String;

#[derive(Debug, Default)]
pub struct HookRegistry {
    captured: Mutex<AHashMap<HookName, Tensor>>,
    subscribed: Mutex<AHashMap<HookName, ()>>,
    patches: Mutex<AHashMap<HookName, Tensor>>,
}

impl HookRegistry {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn subscribe(&self, name: impl Into<HookName>) {
        let n = name.into();
        if let Ok(mut s) = self.subscribed.lock() {
            s.insert(n, ());
        }
    }

    pub fn unsubscribe(&self, name: &str) {
        if let Ok(mut s) = self.subscribed.lock() {
            s.remove(name);
        }
    }

    pub fn is_subscribed(&self, name: &str) -> bool {
        self.subscribed.lock().map(|s| s.contains_key(name)).unwrap_or(false)
    }

    pub fn publish(&self, name: &str, tensor: Tensor) {
        if !self.is_subscribed(name) {
            return;
        }
        if let Ok(mut c) = self.captured.lock() {
            c.insert(name.into(), tensor);
        }
    }

    pub fn install_patch(&self, name: impl Into<HookName>, tensor: Tensor) {
        if let Ok(mut p) = self.patches.lock() {
            p.insert(name.into(), tensor);
        }
    }

    pub fn remove_patch(&self, name: &str) -> Option<Tensor> {
        self.patches.lock().ok()?.remove(name)
    }

    pub fn patch(&self, name: &str) -> Option<Tensor> {
        self.patches.lock().ok()?.get(name).cloned()
    }

    pub fn take(&self, name: &str) -> Option<Tensor> {
        self.captured.lock().ok()?.remove(name)
    }

    pub fn snapshot(&self) -> AHashMap<HookName, Tensor> {
        self.captured.lock().map(|c| c.clone()).unwrap_or_default()
    }

    pub fn clear(&self) {
        if let Ok(mut c) = self.captured.lock() {
            c.clear();
        }
    }

    pub fn clear_patches(&self) {
        if let Ok(mut p) = self.patches.lock() {
            p.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glassbox_core::Shape;

    #[test]
    fn publish_only_when_subscribed() {
        let reg = HookRegistry::new();
        let t = Tensor::from_f32(&[1.0, 2.0], Shape::from([2])).unwrap();
        reg.publish("a", t.clone());
        assert!(reg.take("a").is_none());

        reg.subscribe("a");
        reg.publish("a", t);
        assert!(reg.take("a").is_some());
    }

    #[test]
    fn patches_round_trip() {
        let reg = HookRegistry::new();
        let t = Tensor::from_f32(&[1.0, 2.0], Shape::from([2])).unwrap();
        reg.install_patch("a", t.clone());
        assert!(reg.patch("a").is_some());
        assert!(reg.remove_patch("a").is_some());
        assert!(reg.patch("a").is_none());
    }
}
