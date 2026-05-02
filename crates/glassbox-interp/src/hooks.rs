use std::sync::{Arc, Mutex};

use ahash::AHashMap;
use glassbox_core::Tensor;

pub type HookName = String;

#[derive(Debug, Clone)]
pub struct HookSubscription {
    pub name: HookName,
}

#[derive(Debug, Default)]
pub struct HookRegistry {
    captured: Mutex<AHashMap<HookName, Tensor>>,
    subscribed: Mutex<AHashMap<HookName, ()>>,
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
}
