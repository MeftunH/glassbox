pub use glassbox_runtime::{HookName, HookRegistry};

#[derive(Debug, Clone)]
pub struct HookSubscription {
    pub name: HookName,
}
