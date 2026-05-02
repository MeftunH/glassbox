#![doc = include_str!("../README.md")]

pub mod attribution;
pub mod hooks;
pub mod patching;
pub mod probes;

pub use attribution::DirectLogitAttribution;
pub use hooks::{HookName, HookRegistry, HookSubscription};
pub use patching::{PatchSet, PatchTarget};
pub use probes::{NeuronProbe, NeuronProbeResult};
