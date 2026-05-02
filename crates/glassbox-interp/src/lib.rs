#![doc = include_str!("../README.md")]

pub mod attribution;
pub mod error;
pub mod hooks;
pub mod patching;
pub mod path_patching;
pub mod probes;
pub mod sae;

pub use attribution::DirectLogitAttribution;
pub use error::{InterpError, Result};
pub use hooks::{HookName, HookRegistry, HookSubscription};
pub use patching::{PatchSet, PatchTarget};
pub use path_patching::{run_path_patch, PathPatchResult, PathPatchSpec};
pub use probes::{NeuronProbe, NeuronProbeResult};
pub use sae::{top_k_features, SparseAutoencoder};
