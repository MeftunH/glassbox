use glassbox_core::CoreError;
use glassbox_runtime::RuntimeError;

pub type Result<T, E = ModelError> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("core: {0}")]
    Core(#[from] CoreError),

    #[error("runtime: {0}")]
    Runtime(#[from] RuntimeError),

    #[error("io: {0}")]
    Io(String),

    #[error("malformed glx: {0}")]
    BadGlx(String),

    #[error("missing tensor `{0}`")]
    MissingTensor(String),

    #[error("unknown architecture `{0}`")]
    UnknownArchitecture(String),

    #[error("tokenizer: {0}")]
    Tokenizer(String),
}

impl From<std::io::Error> for ModelError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

impl From<serde_json::Error> for ModelError {
    fn from(e: serde_json::Error) -> Self {
        Self::BadGlx(e.to_string())
    }
}
