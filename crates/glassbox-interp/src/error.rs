pub type Result<T, E = InterpError> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum InterpError {
    #[error("forward failed: {0}")]
    Forward(String),

    #[error("missing hook `{0}`")]
    MissingHook(String),

    #[error("shape mismatch: {what} expected {expected}, got {got}")]
    ShapeMismatch {
        what: &'static str,
        expected: String,
        got: String,
    },
}
