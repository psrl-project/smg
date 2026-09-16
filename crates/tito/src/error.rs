use thiserror::Error;

#[derive(Debug, Error)]
pub enum TitoError {
    #[error("incremental tokenization failed: {0}")]
    EngineFailed(String),
    #[error("automatic trajectory ID space is exhausted")]
    TrajectoryIdExhausted,
    #[error("stored trajectory prefix diverges from the new checkpoint: {0}")]
    PrefixMismatch(String),
}
