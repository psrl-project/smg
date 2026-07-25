use thiserror::Error;

#[derive(Debug, Error)]
pub enum TitoError {
    #[error("appended messages contain an assistant turn, which is not allowed")]
    AssistantInAppended,
    #[error("incremental tokenization failed: {0}")]
    EngineFailed(String),
    #[error("automatic trajectory ID space is exhausted")]
    TrajectoryIdExhausted,
}
