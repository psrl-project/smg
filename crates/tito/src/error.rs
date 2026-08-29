use thiserror::Error;

#[derive(Debug, Error)]
pub enum TitoError {
    #[error("incremental tokenization failed: {0}")]
    EngineFailed(String),
    #[error("automatic trajectory ID space is exhausted")]
    TrajectoryIdExhausted,
    #[error("stored trajectory prefix diverges from the new checkpoint: {0}")]
    PrefixMismatch(String),
    #[error("tokenized full render does not start with its tokenized base ({path})")]
    TokenPrefixMismatch { path: &'static str },
    #[error("tokenizer config is missing an exact hf_model_type")]
    ModelTypeUnavailable,
    #[error("model type '{model_type}' requires tokenizer token '{token}', but it is missing")]
    RequiredTokenMissing {
        model_type: String,
        token: &'static str,
    },
    #[error("model type '{model_type}' requires '{token}' to encode to exactly one token")]
    RequiredTokenEncoding {
        model_type: String,
        token: &'static str,
    },
}
