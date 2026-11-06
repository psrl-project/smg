pub mod engine;
pub mod error;
pub mod harness;
pub mod model_adapter;
pub mod normalizer;
pub mod sampler_distribution;
pub mod store;
pub mod validator;

pub use error::TitoError;
pub use harness::{build_tool_canonicalizer, ToolInputCanonicalizer};
pub use normalizer::{
    assistants_diagnostic_summary, finalize_hash, hash_message_into, hash_messages,
    hash_messages_with_context, messages_structure_summary, PrefixHash, PrefixHasher,
    RenderContext,
};
pub use store::{
    LookupMissReason, MismatchEntry, PrefixLookup, PrefixMatch, ResolvedTrajectoryId,
    TitoSessionData, TitoStore, Trajectory, TrajectoryIdReservation, TrajectoryIdStrategy,
    TrajectoryMeta, TurnCapture, TurnRecord, TurnRoutedExperts, TurnRoutedExpertsDtype,
};

/// HTTP header name for the TITO session identifier.
pub const TITO_SESSION_HEADER: &str = "x-smg-tito-session-id";

/// HTTP header name for the TITO trajectory identifier.
///
/// Used by the manual strategy and defaults to 0 when absent.
pub const TITO_TRAJECTORY_ID_HEADER: &str = "x-smg-tito-trajectory-id";

/// Normalized agent role for one turn.
pub const PSRL_AGENT_ROLE_HEADER: &str = "x-psrl-agent-role";

/// Stable agent identity for one turn.
pub const PSRL_AGENT_ID_HEADER: &str = "x-psrl-agent-id";

/// Parent agent identity for one turn.
pub const PSRL_PARENT_AGENT_ID_HEADER: &str = "x-psrl-parent-agent-id";

/// Per-turn training payload capture mode.
pub const PSRL_TITO_CAPTURE_HEADER: &str = "x-psrl-tito-capture";

pub use sampler_distribution::{validate_sampler_row, SamplerDistribution, ValidatedSamplerRow};
