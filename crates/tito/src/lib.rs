pub mod engine;
pub mod error;
pub mod harness;
pub mod model_adapter;
pub mod normalizer;
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
    MismatchEntry, PrefixLookup, PrefixMatch, ResolvedTrajectoryId, TitoSessionData, TitoStore,
    Trajectory, TrajectoryIdReservation, TrajectoryIdStrategy, TurnRecord, TurnRoutedExperts,
    TurnRoutedExpertsDtype,
};

/// HTTP header name for the TITO session identifier.
pub const TITO_SESSION_HEADER: &str = "x-smg-tito-session-id";

/// HTTP header name for the TITO trajectory identifier.
///
/// Used by the manual strategy and defaults to 0 when absent.
pub const TITO_TRAJECTORY_ID_HEADER: &str = "x-smg-tito-trajectory-id";
