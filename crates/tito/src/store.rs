use std::{
    collections::{HashMap, HashSet},
    fmt,
    str::FromStr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use dashmap::DashMap;
use openai_protocol::chat::ChatMessage;
use serde::{Deserialize, Serialize};

use crate::{
    error::TitoError,
    normalizer::{
        finalize_hash, hash_message_into, hash_messages_with_context, initialize_context_hasher,
        PrefixHash, PrefixHasher, RenderContext,
    },
};

/// Controls how TITO assigns a trajectory identifier to each request.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TrajectoryIdStrategy {
    /// Read `x-smg-tito-trajectory-id` from the request (defaulting to 0).
    #[default]
    Manual,
    /// Continue a matching live leaf, or allocate the next session-local ID.
    Auto,
}

impl fmt::Display for TrajectoryIdStrategy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Manual => "manual",
            Self::Auto => "auto",
        })
    }
}

impl FromStr for TrajectoryIdStrategy {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "manual" => Ok(Self::Manual),
            "auto" => Ok(Self::Auto),
            _ => Err(format!(
                "invalid trajectory ID strategy '{value}'; expected manual or auto"
            )),
        }
    }
}

/// A content-addressed tree node stored per session.
pub struct PrefixEntry {
    /// Concatenation of prompt_token_ids + output_ids from the backend response.
    pub token_ids: Arc<Vec<u32>>,
    /// Same conversation prefix before multimodal anchor expansion, when it
    /// differs from `token_ids`. Pure-text entries reuse `token_ids` directly.
    pub reusable_prefix_token_ids: Option<Arc<Vec<u32>>>,
    /// Hash of the parent prefix (None for Turn 1 root children).
    pub parent_hash: Option<PrefixHash>,
    /// Metadata for the assistant turn that produced this node.
    pub turn_record: TurnRecord,
}

/// Token representations stored for one TITO node.
pub struct StoredTokenSequences {
    expanded: Vec<u32>,
    reusable: Option<Vec<u32>>,
}

impl StoredTokenSequences {
    fn shared(token_ids: Vec<u32>) -> Self {
        Self {
            expanded: token_ids,
            reusable: None,
        }
    }

    /// Keep a distinct reusable sequence only when multimodal expansion
    /// actually changed the token IDs.
    pub fn with_reusable(expanded: Vec<u32>, reusable: Vec<u32>) -> Self {
        let reusable = (expanded != reusable).then_some(reusable);
        Self { expanded, reusable }
    }
}

/// Internal session state managed behind a Mutex.
pub(crate) struct SessionState {
    pub entries: HashMap<PrefixHash, PrefixEntry>,
    /// Set of hashes that are currently leaf nodes (have no children stored yet).
    ///
    /// A hash is a leaf when it has been stored but no child node has been added yet.
    /// When a new child is stored the parent is removed from this set.
    /// This is kept in sync with `trajectory_leaves` via GC: after every `store` call,
    /// any hash present in `leaf_hashes` that is **not** pointed to by any live
    /// trajectory is removed (together with unreachable ancestors).
    pub leaf_hashes: HashSet<PrefixHash>,
    /// Maps `trajectory_id → current leaf hash` for each live trajectory in this session.
    ///
    /// In manual mode IDs are caller-supplied (0 by default); in auto mode they are
    /// assigned from prefix matches. Each `store` advances the selected pointer to
    /// the newly stored node. After updating, any `leaf_hashes` entry no longer
    /// reachable from this map is eligible for GC.
    pub trajectory_leaves: HashMap<u64, PrefixHash>,
    /// Per-trajectory cross-turn `routed_experts_prompt_start` offset.
    pub trajectory_re_offsets: HashMap<u64, u32>,
    /// IDs selected by requests that have not completed storage yet.
    pub inflight_auto_trajectory_ids: HashSet<u64>,
    /// Maximum number of trailing boundary tokens that may be trimmed per non-last turn
    /// during training data construction.  `0` means no trimming is allowed (identity adapter
    /// such as `DefaultAdapter`).  Set once from the model adapter; `1` for Qwen3 and GLM4.7.
    pub max_trim_tokens: usize,
    /// Minimum number of entries that must be present in the session before GC is
    /// triggered. `0` means "always run GC".
    pub gc_threshold: usize,
}

impl SessionState {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            leaf_hashes: HashSet::new(),
            trajectory_leaves: HashMap::new(),
            trajectory_re_offsets: HashMap::new(),
            inflight_auto_trajectory_ids: HashSet::new(),
            max_trim_tokens: 0,
            gc_threshold: 0,
        }
    }
}

/// Per-turn training data record.
#[derive(Clone, Debug, Serialize)]
pub struct TurnRecord {
    pub prompt_token_count: usize,
    pub output_logprobs: Option<Vec<(f32, u32)>>,
    pub finish_reason: String,
    pub mismatch_report: Vec<MismatchEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub routed_experts: Option<TurnRoutedExperts>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight_version: Option<String>,
}

/// Compact NumPy-style dtype descriptor for TITO-tracked routed-experts.
/// Mirrors the gateway-side `RoutedExpertsDtype` enum so converting between
/// the two is a `match`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum TurnRoutedExpertsDtype {
    U8,
    U16,
}

impl TurnRoutedExpertsDtype {
    /// Bytes per element.
    pub const fn size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
        }
    }

    /// String form mirroring `numpy.dtype.str` (used in JSON exports).
    pub const fn wire_str(self) -> &'static str {
        match self {
            Self::U8 => "uint8",
            Self::U16 => "uint16",
        }
    }

    /// Corresponding `.npy` writer dtype.
    const fn as_npy(self) -> openai_protocol::npy::NpyDtype {
        match self {
            Self::U8 => openai_protocol::npy::NpyDtype::U8,
            Self::U16 => openai_protocol::npy::NpyDtype::U16,
        }
    }
}

/// Routed-experts payload attached to a single turn record.
#[derive(Clone, Debug)]
pub struct TurnRoutedExperts {
    pub data: Arc<Vec<u8>>,
    pub num_layers: u32,
    pub top_k: u32,
    pub dtype: TurnRoutedExpertsDtype,
    pub prompt_start: u32,
}

impl TurnRoutedExperts {
    /// Bytes per token (`num_layers * top_k * dtype.size()`).
    const fn token_bytes(&self) -> usize {
        self.num_layers as usize * self.top_k as usize * self.dtype.size()
    }

    /// Number of tokens held in `data`.
    fn num_tokens(&self) -> usize {
        match self.token_bytes() {
            0 => 0,
            row => self.data.len() / row,
        }
    }
}

impl Serialize for TurnRoutedExperts {
    /// Emit `{data: base64(.npy), num_layers, top_k, dtype, prompt_start}`, or
    /// `null` when no tokens were captured (degenerate blobs help no consumer).
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use base64::Engine as _;
        use serde::ser::SerializeStruct;

        let tokens = self.num_tokens();
        if tokens == 0 {
            return serializer.serialize_none();
        }

        let shape = [
            tokens as u64,
            u64::from(self.num_layers),
            u64::from(self.top_k),
        ];
        let npy = openai_protocol::npy::encode_npy(&shape, self.dtype.as_npy(), &self.data);
        let data_b64 = base64::engine::general_purpose::STANDARD.encode(npy);

        let mut state = serializer.serialize_struct("TurnRoutedExperts", 5)?;
        state.serialize_field("data", &data_b64)?;
        state.serialize_field("num_layers", &self.num_layers)?;
        state.serialize_field("top_k", &self.top_k)?;
        state.serialize_field("dtype", self.dtype.wire_str())?;
        state.serialize_field("prompt_start", &self.prompt_start)?;
        state.end()
    }
}

/// A single mismatch between TITO accumulated tokens and canonical retokenization.
#[derive(Clone, Debug, Serialize)]
pub struct MismatchEntry {
    pub mismatch_type: String,
    pub position: usize,
    pub detail: String,
}

/// A complete training trajectory collected from a session leaf.
#[derive(Clone, Debug, Serialize)]
pub struct Trajectory {
    /// Resolved trajectory identifier (header-selected or automatically assigned).
    pub trajectory_id: u64,
    /// Full token ID sequence (prompt_ids + all output_ids concatenated in conversation order).
    pub accumulated_token_ids: Vec<u32>,
    /// One `TurnRecord` per assistant turn, ordered from oldest to newest.
    #[serde(rename = "records")]
    pub turn_records: Vec<TurnRecord>,
}

/// Consistent training-data snapshot returned for every TITO session.
#[derive(Debug, Serialize)]
pub struct TitoSessionData {
    pub session_id: String,
    pub max_trim_tokens: usize,
    /// Always an array: empty, single-trajectory, and multi-trajectory sessions
    /// share the same wire format.
    pub trajectories: Vec<Trajectory>,
}

/// Top-level store: session_id → Arc<Mutex<SessionState>>.
pub struct TitoStore {
    sessions: DashMap<String, Arc<parking_lot::Mutex<SessionState>>>,
    debug: AtomicBool,
    gc_threshold: std::sync::atomic::AtomicUsize,
    trajectory_id_strategy: TrajectoryIdStrategy,
}

/// Result of resolving a request's trajectory identity.
pub struct ResolvedTrajectoryId {
    pub trajectory_id: u64,
    /// Keeps an automatic ID reserved until the request stores or is dropped.
    pub reservation: Option<TrajectoryIdReservation>,
}

/// Request-scoped reservation preventing concurrent branches from sharing an ID.
pub struct TrajectoryIdReservation {
    state: Arc<parking_lot::Mutex<SessionState>>,
    trajectory_id: u64,
    active: AtomicBool,
}

impl TrajectoryIdReservation {
    /// Release the ID as soon as response storage has completed.
    pub fn release(&self) {
        if self.active.swap(false, Ordering::AcqRel) {
            self.state
                .lock()
                .inflight_auto_trajectory_ids
                .remove(&self.trajectory_id);
        }
    }
}

impl Drop for TrajectoryIdReservation {
    fn drop(&mut self) {
        self.release();
    }
}

/// Result of a successful prefix lookup.
pub struct PrefixMatch {
    pub pretokenized_ids: Vec<u32>,
    /// How many messages from the start were matched (messages[..matched_len] is cached).
    pub matched_message_num: usize,
}

/// `running_hasher` is the [`PrefixHasher`] state after folding both the
/// render context and every message in the lookup `messages` slice.
///
/// `parent_hash` is the hash at the **last assistant boundary** in the
/// lookup `messages` slice — i.e. the hash of `messages[..=last_assistant]`.
/// When the caller appends the new assistant message and stores it, this is
/// exactly the parent of the new node in the prefix tree.  `None` means the
/// lookup messages contained no assistant turn at all (root-level store).
pub struct PrefixLookup {
    /// Pretokenized prefix on a cache hit; `None` on miss.
    pub matched: Option<PrefixMatch>,
    /// Hasher state after folding `(render_context, messages...)`.
    /// Clone-and-extend with the new assistant message to derive the leaf hash.
    pub running_hasher: PrefixHasher,
    /// Hash at the last assistant boundary in the lookup `messages`.  This is
    /// the parent hash for any node about to be stored after this lookup.
    pub parent_hash: Option<PrefixHash>,
}

impl Default for TitoStore {
    fn default() -> Self {
        Self::new()
    }
}

impl TitoStore {
    pub fn new() -> Self {
        Self::with_trajectory_id_strategy(TrajectoryIdStrategy::default())
    }

    pub fn with_trajectory_id_strategy(strategy: TrajectoryIdStrategy) -> Self {
        Self {
            sessions: DashMap::new(),
            debug: AtomicBool::new(false),
            gc_threshold: std::sync::atomic::AtomicUsize::new(0),
            trajectory_id_strategy: strategy,
        }
    }

    pub const fn trajectory_id_strategy(&self) -> TrajectoryIdStrategy {
        self.trajectory_id_strategy
    }

    /// Resolve the trajectory for a request after prefix lookup.
    ///
    /// Manual mode returns the caller-provided value. Auto mode continues an
    /// unclaimed trajectory whose current leaf equals `matched_parent_hash`;
    /// otherwise it allocates the next session-local ID.
    pub fn resolve_trajectory_id(
        &self,
        session_id: &str,
        manual_trajectory_id: u64,
        matched_parent_hash: Option<PrefixHash>,
    ) -> Result<ResolvedTrajectoryId, TitoError> {
        if self.trajectory_id_strategy == TrajectoryIdStrategy::Manual {
            return Ok(ResolvedTrajectoryId {
                trajectory_id: manual_trajectory_id,
                reservation: None,
            });
        }

        let state = self.get_or_create_session_arc(session_id);
        let trajectory_id = {
            let mut session = state.lock();
            let matching_id = matched_parent_hash.and_then(|parent_hash| {
                session
                    .trajectory_leaves
                    .iter()
                    .filter(|(trajectory_id, leaf_hash)| {
                        **leaf_hash == parent_hash
                            && !session.inflight_auto_trajectory_ids.contains(trajectory_id)
                    })
                    .map(|(trajectory_id, _)| *trajectory_id)
                    .min()
            });

            let trajectory_id = match matching_id {
                Some(trajectory_id) => trajectory_id,
                None => {
                    let mut candidate = 0;
                    loop {
                        if !session.trajectory_leaves.contains_key(&candidate)
                            && !session.inflight_auto_trajectory_ids.contains(&candidate)
                        {
                            break candidate;
                        }
                        candidate = candidate
                            .checked_add(1)
                            .ok_or(TitoError::TrajectoryIdExhausted)?;
                    }
                }
            };
            session.inflight_auto_trajectory_ids.insert(trajectory_id);
            trajectory_id
        };

        Ok(ResolvedTrajectoryId {
            trajectory_id,
            reservation: Some(TrajectoryIdReservation {
                state,
                trajectory_id,
                active: AtomicBool::new(true),
            }),
        })
    }

    /// Enable or disable TITO mismatch validation (debug/development only).
    pub fn set_debug(&self, enabled: bool) {
        self.debug.store(enabled, Ordering::Release);
    }

    /// Returns `true` if TITO debug (mismatch validation) is enabled.
    pub fn is_debug(&self) -> bool {
        self.debug.load(Ordering::Acquire)
    }

    /// Set the default GC threshold applied to all **newly created** sessions.
    pub fn set_gc_threshold(&self, threshold: usize) {
        self.gc_threshold.store(threshold, Ordering::Release);
    }

    /// Returns the default GC threshold used for newly created sessions.
    pub fn gc_threshold(&self) -> usize {
        self.gc_threshold.load(Ordering::Acquire)
    }

    /// Look up a session and return a cloned Arc to its state, or `None` if the
    /// session does not exist.  The DashMap shard lock is dropped before returning.
    #[inline]
    fn get_session_arc(&self, session_id: &str) -> Option<Arc<parking_lot::Mutex<SessionState>>> {
        self.sessions.get(session_id).map(|g| Arc::clone(&*g))
    }

    /// Look up or lazily create a session and return a cloned Arc to its state.
    /// The DashMap shard lock is dropped before returning.
    #[inline]
    fn get_or_create_session_arc(&self, session_id: &str) -> Arc<parking_lot::Mutex<SessionState>> {
        let threshold = self.gc_threshold.load(Ordering::Acquire);
        Arc::clone(
            &*self
                .sessions
                .entry(session_id.to_owned())
                .or_insert_with(|| {
                    let mut state = SessionState::new();
                    state.gc_threshold = threshold;
                    Arc::new(parking_lot::Mutex::new(state))
                }),
        )
    }

    /// Create a new session with the given session ID.
    pub fn create_session(&self, session_id: &str) {
        let threshold = self.gc_threshold.load(Ordering::Acquire);
        self.sessions
            .entry(session_id.to_owned())
            .or_insert_with(|| {
                let mut state = SessionState::new();
                state.gc_threshold = threshold;
                Arc::new(parking_lot::Mutex::new(state))
            });
    }

    /// Check whether a session has been created.
    pub fn session_exists(&self, session_id: &str) -> bool {
        self.sessions.contains_key(session_id)
    }

    /// Delete a session and all its state.  Idempotent.
    pub fn delete_session(&self, session_id: &str) {
        self.sessions.remove(session_id);
    }

    /// Set the max_trim_tokens ceiling for a session.
    pub fn set_session_max_trim_tokens(&self, session_id: &str, max_trim: usize) {
        if let Some(arc) = self.get_session_arc(session_id) {
            arc.lock().max_trim_tokens = max_trim;
        }
    }

    /// Get the max_trim_tokens for a session (0 if session not found).
    pub fn get_session_max_trim_tokens(&self, session_id: &str) -> usize {
        match self.get_session_arc(session_id) {
            Some(arc) => arc.lock().max_trim_tokens,
            None => 0,
        }
    }

    /// Override the GC threshold for a specific session.
    pub fn set_session_gc_threshold(&self, session_id: &str, threshold: usize) {
        if let Some(arc) = self.get_session_arc(session_id) {
            arc.lock().gc_threshold = threshold;
        }
    }

    /// Get the current GC threshold for a session (0 if session not found).
    pub fn get_session_gc_threshold(&self, session_id: &str) -> usize {
        match self.get_session_arc(session_id) {
            Some(arc) => arc.lock().gc_threshold,
            None => 0,
        }
    }

    /// Look up the longest cached prefix for `messages`.
    pub fn find_prefix(
        &self,
        session_id: &str,
        messages: &[ChatMessage],
        render_context: &RenderContext,
    ) -> Result<Option<PrefixMatch>, TitoError> {
        Ok(self
            .find_prefix_with_lookup(session_id, messages, render_context)?
            .matched)
    }

    /// Look up the longest cached prefix for `messages` and emit hash-chain
    /// state usable by [`Self::store_with_hashes`].
    ///
    /// Returns `Err(TitoError::AssistantInAppended)` if a HIT candidate would
    /// require an assistant turn inside the appended slice (client sequencing
    /// bug).
    ///
    /// On a miss, the running hasher and parent hash are still populated,
    /// so the caller can store a root node for the session without paying
    /// for a second full message walk.
    pub fn find_prefix_with_lookup(
        &self,
        session_id: &str,
        messages: &[ChatMessage],
        render_context: &RenderContext,
    ) -> Result<PrefixLookup, TitoError> {
        let mut hasher = initialize_context_hasher(render_context);
        let mut candidates: Vec<(usize, PrefixHash)> = Vec::new();
        let mut parent_hash: Option<PrefixHash> = None;

        for (i, msg) in messages.iter().enumerate() {
            hash_message_into(&mut hasher, msg);
            let k = i + 1;
            if is_assistant_role(msg) {
                let h = finalize_hash(&hasher);
                parent_hash = Some(h);
                if k < messages.len() {
                    candidates.push((k, h));
                }
            }
        }

        // messages too short to possibly cover a cached prefix.
        // We still return the running hasher so the caller
        // can store a root-level entry.
        if messages.len() < 2 || candidates.is_empty() {
            tracing::debug!(
                session_id = %session_id,
                msg_count = messages.len(),
                candidate_count = candidates.len(),
                "find_prefix_with_lookup: no candidates"
            );
            return Ok(PrefixLookup {
                matched: None,
                running_hasher: hasher,
                parent_hash,
            });
        }

        tracing::debug!(
            session_id = %session_id,
            candidate_count = candidates.len(),
            entries_count = self
                .get_session_arc(session_id)
                .map(|arc| arc.lock().entries.len())
                .unwrap_or(0),
            "find_prefix_with_lookup: lookup start"
        );

        // Check from longest prefix to shortest. `get_session_arc` ensures the
        // DashMap shard lock is released before we lock the per-session Mutex.
        let arc = match self.get_session_arc(session_id) {
            Some(arc) => arc,
            None => {
                tracing::debug!(session_id = %session_id, "find_prefix_with_lookup: session not found");
                return Ok(PrefixLookup {
                    matched: None,
                    running_hasher: hasher,
                    parent_hash,
                });
            }
        };
        let state = arc.lock();

        for (k, hash) in candidates.iter().rev() {
            if let Some(entry) = state.entries.get(hash) {
                let appended = &messages[*k..];
                if appended.iter().any(is_assistant_role) {
                    return Err(TitoError::AssistantInAppended);
                }
                tracing::debug!(
                    session_id = %session_id,
                    matched_len = *k,
                    prefix_tokens = entry.token_ids.len(),
                    "find_prefix_with_lookup: HIT"
                );
                let reusable_prefix_token_ids = entry
                    .reusable_prefix_token_ids
                    .as_deref()
                    .unwrap_or(entry.token_ids.as_ref());
                return Ok(PrefixLookup {
                    matched: Some(PrefixMatch {
                        pretokenized_ids: reusable_prefix_token_ids.clone(),
                        matched_message_num: *k,
                    }),
                    running_hasher: hasher,
                    parent_hash,
                });
            }
        }

        Ok(PrefixLookup {
            matched: None,
            running_hasher: hasher,
            parent_hash,
        })
    }

    /// Store token IDs for a completed generation.
    pub fn store(
        &self,
        session_id: &str,
        messages: &[ChatMessage],
        token_ids: Vec<u32>,
        turn_record: TurnRecord,
        render_context: &RenderContext,
        trajectory_id: u64,
    ) -> Result<(), TitoError> {
        let leaf_hash = hash_messages_with_context(messages, render_context);
        let parent_hash = compute_parent_hash(messages, render_context);
        self.store_with_hashes(
            session_id,
            leaf_hash,
            parent_hash,
            token_ids,
            turn_record,
            trajectory_id,
        )
    }

    /// Store token IDs for a completed generation using caller-supplied hashes.
    ///
    /// `leaf_hash` must equal `hash_messages_with_context(all_messages,
    /// render_context)` where `all_messages` is the full conversation
    /// including the new assistant turn.  `parent_hash` must equal the
    /// `hash_messages_with_context` of the prefix that ends at the
    /// second-to-last assistant turn (or `None` for a root-level node).
    /// Callers that obtain these hashes from [`PrefixLookup`] satisfy this
    /// invariant by construction; other callers should prefer the higher-level
    /// [`Self::store`] which derives the hashes itself.
    pub fn store_with_hashes(
        &self,
        session_id: &str,
        leaf_hash: PrefixHash,
        parent_hash: Option<PrefixHash>,
        token_ids: Vec<u32>,
        turn_record: TurnRecord,
        trajectory_id: u64,
    ) -> Result<(), TitoError> {
        self.store_token_sequences(
            session_id,
            leaf_hash,
            parent_hash,
            StoredTokenSequences::shared(token_ids),
            turn_record,
            trajectory_id,
        )
    }

    /// Store the expanded training sequence and the unexpanded reusable TITO
    /// prefix as separate representations of the same conversation node.
    pub fn store_with_hashes_and_reusable(
        &self,
        session_id: &str,
        leaf_hash: PrefixHash,
        parent_hash: Option<PrefixHash>,
        token_ids: StoredTokenSequences,
        turn_record: TurnRecord,
        trajectory_id: u64,
    ) -> Result<(), TitoError> {
        self.store_token_sequences(
            session_id,
            leaf_hash,
            parent_hash,
            token_ids,
            turn_record,
            trajectory_id,
        )
    }

    fn store_token_sequences(
        &self,
        session_id: &str,
        leaf_hash: PrefixHash,
        parent_hash: Option<PrefixHash>,
        token_ids: StoredTokenSequences,
        turn_record: TurnRecord,
        trajectory_id: u64,
    ) -> Result<(), TitoError> {
        let arc = self.get_or_create_session_arc(session_id);
        let mut state = arc.lock();

        // Leaf tracking: the parent is no longer a leaf once we add a child.
        if let Some(ph) = parent_hash {
            state.leaf_hashes.remove(&ph);
        }
        state.leaf_hashes.insert(leaf_hash);

        state.entries.insert(
            leaf_hash,
            PrefixEntry {
                token_ids: Arc::new(token_ids.expanded),
                reusable_prefix_token_ids: token_ids.reusable.map(Arc::new),
                parent_hash,
                turn_record,
            },
        );

        // Advance the trajectory pointer for this trajectory_id to the newly stored hash.
        state.trajectory_leaves.insert(trajectory_id, leaf_hash);

        // GC: remove leaves (and their unreachable ancestors) that no live trajectory
        // points to.  We compute the set of hashes reachable from all trajectory
        // pointers (transitively through parent_hash chains) and prune everything else.
        gc_unreachable(&mut state);

        Ok(())
    }

    /// Return all trajectories rooted at current leaf nodes.
    ///
    /// Each [`Trajectory`] contains the full accumulated token sequence for that
    /// leaf and all [`TurnRecord`]s from root to leaf in conversation order.
    ///
    /// Trajectories are returned sorted by [`Trajectory::trajectory_id`] ascending so
    /// that callers receive a deterministic, ordered list suitable for training pipelines.
    pub fn get_all_trajectories(&self, session_id: &str) -> Vec<Trajectory> {
        let arc = match self.get_session_arc(session_id) {
            Some(arc) => arc,
            None => return Vec::new(),
        };
        let state = arc.lock();

        collect_trajectories(&state)
    }

    /// Return an atomic session snapshot with a stable multi-trajectory shape.
    ///
    /// The session existence check, metadata read, and trajectory collection all
    /// happen under one lock so callers cannot observe a mixed snapshot.
    pub fn get_session_data(&self, session_id: &str) -> Option<TitoSessionData> {
        let arc = self.get_session_arc(session_id)?;
        let state = arc.lock();

        Some(TitoSessionData {
            session_id: session_id.to_string(),
            max_trim_tokens: state.max_trim_tokens,
            trajectories: collect_trajectories(&state),
        })
    }

    /// Look up the next-turn dispatch's `routed_experts_prompt_start` for
    /// the given trajectory.  Returns 0 when the (session, trajectory) pair
    /// is new — turn 1 captures the full prompt by default.
    ///
    /// Called by chat preparation before dispatching turn k.  The store
    /// returns an *advisory* value; the caller may still override it with
    /// a partial-rollout-injected loopback offset.
    pub fn next_routed_experts_prompt_start(&self, session_id: &str, trajectory_id: u64) -> u32 {
        let Some(arc) = self.get_session_arc(session_id) else {
            return 0;
        };
        let state = arc.lock();
        state
            .trajectory_re_offsets
            .get(&trajectory_id)
            .copied()
            .unwrap_or(0)
    }

    /// Record the position upper-bound captured by this turn so the next
    /// turn can pick up where it left off.
    pub fn advance_routed_experts_offset(
        &self,
        session_id: &str,
        trajectory_id: u64,
        captured_upper_bound: u32,
    ) {
        let arc = self
            .sessions
            .entry(session_id.to_string())
            .or_insert_with(|| Arc::new(parking_lot::Mutex::new(SessionState::new())))
            .clone();
        let mut state = arc.lock();
        let slot = state
            .trajectory_re_offsets
            .entry(trajectory_id)
            .or_insert(0);
        *slot = captured_upper_bound;
    }
}

/// Compute the hash of the prefix that ends at the second-to-last assistant turn.
/// Returns None if `messages` has fewer than two assistant turns (i.e., this is a root node).
fn compute_parent_hash(
    messages: &[ChatMessage],
    render_context: &RenderContext,
) -> Option<PrefixHash> {
    // Find the index of the last assistant turn in this message list.
    let last_asst = messages.iter().rposition(is_assistant_role)?;
    // Find the second-to-last assistant turn (the one before last_asst).
    // The parent's message list ends with that assistant turn.
    let second_last_asst = messages[..last_asst].iter().rposition(is_assistant_role)?;
    // Parent hash = hash of messages[0..=second_last_asst] with the same render context.
    Some(hash_messages_with_context(
        &messages[..=second_last_asst],
        render_context,
    ))
}

/// Garbage-collect all entries (and leaf-hash records) that are no longer reachable
/// from any live trajectory pointer in `state.trajectory_leaves`.
///
/// ## Full GC algorithm (O(|entries| + |trajectory_leaves|))
///
/// 1. Walk the ancestor chain for every trajectory leaf pointer, collecting all
///    reachable hashes into a `HashSet`.
/// 2. Drain every entry whose hash is **not** in the reachable set from
///    `state.entries`.
/// 3. Rebuild `state.leaf_hashes` as the intersection of the old leaf set with the
///    reachable set — removing any dead leaves.
fn gc_unreachable(state: &mut SessionState) {
    if state.trajectory_leaves.is_empty() {
        // No live trajectory pointers — remove everything.
        state.entries.clear();
        state.leaf_hashes.clear();
        return;
    }

    // If the number of entries is below the threshold, skip GC.
    if state.gc_threshold > 0 && state.entries.len() <= state.gc_threshold {
        return;
    }

    // If all leaves are reachable from trajectories, skip GC.
    if state.leaf_hashes.len() <= state.trajectory_leaves.len() {
        return;
    }

    // Step 1: collect all reachable hashes by walking ancestor chains.
    let mut reachable: HashSet<PrefixHash> =
        HashSet::with_capacity(state.trajectory_leaves.len() * 4);

    for &leaf_hash in state.trajectory_leaves.values() {
        let mut current = Some(leaf_hash);
        while let Some(hash) = current {
            if !reachable.insert(hash) {
                // Already visited this node and all its ancestors — short-circuit.
                break;
            }
            current = state.entries.get(&hash).and_then(|e| e.parent_hash);
        }
    }

    // Step 2: remove unreachable entries.
    state.entries.retain(|hash, _| reachable.contains(hash));

    // Step 3: prune leaf_hashes to reachable only.
    state.leaf_hashes.retain(|hash| reachable.contains(hash));
}

fn collect_records_for_leaf(state: &SessionState, leaf_hash: PrefixHash) -> Vec<TurnRecord> {
    let mut hashes = Vec::new();
    let mut current = Some(leaf_hash);

    while let Some(hash) = current {
        hashes.push(hash);
        current = state.entries.get(&hash).and_then(|entry| entry.parent_hash);
    }

    hashes.reverse();
    hashes
        .into_iter()
        .filter_map(|hash| {
            state
                .entries
                .get(&hash)
                .map(|entry| entry.turn_record.clone())
        })
        .collect()
}

fn collect_trajectories(state: &SessionState) -> Vec<Trajectory> {
    let mut pairs: Vec<(u64, PrefixHash)> = state
        .trajectory_leaves
        .iter()
        .map(|(&trajectory_id, &leaf_hash)| (trajectory_id, leaf_hash))
        .collect();
    pairs.sort_unstable_by_key(|(trajectory_id, _)| *trajectory_id);

    pairs
        .into_iter()
        .filter_map(|(trajectory_id, leaf_hash)| {
            let entry = state.entries.get(&leaf_hash)?;
            Some(Trajectory {
                trajectory_id,
                accumulated_token_ids: (*entry.token_ids).clone(),
                turn_records: collect_records_for_leaf(state, leaf_hash),
            })
        })
        .collect()
}

fn is_assistant_role(msg: &ChatMessage) -> bool {
    matches!(msg, ChatMessage::Assistant { .. })
}

#[cfg(test)]
mod tests {
    use openai_protocol::chat::{ChatMessage, MessageContent};

    use super::*;

    fn user_msg(content: &str) -> ChatMessage {
        ChatMessage::User {
            content: MessageContent::Text(content.to_string()),
            name: None,
        }
    }

    fn assistant_msg(content: &str) -> ChatMessage {
        ChatMessage::Assistant {
            content: Some(MessageContent::Text(content.to_string())),
            name: None,
            tool_calls: None,
            reasoning_content: None,
        }
    }

    fn tool_msg(content: &str, call_id: &str) -> ChatMessage {
        ChatMessage::Tool {
            content: MessageContent::Text(content.to_string()),
            tool_call_id: call_id.to_string(),
        }
    }

    fn make_store() -> TitoStore {
        TitoStore::new()
    }

    fn make_auto_store() -> TitoStore {
        TitoStore::with_trajectory_id_strategy(TrajectoryIdStrategy::Auto)
    }

    fn render_context() -> RenderContext {
        RenderContext::default()
    }

    fn record(prompt_token_count: usize, finish_reason: &str) -> TurnRecord {
        TurnRecord {
            prompt_token_count,
            output_logprobs: None,
            finish_reason: finish_reason.to_string(),
            mismatch_report: vec![],
            routed_experts: None,
            weight_version: None,
        }
    }

    #[test]
    fn manual_strategy_preserves_header_selected_id() {
        let store = make_store();
        assert_eq!(store.trajectory_id_strategy(), TrajectoryIdStrategy::Manual);

        let resolved = store.resolve_trajectory_id("s1", 42, None).unwrap();
        assert_eq!(resolved.trajectory_id, 42);
        assert!(resolved.reservation.is_none());
    }

    #[test]
    fn trajectory_id_strategy_rejects_unknown_value() {
        assert_eq!(
            "auto".parse::<TrajectoryIdStrategy>().unwrap(),
            TrajectoryIdStrategy::Auto
        );
        assert!("random".parse::<TrajectoryIdStrategy>().is_err());
    }

    #[test]
    fn auto_strategy_reuses_id_for_single_linear_trajectory() {
        let store = make_auto_store();
        let context = render_context();
        let root = vec![user_msg("hi"), assistant_msg("hello")];
        let root_hash = hash_messages_with_context(&root, &context);

        let first = store.resolve_trajectory_id("s1", 99, None).unwrap();
        assert_eq!(first.trajectory_id, 0, "auto mode ignores the header ID");
        store
            .store(
                "s1",
                &root,
                vec![1, 2],
                record(1, "root"),
                &context,
                first.trajectory_id,
            )
            .unwrap();
        first.reservation.as_ref().unwrap().release();

        let next = store
            .resolve_trajectory_id("s1", 99, Some(root_hash))
            .unwrap();
        assert_eq!(next.trajectory_id, 0);
    }

    #[test]
    fn auto_strategy_allocates_next_id_when_branching_from_internal_node() {
        let store = make_auto_store();
        let context = render_context();
        let root = vec![user_msg("hi"), assistant_msg("hello")];
        let root_hash = hash_messages_with_context(&root, &context);
        let branch_a = vec![
            user_msg("hi"),
            assistant_msg("hello"),
            user_msg("path A"),
            assistant_msg("answer A"),
        ];

        let root_id = store.resolve_trajectory_id("s1", 0, None).unwrap();
        store
            .store(
                "s1",
                &root,
                vec![1, 2],
                record(1, "root"),
                &context,
                root_id.trajectory_id,
            )
            .unwrap();
        root_id.reservation.as_ref().unwrap().release();

        let first_branch = store
            .resolve_trajectory_id("s1", 0, Some(root_hash))
            .unwrap();
        assert_eq!(first_branch.trajectory_id, 0);
        store
            .store(
                "s1",
                &branch_a,
                vec![1, 2, 3, 4],
                record(3, "branch A"),
                &context,
                first_branch.trajectory_id,
            )
            .unwrap();
        first_branch.reservation.as_ref().unwrap().release();

        let second_branch = store
            .resolve_trajectory_id("s1", 0, Some(root_hash))
            .unwrap();
        assert_eq!(second_branch.trajectory_id, 1);
    }

    #[test]
    fn auto_strategy_reserves_distinct_ids_for_concurrent_leaf_branches() {
        let store = make_auto_store();
        let context = render_context();
        let root = vec![user_msg("hi"), assistant_msg("hello")];
        let root_hash = hash_messages_with_context(&root, &context);

        let root_id = store.resolve_trajectory_id("s1", 0, None).unwrap();
        store
            .store(
                "s1",
                &root,
                vec![1, 2],
                record(1, "root"),
                &context,
                root_id.trajectory_id,
            )
            .unwrap();
        root_id.reservation.as_ref().unwrap().release();

        let first = store
            .resolve_trajectory_id("s1", 0, Some(root_hash))
            .unwrap();
        let second = store
            .resolve_trajectory_id("s1", 0, Some(root_hash))
            .unwrap();
        assert_eq!(first.trajectory_id, 0);
        assert_eq!(second.trajectory_id, 1);
    }

    #[test]
    fn auto_strategy_allocates_consecutive_ids_for_new_inflight_trajectories() {
        let store = make_auto_store();
        let reservations: Vec<_> = (0..4)
            .map(|_| store.resolve_trajectory_id("s1", 99, None).unwrap())
            .collect();
        let trajectory_ids: Vec<_> = reservations
            .iter()
            .map(|resolved| resolved.trajectory_id)
            .collect();

        assert_eq!(trajectory_ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn auto_strategy_reuses_unstored_id_after_reservation_is_dropped() {
        let store = make_auto_store();

        let abandoned = store.resolve_trajectory_id("s1", 0, None).unwrap();
        assert_eq!(abandoned.trajectory_id, 0);
        drop(abandoned);

        let retry = store.resolve_trajectory_id("s1", 0, None).unwrap();
        assert_eq!(retry.trajectory_id, 0);
    }

    fn store_turn(
        store: &TitoStore,
        session_id: &str,
        messages: &[ChatMessage],
        token_ids: Vec<u32>,
        turn_record: TurnRecord,
        render_context: &RenderContext,
    ) {
        store
            .store(
                session_id,
                messages,
                token_ids,
                turn_record,
                render_context,
                0,
            )
            .unwrap();
    }

    #[test]
    fn find_prefix_returns_none_when_empty() {
        let store = make_store();
        store.create_session("s1");
        let msgs = vec![user_msg("hi"), assistant_msg("hello")];
        assert!(store
            .find_prefix("s1", &msgs, &render_context())
            .unwrap()
            .is_none());
    }

    #[test]
    fn store_then_find_returns_hit() {
        let store = make_store();
        store.create_session("s1");
        let ctx = render_context();
        let msgs = vec![user_msg("hi"), assistant_msg("hello")];
        let ids = vec![1u32, 2, 3];
        store_turn(&store, "s1", &msgs, ids.clone(), record(2, "stop"), &ctx);
        let query = vec![
            user_msg("hi"),
            assistant_msg("hello"),
            tool_msg("result", "call_1"),
        ];
        let hit = store.find_prefix("s1", &query, &ctx).unwrap().unwrap();
        assert_eq!(hit.pretokenized_ids, ids);
        assert_eq!(hit.matched_message_num, 2);
    }

    #[test]
    fn find_prefix_short_messages_returns_none() {
        let store = make_store();
        store.create_session("s1");
        let msgs = vec![user_msg("hi")];
        assert!(store
            .find_prefix("s1", &msgs, &render_context())
            .unwrap()
            .is_none());
    }

    #[test]
    fn find_prefix_assistant_in_appended_returns_error() {
        let store = make_store();
        store.create_session("s1");
        let ctx = render_context();
        let prefix = vec![user_msg("hi"), assistant_msg("hello")];
        store_turn(
            &store,
            "s1",
            &prefix,
            vec![1, 2, 3],
            record(2, "stop"),
            &ctx,
        );
        let msgs = vec![
            user_msg("hi"),
            assistant_msg("hello"),
            tool_msg("result", "call_1"),
            assistant_msg("again"),
        ];
        assert!(matches!(
            store.find_prefix("s1", &msgs, &ctx),
            Err(TitoError::AssistantInAppended)
        ));
    }

    #[test]
    fn delete_session_is_idempotent() {
        let store = make_store();
        store.create_session("s1");
        store.delete_session("s1");
        store.delete_session("s1");
    }

    #[test]
    fn find_prefix_incremental_finds_longest_match() {
        let store = make_store();
        store.create_session("s1");
        let ctx = render_context();

        let turn1 = vec![user_msg("hi"), assistant_msg("hello")];
        let turn2 = vec![
            user_msg("hi"),
            assistant_msg("hello"),
            user_msg("more"),
            assistant_msg("yes"),
        ];

        store_turn(&store, "s1", &turn1, vec![10, 20], record(2, "turn1"), &ctx);
        store_turn(
            &store,
            "s1",
            &turn2,
            vec![10, 20, 30, 40],
            record(4, "turn2"),
            &ctx,
        );

        let query = vec![
            user_msg("hi"),
            assistant_msg("hello"),
            user_msg("more"),
            assistant_msg("yes"),
            user_msg("final"),
        ];
        let hit = store.find_prefix("s1", &query, &ctx).unwrap().unwrap();
        assert_eq!(hit.matched_message_num, 4);
        assert_eq!(hit.pretokenized_ids, vec![10, 20, 30, 40]);
    }

    #[test]
    fn get_all_trajectories_returns_leaf_tokens_and_parent_records() {
        let store = make_store();
        store.create_session("s1");
        let ctx = render_context();

        let turn1 = vec![user_msg("a"), assistant_msg("b")];
        let turn2 = vec![
            user_msg("a"),
            assistant_msg("b"),
            user_msg("c"),
            assistant_msg("d"),
        ];

        store_turn(&store, "s1", &turn1, vec![1, 2], record(2, "turn1"), &ctx);
        store_turn(
            &store,
            "s1",
            &turn2,
            vec![1, 2, 3, 4],
            record(4, "turn2"),
            &ctx,
        );

        let trajectories = store.get_all_trajectories("s1");
        assert_eq!(trajectories.len(), 1);
        assert_eq!(trajectories[0].accumulated_token_ids, vec![1, 2, 3, 4]);
        assert_eq!(trajectories[0].turn_records.len(), 2);
        assert_eq!(trajectories[0].turn_records[0].finish_reason, "turn1");
        assert_eq!(trajectories[0].turn_records[1].finish_reason, "turn2");
    }

    #[test]
    fn session_data_uses_one_shape_for_empty_single_and_multiple_trajectories() {
        let context = render_context();

        let empty_store = make_store();
        empty_store.create_session("empty");
        let empty = serde_json::to_value(empty_store.get_session_data("empty").unwrap()).unwrap();
        assert_eq!(
            empty,
            serde_json::json!({
                "session_id": "empty",
                "max_trim_tokens": 0,
                "trajectories": [],
            })
        );
        assert!(empty_store.get_session_data("missing").is_none());

        let single_store = make_store();
        single_store.create_session("single");
        single_store
            .store(
                "single",
                &[user_msg("one"), assistant_msg("answer one")],
                vec![1, 2],
                record(1, "single"),
                &context,
                0,
            )
            .unwrap();
        let single =
            serde_json::to_value(single_store.get_session_data("single").unwrap()).unwrap();
        assert_eq!(single["trajectories"].as_array().unwrap().len(), 1);
        assert!(single.get("trajectory_id").is_none());
        assert!(single.get("accumulated_token_ids").is_none());
        assert!(single.get("records").is_none());
        assert_eq!(single["trajectories"][0]["trajectory_id"], 0);
        assert!(single["trajectories"][0].get("records").is_some());
        assert!(single["trajectories"][0].get("turn_records").is_none());

        let multi_store = make_store();
        multi_store.create_session("multi");
        multi_store
            .store(
                "multi",
                &[user_msg("zero"), assistant_msg("answer zero")],
                vec![1, 2],
                record(1, "zero"),
                &context,
                0,
            )
            .unwrap();
        multi_store
            .store(
                "multi",
                &[user_msg("one"), assistant_msg("answer one")],
                vec![3, 4],
                record(1, "one"),
                &context,
                1,
            )
            .unwrap();
        let multi = serde_json::to_value(multi_store.get_session_data("multi").unwrap()).unwrap();
        assert_eq!(multi["trajectories"].as_array().unwrap().len(), 2);
        assert_eq!(multi["trajectories"][0]["trajectory_id"], 0);
        assert_eq!(multi["trajectories"][1]["trajectory_id"], 1);
    }

    #[test]
    fn sequential_store_leaf_is_only_latest() {
        let store = make_store();
        store.create_session("s1");
        let ctx = render_context();

        let turn1 = vec![user_msg("hi"), assistant_msg("hello")];
        let turn2 = vec![
            user_msg("hi"),
            assistant_msg("hello"),
            user_msg("more"),
            assistant_msg("yes"),
        ];
        let hash_turn2 = hash_messages_with_context(&turn2, &ctx);

        store_turn(&store, "s1", &turn1, vec![1, 2], record(2, "turn1"), &ctx);
        store_turn(
            &store,
            "s1",
            &turn2,
            vec![1, 2, 3, 4],
            record(4, "turn2"),
            &ctx,
        );

        let arc = Arc::clone(&*store.sessions.get("s1").unwrap());
        let state = arc.lock();
        assert_eq!(state.leaf_hashes.len(), 1);
        assert!(state.leaf_hashes.contains(&hash_turn2));
    }

    #[test]
    fn retry_branching_returns_two_trajectories() {
        let store = make_store();
        store.create_session("s1");
        let ctx = render_context();

        let turn1 = vec![user_msg("hi"), assistant_msg("hello")];
        let branch_a = vec![
            user_msg("hi"),
            assistant_msg("hello"),
            user_msg("path A"),
            assistant_msg("answer A"),
        ];
        let branch_b = vec![
            user_msg("hi"),
            assistant_msg("hello"),
            user_msg("path B"),
            assistant_msg("answer B"),
        ];

        // Turn 1 is the shared root for trajectory 0.
        store_turn(&store, "s1", &turn1, vec![1, 2], record(2, "root"), &ctx);
        // Branch A extends trajectory 0.
        store_turn(
            &store,
            "s1",
            &branch_a,
            vec![1, 2, 3, 4],
            record(4, "branch_a"),
            &ctx,
        );
        // Branch B is a separate trajectory (id=1) rooted at the same turn1.
        store
            .store(
                "s1",
                &branch_b,
                vec![1, 2, 5, 6],
                record(4, "branch_b"),
                &ctx,
                1,
            )
            .unwrap();

        let trajectories = store.get_all_trajectories("s1");
        assert_eq!(trajectories.len(), 2);

        // Sorted by trajectory_id: 0 first, then 1.
        assert_eq!(trajectories[0].trajectory_id, 0);
        assert_eq!(trajectories[1].trajectory_id, 1);

        let mut all_ids: Vec<Vec<u32>> = trajectories
            .into_iter()
            .map(|t| t.accumulated_token_ids)
            .collect();
        all_ids.sort();
        assert_eq!(all_ids[0], vec![1, 2, 3, 4]);
        assert_eq!(all_ids[1], vec![1, 2, 5, 6]);
    }

    #[test]
    fn render_context_is_part_of_cache_key() {
        let store = make_store();
        store.create_session("s1");
        let msgs = vec![user_msg("hi"), assistant_msg("hello")];
        let ctx_a = RenderContext::new(Some(vec![serde_json::json!({"name":"tool_a"})]), None);
        let ctx_b = RenderContext::new(Some(vec![serde_json::json!({"name":"tool_b"})]), None);

        store_turn(
            &store,
            "s1",
            &msgs,
            vec![1, 2, 3],
            record(2, "stop"),
            &ctx_a,
        );
        let query = vec![
            user_msg("hi"),
            assistant_msg("hello"),
            tool_msg("result", "call_1"),
        ];

        assert!(store.find_prefix("s1", &query, &ctx_b).unwrap().is_none());
        assert!(store.find_prefix("s1", &query, &ctx_a).unwrap().is_some());
    }

    #[test]
    fn replacing_same_entry_replaces_turn_record() {
        let store = make_store();
        store.create_session("s1");
        let ctx = render_context();
        let msgs = vec![user_msg("hi"), assistant_msg("hello")];

        store_turn(&store, "s1", &msgs, vec![1, 2], record(2, "old"), &ctx);
        store_turn(&store, "s1", &msgs, vec![1, 2, 3], record(3, "new"), &ctx);

        let trajectories = store.get_all_trajectories("s1");
        assert_eq!(trajectories.len(), 1);
        assert_eq!(trajectories[0].accumulated_token_ids, vec![1, 2, 3]);
        assert_eq!(trajectories[0].turn_records.len(), 1);
        assert_eq!(trajectories[0].turn_records[0].finish_reason, "new");
    }

    #[test]
    fn delete_session_cleans_entries_and_records() {
        let store = make_store();
        store.create_session("s1");
        let ctx = render_context();
        let msgs = vec![user_msg("hi"), assistant_msg("hello")];
        store_turn(&store, "s1", &msgs, vec![1, 2, 3], record(3, "stop"), &ctx);
        assert_eq!(store.get_all_trajectories("s1").len(), 1);
        store.delete_session("s1");
        assert_eq!(store.get_all_trajectories("s1").len(), 0);
    }

    #[test]
    fn node_record_carries_logprobs_and_mismatch_report() {
        let store = make_store();
        store.create_session("s1");
        let ctx = render_context();
        let msgs = vec![user_msg("hi"), assistant_msg("hello")];
        store
            .store(
                "s1",
                &msgs,
                vec![1, 2, 3],
                TurnRecord {
                    prompt_token_count: 42,
                    output_logprobs: Some(vec![(-0.5, 100), (-1.2, 200)]),
                    finish_reason: "stop".to_string(),
                    mismatch_report: vec![MismatchEntry {
                        mismatch_type: "token_diff".to_string(),
                        position: 3,
                        detail: "expected 10, got 11".to_string(),
                    }],
                    routed_experts: None,
                    weight_version: None,
                },
                &ctx,
                0,
            )
            .unwrap();

        let trajectories = store.get_all_trajectories("s1");
        let record = &trajectories[0].turn_records[0];
        assert_eq!(record.prompt_token_count, 42);
        assert_eq!(record.finish_reason, "stop");
        assert_eq!(
            record.output_logprobs.as_ref().unwrap(),
            &vec![(-0.5, 100), (-1.2, 200)]
        );
        assert_eq!(record.mismatch_report[0].position, 3);
    }

    #[test]
    fn debug_flag_defaults_false() {
        let store = TitoStore::new();
        assert!(!store.is_debug(), "debug should default to false");
    }

    #[test]
    fn debug_flag_set_true() {
        let store = TitoStore::new();
        store.set_debug(true);
        assert!(
            store.is_debug(),
            "debug should be true after set_debug(true)"
        );
    }

    #[test]
    fn debug_flag_set_false_after_true() {
        let store = TitoStore::new();
        store.set_debug(true);
        store.set_debug(false);
        assert!(
            !store.is_debug(),
            "debug should be false after set_debug(false)"
        );
    }

    // ── GC threshold tests ──────────────────────────────────────────────────

    /// When the threshold is 0 (default), GC always runs and orphaned entries are removed.
    #[test]
    fn gc_runs_by_default_threshold_zero() {
        let store = make_store();
        store.create_session("s1");
        let ctx = render_context();

        // Store two turns on trajectory 0 (linear chain).
        let turn1 = vec![user_msg("hi"), assistant_msg("a")];
        let turn2 = vec![
            user_msg("hi"),
            assistant_msg("a"),
            user_msg("next"),
            assistant_msg("b"),
        ];
        store_turn(&store, "s1", &turn1, vec![1, 2], record(2, "t1"), &ctx);
        store_turn(
            &store,
            "s1",
            &turn2,
            vec![1, 2, 3, 4],
            record(4, "t2"),
            &ctx,
        );

        // After both stores the first turn must have been GC'd (not a leaf anymore
        // and only reachable via the second turn's parent chain — which is still live,
        // so actually the parent entry is retained).  What we care about here is that
        // the store has exactly 2 entries (both turns are on the live chain).
        let arc = Arc::clone(&*store.sessions.get("s1").unwrap());
        let state = arc.lock();
        // Both entries are reachable from trajectory 0; neither should be removed.
        assert_eq!(state.entries.len(), 2, "both turns should be retained");
        // Only the leaf of trajectory 0 should be in leaf_hashes.
        assert_eq!(state.leaf_hashes.len(), 1);
    }

    /// When gc_threshold is set to N, GC is skipped while entries ≤ N.
    #[test]
    fn gc_skipped_below_threshold() {
        let store = make_store();
        // Set a very large threshold so GC never triggers in this test.
        store.set_gc_threshold(1000);
        store.create_session("s1");
        let ctx = render_context();

        // Store turn 1 on trajectory 0.
        let turn1 = vec![user_msg("hi"), assistant_msg("v1")];
        store_turn(&store, "s1", &turn1, vec![1, 2], record(2, "t1"), &ctx);

        // Now store the same turn again under a different trajectory so a new
        // entry is added but the old one would normally be garbage-collected if
        // trajectory 0 had moved on.  Here we keep trajectory 0 pointing at turn1
        // and add trajectory 1 pointing at an independent turn1-variant.
        let turn1_b = vec![user_msg("hi"), assistant_msg("v1")];
        store
            .store("s1", &turn1_b, vec![10, 20], record(2, "t1b"), &ctx, 1)
            .unwrap();

        // Because threshold > entries, GC is skipped and both entries survive.
        let arc = Arc::clone(&*store.sessions.get("s1").unwrap());
        let state = arc.lock();
        // The two "different" stores wrote to the same hash (same messages),
        // so there is only 1 unique entry; both trajectory pointers reference it.
        // The point is that no panic / incorrect pruning occurred.
        assert!(
            !state.entries.is_empty(),
            "entries must survive when below threshold"
        );
    }

    /// When gc_threshold is 0 and only one trajectory exists, the balanced-leaf
    /// fast-path fires: no orphaned nodes → GC body is skipped (observable via
    /// the absence of extra allocations; we verify correctness, not internals).
    #[test]
    fn balanced_fast_path_single_trajectory_correctness() {
        let store = make_store();
        store.create_session("s1");
        let ctx = render_context();

        let turn1 = vec![user_msg("a"), assistant_msg("b")];
        let turn2 = vec![
            user_msg("a"),
            assistant_msg("b"),
            user_msg("c"),
            assistant_msg("d"),
        ];

        store_turn(&store, "s1", &turn1, vec![1, 2], record(2, "t1"), &ctx);
        store_turn(
            &store,
            "s1",
            &turn2,
            vec![1, 2, 3, 4],
            record(4, "t2"),
            &ctx,
        );

        // After the second store trajectory 0 points at turn2; turn1 is its ancestor.
        // The fast-path should detect that leaf_hashes == {turn2_hash} ⊆ live trajectory
        // values, avoid the full walk, and leave both entries intact (they're on the
        // live chain).
        let arc = Arc::clone(&*store.sessions.get("s1").unwrap());
        let state = arc.lock();
        assert_eq!(state.entries.len(), 2);
        assert_eq!(state.leaf_hashes.len(), 1);
    }

    /// When a session has two trajectories sharing a root, one trajectory
    /// advancing should not collect the root shared with the other.
    #[test]
    fn gc_retains_shared_ancestor_across_trajectories() {
        let store = make_store();
        store.create_session("s1");
        let ctx = render_context();

        let shared_root = vec![user_msg("root"), assistant_msg("shared")];
        let branch_a = vec![
            user_msg("root"),
            assistant_msg("shared"),
            user_msg("q"),
            assistant_msg("a1"),
        ];
        let branch_b = vec![
            user_msg("root"),
            assistant_msg("shared"),
            user_msg("q"),
            assistant_msg("a2"),
        ];

        // Trajectory 0 stores root then advances to branch_a.
        store_turn(
            &store,
            "s1",
            &shared_root,
            vec![1, 2],
            record(2, "root"),
            &ctx,
        );
        store_turn(
            &store,
            "s1",
            &branch_a,
            vec![1, 2, 3, 4],
            record(4, "a1"),
            &ctx,
        );

        // Trajectory 1 advances independently to branch_b.
        store
            .store("s1", &branch_b, vec![1, 2, 5, 6], record(4, "a2"), &ctx, 1)
            .unwrap();

        let arc = Arc::clone(&*store.sessions.get("s1").unwrap());
        let state = arc.lock();
        // shared_root + branch_a + branch_b = 3 entries, all reachable.
        assert_eq!(state.entries.len(), 3, "shared root must not be GC'd");
    }

    /// gc_threshold is propagated to sessions created after the call.
    #[test]
    fn gc_threshold_propagates_to_new_sessions() {
        let store = make_store();
        store.set_gc_threshold(42);
        store.create_session("t1");
        assert_eq!(store.get_session_gc_threshold("t1"), 42);
    }

    /// set_session_gc_threshold overrides the per-session value independently.
    #[test]
    fn set_session_gc_threshold_overrides_default() {
        let store = make_store();
        store.set_gc_threshold(10);
        store.create_session("t1");
        store.set_session_gc_threshold("t1", 99);
        assert_eq!(store.get_session_gc_threshold("t1"), 99);
        // A second session still gets the default.
        store.create_session("t2");
        assert_eq!(store.get_session_gc_threshold("t2"), 10);
    }

    // -- Hash-reuse contract tests ---------------------------------------------
    //
    // The chat path obtains the leaf and parent hashes from `PrefixLookup`
    // rather than walking the message slice twice.  These tests pin down that
    // those reused hashes are byte-identical to what the legacy code path
    // (`hash_messages_with_context` + `compute_parent_hash`) would have
    // produced, so the on-disk prefix tree stays compatible.

    fn extend_for_assistant(lookup: &PrefixLookup, assistant: &ChatMessage) -> PrefixHash {
        let mut hasher = lookup.running_hasher.clone();
        hash_message_into(&mut hasher, assistant);
        finalize_hash(&hasher)
    }

    #[test]
    fn lookup_running_hasher_matches_full_hash_no_assistant() {
        let store = make_store();
        let ctx = render_context();
        let request_msgs = vec![user_msg("hi")];
        let new_assistant = assistant_msg("hello");

        let mut all_msgs = request_msgs.clone();
        all_msgs.push(new_assistant.clone());

        let lookup = store
            .find_prefix_with_lookup("s1", &request_msgs, &ctx)
            .unwrap();
        let reused_leaf = extend_for_assistant(&lookup, &new_assistant);
        let canonical_leaf = hash_messages_with_context(&all_msgs, &ctx);
        assert_eq!(reused_leaf, canonical_leaf);
        // No prior assistant in request → root node.
        assert!(lookup.parent_hash.is_none());
    }

    #[test]
    fn lookup_parent_hash_matches_compute_parent_hash() {
        let store = make_store();
        let ctx = render_context();
        let request_msgs = vec![
            user_msg("a"),
            assistant_msg("b"),
            user_msg("c"),
            assistant_msg("d"),
            user_msg("e"),
        ];
        let new_assistant = assistant_msg("f");

        let mut all_msgs = request_msgs.clone();
        all_msgs.push(new_assistant.clone());

        let lookup = store
            .find_prefix_with_lookup("s1", &request_msgs, &ctx)
            .unwrap();
        // Leaf hash via reused hasher must equal the canonical one-shot hash.
        let reused_leaf = extend_for_assistant(&lookup, &new_assistant);
        assert_eq!(reused_leaf, hash_messages_with_context(&all_msgs, &ctx));
        // Parent hash via lookup must equal the legacy compute_parent_hash.
        assert_eq!(
            lookup.parent_hash,
            compute_parent_hash(&all_msgs, &ctx),
            "parent_hash from lookup must match legacy compute_parent_hash"
        );
    }

    #[test]
    fn lookup_parent_hash_handles_assistant_at_end_of_request() {
        // Pathological: client sent a request whose final message is an
        // assistant.  Legacy `compute_parent_hash` finds the assistant inside
        // request.messages as the second-to-last assistant in all_messages
        // (where all_messages = request + new_assistant).  Our running
        // parent_hash must match that for the on-disk tree to stay coherent.
        let store = make_store();
        let ctx = render_context();
        let request_msgs = vec![
            user_msg("a"),
            assistant_msg("b"),
            user_msg("c"),
            assistant_msg("d"),
        ];
        let new_assistant = assistant_msg("e");

        let mut all_msgs = request_msgs.clone();
        all_msgs.push(new_assistant.clone());

        let lookup = store
            .find_prefix_with_lookup("s1", &request_msgs, &ctx)
            .unwrap();
        assert_eq!(lookup.parent_hash, compute_parent_hash(&all_msgs, &ctx));
    }

    #[test]
    fn store_with_hashes_matches_store() {
        // Same input through both paths must produce identical tree state.
        let ctx = render_context();
        let request_msgs = vec![user_msg("a"), assistant_msg("b"), user_msg("c")];
        let new_assistant = assistant_msg("d");

        let mut all_msgs = request_msgs.clone();
        all_msgs.push(new_assistant.clone());

        // Reference: legacy `store` derives both hashes itself.
        let reference = make_store();
        reference.create_session("s");
        reference
            .store("s", &all_msgs, vec![1, 2, 3], record(3, "stop"), &ctx, 42)
            .unwrap();

        // Reused: caller passes hashes obtained from the lookup.
        let reused = make_store();
        reused.create_session("s");
        let lookup = reused
            .find_prefix_with_lookup("s", &request_msgs, &ctx)
            .unwrap();
        let leaf_hash = extend_for_assistant(&lookup, &new_assistant);
        reused
            .store_with_hashes(
                "s",
                leaf_hash,
                lookup.parent_hash,
                vec![1, 2, 3],
                record(3, "stop"),
                42,
            )
            .unwrap();

        // Both stores should have identical trajectory output.
        let trajs_ref = reference.get_all_trajectories("s");
        let trajs_new = reused.get_all_trajectories("s");
        assert_eq!(trajs_ref.len(), trajs_new.len());
        let r = &trajs_ref[0];
        let n = &trajs_new[0];
        assert_eq!(r.trajectory_id, n.trajectory_id);
        assert_eq!(r.accumulated_token_ids, n.accumulated_token_ids);
        assert_eq!(r.turn_records.len(), n.turn_records.len());

        // Pure-text nodes keep one token buffer: incremental lookup falls
        // back to the expanded/training sequence because both are identical.
        {
            let session = reused
                .get_session_arc("s")
                .expect("the text session should exist");
            let state = session.lock();
            let entry = state
                .entries
                .get(&leaf_hash)
                .expect("the text prefix should be stored");
            assert!(entry.reusable_prefix_token_ids.is_none());
        }

        let mut next_request = all_msgs;
        next_request.push(user_msg("e"));
        let matched = reused
            .find_prefix_with_lookup("s", &next_request, &ctx)
            .unwrap()
            .matched
            .expect("the text prefix should remain reusable");
        assert_eq!(matched.pretokenized_ids, vec![1, 2, 3]);
    }

    #[test]
    fn stored_token_sequences_deduplicates_identical_reusable_ids() {
        let token_ids = StoredTokenSequences::with_reusable(vec![1, 2, 3], vec![1, 2, 3]);

        assert_eq!(token_ids.expanded, vec![1, 2, 3]);
        assert!(token_ids.reusable.is_none());
    }

    #[test]
    fn multimodal_prefix_reuses_unexpanded_ids_but_exports_expanded_trajectory() {
        let store = make_store();
        let ctx = render_context();
        let first_turn = vec![user_msg("<image> describe"), assistant_msg("cat")];
        let leaf_hash = hash_messages_with_context(&first_turn, &ctx);
        store
            .store_with_hashes_and_reusable(
                "mm-session",
                leaf_hash,
                None,
                StoredTokenSequences::with_reusable(
                    vec![10, 99, 99, 99, 20], // training sequence
                    vec![10, 42, 20],         // one media anchor
                ),
                record(5, "stop"),
                0,
            )
            .unwrap();

        {
            let session = store
                .get_session_arc("mm-session")
                .expect("the multimodal session should exist");
            let state = session.lock();
            let entry = state
                .entries
                .get(&leaf_hash)
                .expect("the multimodal prefix should be stored");
            assert!(entry.reusable_prefix_token_ids.is_some());
        }

        let next_request = vec![
            user_msg("<image> describe"),
            assistant_msg("cat"),
            user_msg("and this <image>?"),
        ];
        let lookup = store
            .find_prefix_with_lookup("mm-session", &next_request, &ctx)
            .unwrap();
        let matched = lookup
            .matched
            .expect("the first multimodal turn should match");
        assert_eq!(matched.pretokenized_ids, vec![10, 42, 20]);

        let trajectories = store.get_all_trajectories("mm-session");
        assert_eq!(
            trajectories[0].accumulated_token_ids,
            vec![10, 99, 99, 99, 20]
        );
    }

    #[test]
    fn find_prefix_compatibility_wrapper_returns_same_match() {
        // The legacy `find_prefix(...) -> Option<PrefixMatch>` API must keep
        // returning exactly what it used to: HIT data when the prefix exists,
        // None otherwise.  We rely on this for back-compat with callers that
        // haven't migrated to `find_prefix_with_lookup`.
        let store = make_store();
        let ctx = render_context();
        let prefix = vec![user_msg("hi"), assistant_msg("hello")];
        store_turn(
            &store,
            "s1",
            &prefix,
            vec![1, 2, 3],
            record(2, "stop"),
            &ctx,
        );
        let query = vec![user_msg("hi"), assistant_msg("hello"), user_msg("more")];

        let legacy_hit = store.find_prefix("s1", &query, &ctx).unwrap();
        let lookup = store.find_prefix_with_lookup("s1", &query, &ctx).unwrap();
        assert!(legacy_hit.is_some());
        assert!(lookup.matched.is_some());
        let legacy = legacy_hit.unwrap();
        let new = lookup.matched.unwrap();
        assert_eq!(legacy.pretokenized_ids, new.pretokenized_ids);
        assert_eq!(legacy.matched_message_num, new.matched_message_num);
    }

    #[test]
    fn routed_experts_serializes_as_base64_npy_blob() {
        use base64::Engine as _;

        // 2 rows × 2 layers × 3 top_k = 12 uint8 bytes.
        let re = TurnRoutedExperts {
            data: Arc::new((0u8..12).collect()),
            num_layers: 2,
            top_k: 3,
            dtype: TurnRoutedExpertsDtype::U8,
            prompt_start: 5,
        };
        let value = serde_json::to_value(&re).unwrap();
        assert_eq!(value["num_layers"], 2);
        assert_eq!(value["top_k"], 3);
        assert_eq!(value["dtype"], "uint8");
        assert_eq!(value["prompt_start"], 5);
        let blob = base64::engine::general_purpose::STANDARD
            .decode(value["data"].as_str().unwrap())
            .unwrap();
        assert_eq!(&blob[..6], b"\x93NUMPY");
    }

    #[test]
    fn routed_experts_zero_rows_serializes_as_null() {
        let re = TurnRoutedExperts {
            data: Arc::new(Vec::new()),
            num_layers: 2,
            top_k: 3,
            dtype: TurnRoutedExpertsDtype::U8,
            prompt_start: 0,
        };
        assert!(serde_json::to_value(&re).unwrap().is_null());
    }

    #[test]
    fn turn_record_omits_routed_experts_when_absent() {
        let record = TurnRecord {
            prompt_token_count: 3,
            output_logprobs: None,
            finish_reason: "stop".to_string(),
            mismatch_report: Vec::new(),
            routed_experts: None,
            weight_version: None,
        };
        let value = serde_json::to_value(&record).unwrap();
        assert!(value.get("routed_experts").is_none());
    }
}
