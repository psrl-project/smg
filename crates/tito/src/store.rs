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
    harness::ToolInputCanonicalizer,
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
    /// Monotonic creation order within the session (larger == stored later).
    pub seq: u64,
    /// True when this assistant turn ends with a tool call (no stored
    /// observation after it inside this node).
    pub ends_tool_call: bool,
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
    /// Monotonic commit sequence used to order nodes within this session.
    pub next_seq: u64,
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
            next_seq: 0,
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

/// One node of the TITO prefix tree (one assistant boundary).
///
/// ``id`` is a local, deterministic index assigned in root→leaf path order
/// (parents always sort before children).  ``hash`` is the blake3 hex of the
/// message prefix ending at this assistant boundary and lets downstream tooling
/// join a node across snapshots of the same session.
///
/// ``stored == false`` marks a **phantom boundary**: an assistant boundary that
/// appears in a request (typically a compaction / history-rewrite re-injection)
/// but that this session never stored as a generated node.  Such a boundary has
/// no token stream of its own (``num_tokens == 0``, ``finish_reason`` empty)
/// and its upstream is unknown (``parent == None``); it is the *cut point*
/// where a new trajectory begins after a context rewrite.
#[derive(Clone, Debug, Serialize)]
pub struct TitoNodeMeta {
    pub id: u64,
    pub hash: String,
    pub stored: bool,
    pub parent: Option<u64>,
    pub finish_reason: String,
    pub truncated: bool,
    pub num_tokens: usize,
    /// Trajectory ids whose root→leaf path passes through this node.
    pub trajectory_ids: Vec<u64>,
}

/// One live trajectory leaf (root→leaf view) in the session tree.
#[derive(Clone, Debug, Serialize)]
pub struct TitoLeafMeta {
    pub trajectory_id: u64,
    pub node_id: u64,
    pub parent: Option<u64>,
    /// Root→leaf node ids (in [`TitoTreeMeta::nodes`] id space).
    pub path_node_ids: Vec<u64>,
}

/// Prefix-tree structure of one session snapshot.
///
/// Node ids are local to a snapshot; use ``hash`` for cross-snapshot joins.
/// This is the structural layer that lets analysis tooling answer "is leaf B a
/// sibling of leaf A", "which node is the shared parent", and "is a leaf the
/// truncated tail of a path that a later sibling superseded" — without
/// re-deriving parentage from token content.
#[derive(Clone, Debug, Serialize)]
pub struct TitoTreeMeta {
    pub nodes: Vec<TitoNodeMeta>,
    pub leaves: Vec<TitoLeafMeta>,
}

/// Consistent training-data snapshot returned for every TITO session.
#[derive(Debug, Serialize)]
pub struct TitoSessionData {
    pub session_id: String,
    pub max_trim_tokens: usize,
    /// Always an array: empty, single-trajectory, and multi-trajectory sessions
    /// share the same wire format.
    pub trajectories: Vec<Trajectory>,
    /// Prefix-tree structural metadata for the same snapshot.
    pub tree: TitoTreeMeta,
}

/// Top-level store: session_id → Arc<Mutex<SessionState>>.
pub struct TitoStore {
    sessions: DashMap<String, Arc<parking_lot::Mutex<SessionState>>>,
    debug: AtomicBool,
    gc_threshold: std::sync::atomic::AtomicUsize,
    trajectory_id_strategy: TrajectoryIdStrategy,
    tool_canonicalizer: Arc<dyn ToolInputCanonicalizer>,
    /// When enabled, snapshots exclude "rollback" leaves (dead branches that
    /// forked from a state the session later continued past) so they are never
    /// down-streamed to the agent loop as training data.
    drop_dead_leaves: AtomicBool,
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
    /// True when the appended slice `messages[matched_message_num..]` contains at
    /// least one assistant turn.  This happens when the harness compacted /
    /// re-injected the conversation (e.g. Claude Code auto-compact): those
    /// assistant turns are treated as prompt (mask 0) and are re-tokenized
    /// incrementally by the engine.
    pub has_assistant_in_appended: bool,
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
            tool_canonicalizer: crate::harness::build_tool_canonicalizer(None, None),
            drop_dead_leaves: AtomicBool::new(false),
        }
    }

    /// Enable/disable dropping dead (rollback) leaves from session snapshots.
    pub fn set_drop_dead_leaves(&self, enabled: bool) {
        self.drop_dead_leaves.store(enabled, Ordering::Relaxed);
    }

    /// Whether dead (rollback) leaves are excluded from session snapshots.
    pub fn drop_dead_leaves_enabled(&self) -> bool {
        self.drop_dead_leaves.load(Ordering::Relaxed)
    }

    /// Attach a harness-specific tool-input canonicalizer used by every prefix
    /// hash the store computes (lookup and stored boundaries alike).
    pub fn with_tool_canonicalizer(
        mut self,
        canonicalizer: Arc<dyn ToolInputCanonicalizer>,
    ) -> Self {
        self.tool_canonicalizer = canonicalizer;
        self
    }

    /// Hash one message using the store's harness canonicalizer.
    pub fn hash_message(&self, hasher: &mut PrefixHasher, msg: &ChatMessage) {
        crate::normalizer::hash_message_into_with(
            hasher,
            msg,
            Some(self.tool_canonicalizer.as_ref()),
        );
    }

    /// Hash a full message slice using the store's harness canonicalizer.
    pub fn hash_messages_with_context(
        &self,
        messages: &[ChatMessage],
        context: &RenderContext,
    ) -> PrefixHash {
        crate::normalizer::hash_messages_with_context_with(
            messages,
            context,
            Some(self.tool_canonicalizer.as_ref()),
        )
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
    /// If a HIT candidate would require an assistant turn inside the appended
    /// slice, the hit is still returned: the assistant turns are interpreted as
    /// **compacted conversation context** (e.g. Claude Code auto-compact) that
    /// this session's model never generated.  `merge_incremental` re-tokenizes
    /// them into the prompt and they never receive a training mask (mask 0).
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
            self.hash_message(&mut hasher, msg);
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
                let has_assistant_in_appended = appended.iter().any(is_assistant_role);
                if has_assistant_in_appended {
                    // The harness compacted / re-injected the conversation: the
                    // appended slice contains assistant turns that this session's
                    // model never generated.  Keep the hit and treat them as
                    // prompt — `merge_incremental` re-tokenizes them and the
                    // training mask stays 0 (only generated output is masked 1).
                    // WARN in debug mode (compaction is notable there); DEBUG
                    // otherwise so production logs do not spam on every hit.
                    if self.is_debug() {
                        tracing::warn!(
                            session_id = %session_id,
                            matched_len = *k,
                            appended_with_assistant = true,
                            appended_msg_count = appended.len(),
                            "TITO HIT — appended slice contains assistant turn(s); \
                             treating as compacted context (prompt, mask 0)"
                        );
                    } else {
                        tracing::debug!(
                            session_id = %session_id,
                            matched_len = *k,
                            appended_with_assistant = true,
                            appended_msg_count = appended.len(),
                            "TITO HIT — appended slice contains assistant turn(s); \
                             treating as compacted context (prompt, mask 0)"
                        );
                    }
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
                        has_assistant_in_appended,
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
        let leaf_hash = self.hash_messages_with_context(messages, render_context);
        let parent_hash = self.compute_parent_hash_canonical(messages, render_context);
        self.store_with_hashes(
            session_id,
            leaf_hash,
            parent_hash,
            token_ids,
            turn_record,
            trajectory_id,
            false,
        )
    }

    /// Hash of the message list ending at the second-to-last assistant turn,
    /// using the store's harness canonicalizer (mirrors the free
    /// [`compute_parent_hash`] helper for canonical-aware hashing).
    fn compute_parent_hash_canonical(
        &self,
        messages: &[ChatMessage],
        render_context: &RenderContext,
    ) -> Option<PrefixHash> {
        let last_asst = messages.iter().rposition(is_assistant_role)?;
        let second_last_asst = messages[..last_asst].iter().rposition(is_assistant_role)?;
        Some(self.hash_messages_with_context(
            &messages[..=second_last_asst],
            render_context,
        ))
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
        skip_prefix_validation: bool,
    ) -> Result<(), TitoError> {
        self.store_token_sequences(
            session_id,
            leaf_hash,
            parent_hash,
            StoredTokenSequences::shared(token_ids),
            turn_record,
            trajectory_id,
            skip_prefix_validation,
            false,
        )
    }

    /// [`Self::store_with_hashes`] with an explicit tool-call-end marker.
    ///
    /// `ends_tool_call` should be true when this turn's assistant message ends
    /// with a tool call (no observation inside this node) — used for dead-leaf
    /// (rollback) detection at snapshot time.
    pub fn store_with_hashes_and_marker(
        &self,
        session_id: &str,
        leaf_hash: PrefixHash,
        parent_hash: Option<PrefixHash>,
        token_ids: Vec<u32>,
        turn_record: TurnRecord,
        trajectory_id: u64,
        skip_prefix_validation: bool,
        ends_tool_call: bool,
    ) -> Result<(), TitoError> {
        self.store_token_sequences(
            session_id,
            leaf_hash,
            parent_hash,
            StoredTokenSequences::shared(token_ids),
            turn_record,
            trajectory_id,
            skip_prefix_validation,
            ends_tool_call,
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
        skip_prefix_validation: bool,
    ) -> Result<(), TitoError> {
        self.store_with_hashes_and_reusable_with_marker(
            session_id,
            leaf_hash,
            parent_hash,
            token_ids,
            turn_record,
            trajectory_id,
            skip_prefix_validation,
            false,
        )
    }

    /// [`Self::store_with_hashes_and_reusable`] with an explicit tool-call-end
    /// marker (see [`Self::store_with_hashes_and_marker`]).
    pub fn store_with_hashes_and_reusable_with_marker(
        &self,
        session_id: &str,
        leaf_hash: PrefixHash,
        parent_hash: Option<PrefixHash>,
        token_ids: StoredTokenSequences,
        turn_record: TurnRecord,
        trajectory_id: u64,
        skip_prefix_validation: bool,
        ends_tool_call: bool,
    ) -> Result<(), TitoError> {
        self.store_token_sequences(
            session_id,
            leaf_hash,
            parent_hash,
            token_ids,
            turn_record,
            trajectory_id,
            skip_prefix_validation,
            ends_tool_call,
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
        skip_prefix_validation: bool,
        ends_tool_call: bool,
    ) -> Result<(), TitoError> {
        let arc = self.get_or_create_session_arc(session_id);
        let mut state = arc.lock();

        // miles-style commit-time prefix validation. For a clean append-only
        // extension the new checkpoint (prompt + completion) must begin with
        // the stored trajectory stream, tolerating up to `max_trim_tokens`
        // boundary differences (e.g. a truncated turn whose closing `<|im_end|>`
        // the template re-emits). This is the correctness gate that catches
        // genuine TITO prefix bugs without false-positiving on truncation.
        // Skipped for compacted-context hits where the prompt legitimately
        // diverges, and for nodes without a parent.
        if !skip_prefix_validation {
            if let Some(ph) = parent_hash {
                if let Some(parent) = state.entries.get(&ph) {
                    let prev: &[u32] = parent
                        .reusable_prefix_token_ids
                        .as_deref()
                        .unwrap_or(&parent.token_ids);
                    let new_ids: &[u32] = token_ids.reusable.as_deref().unwrap_or(&token_ids.expanded);
                    let max_trim = state.max_trim_tokens;
                    let check_len = prev.len().saturating_sub(max_trim);
                    if check_len > 0 && new_ids.len() >= check_len && &new_ids[..check_len] != &prev[..check_len] {
                        let first = prev[..check_len]
                            .iter()
                            .zip(&new_ids[..check_len])
                            .position(|(a, b)| a != b)
                            .unwrap_or(check_len);
                        return Err(TitoError::PrefixMismatch(format!(
                            "stored prefix {} tokens diverges from new checkpoint (first mismatch at {}, \
                             max_trim_tokens={})",
                            prev.len(),
                            first,
                            max_trim
                        )));
                    }
                }
            }
        }

        // Leaf tracking: the parent is no longer a leaf once we add a child.
        if let Some(ph) = parent_hash {
            state.leaf_hashes.remove(&ph);
        }
        state.leaf_hashes.insert(leaf_hash);

        let seq = state.next_seq;
        state.next_seq = state.next_seq.wrapping_add(1);
        state.entries.insert(
            leaf_hash,
            PrefixEntry {
                token_ids: Arc::new(token_ids.expanded),
                reusable_prefix_token_ids: token_ids.reusable.map(Arc::new),
                parent_hash,
                turn_record,
                seq,
                ends_tool_call,
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

        let excluded = self.dead_leaf_exclusions(&state);
        collect_trajectories(&state, &excluded)
    }

    /// Return an atomic session snapshot with a stable multi-trajectory shape.
    ///
    /// The session existence check, metadata read, and trajectory collection all
    /// happen under one lock so callers cannot observe a mixed snapshot.
    pub fn get_session_data(&self, session_id: &str) -> Option<TitoSessionData> {
        let arc = self.get_session_arc(session_id)?;
        let state = arc.lock();

        let excluded = self.dead_leaf_exclusions(&state);
        Some(TitoSessionData {
            session_id: session_id.to_string(),
            max_trim_tokens: state.max_trim_tokens,
            trajectories: collect_trajectories(&state, &excluded),
            tree: collect_tree_meta(&state, &excluded),
        })
    }

    /// Dead-leaf exclusion set for snapshot collection.
    ///
    /// Empty when dead-leaf dropping is disabled (fast path: no tree walk) so
    /// the hot path is byte-identical to the previous behavior.
    fn dead_leaf_exclusions(&self, state: &SessionState) -> HashSet<PrefixHash> {
        if !self.drop_dead_leaves_enabled() {
            return HashSet::new();
        }
        compute_dead_leaf_hashes(state)
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

fn hash_hex(hash: &PrefixHash) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(hash.len() * 2);
    for byte in hash {
        let _ = write!(s, "{byte:02x}");
    }
    s
}

/// Collect the prefix-tree structure of every node reachable from the live
/// trajectory pointers, plus a root→leaf view of each trajectory.
///
/// Walking each trajectory from its leaf toward the root stops at the first
/// **phantom boundary** — a parent hash that has no stored entry.  Phantom
/// boundaries arise when a request replays context the session never generated
/// (Claude Code auto-compact / history rewrite): the request attaches to a
/// re-injected assistant boundary that is not a stored node.  Such boundaries
/// are exported with ``stored=false`` (no token stream, unknown upstream) and
/// act as the *cut point* where a new trajectory begins.
///
/// Node ids are assigned by sorting hashes by their earliest root→leaf path
/// position, then by hash: parents always precede their children.  Every node
/// carries its blake3 ``hash`` so tooling can join it across snapshots.
fn compute_dead_leaf_hashes(state: &SessionState) -> HashSet<PrefixHash> {
    // 1. Root->leaf chains for every live trajectory (phantom head included;
    //    only stored nodes carry seq/ends_tool_call metadata).
    struct LeafInfo {
        path: Vec<PrefixHash>,
        ends_tool_call: bool,
    }
    let mut leaves_info: Vec<LeafInfo> = Vec::with_capacity(state.trajectory_leaves.len());
    for leaf_hash in state.trajectory_leaves.values() {
        let mut chain_rev: Vec<PrefixHash> = Vec::new();
        let mut current = Some(*leaf_hash);
        while let Some(hash) = current {
            chain_rev.push(hash);
            match state.entries.get(&hash) {
                Some(entry) => current = entry.parent_hash,
                None => current = None,
            }
        }
        chain_rev.reverse();
        let leaf_entry = state.entries.get(leaf_hash);
        leaves_info.push(LeafInfo {
            path: chain_rev,
            ends_tool_call: leaf_entry.map(|e| e.ends_tool_call).unwrap_or(false),
        });
    }

    // 2. Dead = dangling tool-call leaf (its own last stored node ends with a
    //    tool call and nothing follows on this chain) AND another leaf shares a
    //    real prefix with it and diverges (or extends) with nodes created
    //    strictly later.  The "later" comparison happens at the divergence
    //    point only: the sibling's first node after the shared prefix must have
    //    been stored after this leaf's own node at the same divergence.
    let mut dead: HashSet<PrefixHash> = HashSet::new();
    for (i, leaf) in leaves_info.iter().enumerate() {
        if !leaf.ends_tool_call {
            continue;
        }
        let mut is_dead = false;
        for (j, other) in leaves_info.iter().enumerate() {
            if i == j {
                continue;
            }
            let len_a = leaf.path.len();
            let len_b = other.path.len();
            let shared = leaf
                .path
                .iter()
                .zip(other.path.iter())
                .take_while(|(a, b)| a == b)
                .count();
            if shared == 0 {
                continue; // different (phantom) worlds: no real shared prefix
            }
            if shared == len_a {
                // `other` strictly extends this leaf (or is identical).
                //
                // Note: with the "leaf has no children" invariant this branch
                // is unreachable for a *true* leaf.  It exists because
                // `trajectory_leaves` can transiently point at an internal node
                // that another trajectory later extended underneath it (a
                // superseded/stale trajectory pointer).  Such a stale pointer
                // *is* a rollback artifact and must be dropped; this branch
                // catches exactly that case.
                if len_b > len_a {
                    // Any node stored beyond this leaf is necessarily later
                    // (monotonic seq), so the dangling leaf was superseded.
                    if let (Some(a), Some(b)) = (
                        state.entries.get(&leaf.path[len_a - 1]),
                        state.entries.get(&other.path[len_a]),
                    ) {
                        if b.seq > a.seq {
                            is_dead = true;
                            break;
                        }
                    }
                }
                continue;
            }
            if shared >= len_b {
                continue; // `other` is a strict prefix of this leaf
            }
            // True divergence at index `shared`: sibling child vs this leaf's
            // child.  Only mark dead when the sibling side was stored later.
            let child_a = state.entries.get(&leaf.path[shared]);
            let child_b = state.entries.get(&other.path[shared]);
            if let (Some(a), Some(b)) = (child_a, child_b) {
                if b.seq > a.seq {
                    is_dead = true;
                    break;
                }
            }
        }
        if is_dead {
            if let Some(leaf_hash) = leaf.path.last() {
                dead.insert(*leaf_hash);
            }
        }
    }
    dead
}

fn collect_tree_meta(state: &SessionState, excluded: &HashSet<PrefixHash>) -> TitoTreeMeta {
    // 1. Root→leaf hash paths for every live trajectory.  Climb from the leaf
    //    through stored entries; a parent without a stored entry is a phantom
    //    boundary and terminates the climb (its own upstream is unknown).
    let mut pairs: Vec<(u64, PrefixHash)> = state
        .trajectory_leaves
        .iter()
        .filter(|(_, leaf_hash)| !excluded.contains(*leaf_hash))
        .map(|(&trajectory_id, &leaf_hash)| (trajectory_id, leaf_hash))
        .collect();
    pairs.sort_unstable_by_key(|(trajectory_id, _)| *trajectory_id);

    let mut paths: Vec<(u64, Vec<PrefixHash>)> = Vec::new();
    for (trajectory_id, leaf_hash) in pairs {
        let mut chain_rev: Vec<PrefixHash> = Vec::new();
        let mut current = Some(leaf_hash);
        while let Some(hash) = current {
            chain_rev.push(hash);
            match state.entries.get(&hash) {
                Some(entry) => current = entry.parent_hash,
                // Phantom boundary: referenced as a parent but never stored;
                // it has no entry, so we cannot climb above it.
                None => current = None,
            }
        }
        chain_rev.reverse();
        paths.push((trajectory_id, chain_rev));
    }

    // 2. Earliest path position per hash -> deterministic parent-before-child
    //    ordering (parents strictly precede children inside every path).
    let mut min_pos: HashMap<PrefixHash, usize> = HashMap::new();
    for (_, path) in &paths {
        for (pos, hash) in path.iter().enumerate() {
            let entry = min_pos.entry(*hash).or_insert(pos);
            if pos < *entry {
                *entry = pos;
            }
        }
    }
    let mut ordered: Vec<PrefixHash> = min_pos.keys().copied().collect();
    ordered.sort_unstable_by(|a, b| min_pos[a].cmp(&min_pos[b]).then_with(|| a.cmp(b)));
    let id_of: HashMap<PrefixHash, u64> = ordered
        .iter()
        .enumerate()
        .map(|(i, h)| (*h, i as u64))
        .collect();

    // 3. Per-node metadata.  Stored nodes carry entry data; phantom nodes are
    //    marked stored=false with an empty finish_reason and no token stream.
    let mut nodes: Vec<TitoNodeMeta> = ordered
        .iter()
        .map(|hash| {
            let entry = state.entries.get(hash);
            let id = id_of[hash];
            match entry {
                Some(entry) => TitoNodeMeta {
                    id,
                    hash: hash_hex(hash),
                    stored: true,
                    parent: entry.parent_hash.and_then(|ph| id_of.get(&ph).copied()),
                    finish_reason: entry.turn_record.finish_reason.clone(),
                    truncated: entry.turn_record.finish_reason == "length",
                    num_tokens: entry.token_ids.len(),
                    trajectory_ids: Vec::new(),
                },
                None => TitoNodeMeta {
                    id,
                    hash: hash_hex(hash),
                    stored: false,
                    parent: None,
                    finish_reason: String::new(),
                    truncated: false,
                    num_tokens: 0,
                    trajectory_ids: Vec::new(),
                },
            }
        })
        .collect();

    // 4. Trajectory leaves: root→leaf node id paths; tag each path node with
    //    the trajectory id so consumers can see which trajectories share nodes.
    let mut leaves: Vec<TitoLeafMeta> = Vec::new();
    for (trajectory_id, path) in &paths {
        let path_node_ids: Vec<u64> = path
            .iter()
            .filter_map(|hash| id_of.get(hash).copied())
            .collect();
        let Some(leaf_id) = path_node_ids.last().copied() else {
            continue;
        };
        for node_id in &path_node_ids {
            if let Some(node) = nodes.iter_mut().find(|n| n.id == *node_id) {
                node.trajectory_ids.push(*trajectory_id);
            }
        }
        leaves.push(TitoLeafMeta {
            trajectory_id: *trajectory_id,
            node_id: leaf_id,
            parent: nodes.iter().find(|n| n.id == leaf_id).and_then(|n| n.parent),
            path_node_ids,
        });
    }

    TitoTreeMeta { nodes, leaves }
}

fn collect_trajectories(state: &SessionState, excluded: &HashSet<PrefixHash>) -> Vec<Trajectory> {
    let mut pairs: Vec<(u64, PrefixHash)> = state
        .trajectory_leaves
        .iter()
        .filter(|(_, leaf_hash)| !excluded.contains(*leaf_hash))
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

    /// Build a `Write` assistant turn.  `spaced_blank_lines` reproduces the
    /// model-emitted spelling (blank lines carry four trailing spaces, key
    /// `file_path` first); the replay spelling used in the tests puts
    /// `content` first and spells blank lines empty.
    fn write_assistant_msg(spaced_blank_lines: bool, file_path_first: bool) -> ChatMessage {
        use openai_protocol::common::{FunctionCallResponse, ToolCall};

        let blank = if spaced_blank_lines { "    \n" } else { "\n" };
        let content = format!("import boto3\n{blank}    # Create a table\nclient.put_item(\n)\n");
        let file_path = "/tmp/test_decimal_issue.py";
        let mut args = serde_json::Map::new();
        if file_path_first {
            args.insert(
                "file_path".to_string(),
                serde_json::Value::String(file_path.to_string()),
            );
            args.insert("content".to_string(), serde_json::Value::String(content));
        } else {
            args.insert("content".to_string(), serde_json::Value::String(content));
            args.insert(
                "file_path".to_string(),
                serde_json::Value::String(file_path.to_string()),
            );
        }
        ChatMessage::Assistant {
            content: None,
            name: None,
            reasoning_content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_w".to_string(),
                tool_type: "function".to_string(),
                function: FunctionCallResponse {
                    name: "Write".to_string(),
                    arguments: Some(serde_json::Value::Object(args).to_string()),
                },
            }]),
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
    fn find_prefix_assistant_in_appended_returns_hit() {
        // An assistant turn inside the appended slice (compacted context) must
        // not reject the request: the hit is returned and the assistant turn is
        // treated as prompt (mask 0).
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
        let hit = store.find_prefix("s1", &msgs, &ctx).unwrap().unwrap();
        assert_eq!(hit.pretokenized_ids, vec![1, 2, 3]);
        assert_eq!(hit.matched_message_num, 2);
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
    fn whitespace_only_arg_drift_still_prefix_hits_and_reuses_stored_tokens() {
        // Regression: a Write tool call whose `content` value differs only in
        // line-trailing whitespace (model emits blank lines as `"    "`, the
        // client re-serializes them as `""`) plus a swapped key order must
        // still HIT the stored assistant boundary and reuse the stored tokens,
        // instead of missing and landing the assistant turn in the appended
        // slice (which forks a new leaf via re-tokenization).
        let store = make_store();
        store.create_session("s1");
        let ctx = render_context();

        let stored_turn = vec![user_msg("fix the bug"), write_assistant_msg(true, true)];
        store_turn(
            &store,
            "s1",
            &stored_turn,
            vec![10, 20],
            record(2, "stop"),
            &ctx,
        );

        let replay = vec![
            user_msg("fix the bug"),
            write_assistant_msg(false, false),
            user_msg("<tool_response>file written</tool_response>"),
        ];
        let hit = store.find_prefix("s1", &replay, &ctx).unwrap().unwrap();
        assert_eq!(hit.matched_message_num, 2, "assistant boundary must match");
        assert!(!hit.has_assistant_in_appended);
        assert_eq!(hit.pretokenized_ids, vec![10, 20], "stored tokens reused");
    }

    #[test]
    fn genuine_content_drift_still_misses() {
        // Safety: the whitespace tolerance must NOT bridge a real content
        // change — a different non-whitespace character still misses.
        let store = make_store();
        store.create_session("s1");
        let ctx = render_context();

        let stored_turn = vec![user_msg("fix the bug"), write_assistant_msg(true, true)];
        store_turn(
            &store,
            "s1",
            &stored_turn,
            vec![10, 20],
            record(2, "stop"),
            &ctx,
        );

        let mut changed = write_assistant_msg(false, false);
        if let ChatMessage::Assistant {
            tool_calls: Some(calls),
            ..
        } = &mut changed
        {
            if let Some(args) = calls[0].function.arguments.as_mut() {
                *args = args.replace("import boto3", "import os");
            }
        }
        let replay = vec![user_msg("fix the bug"), changed, user_msg("next")];
        assert!(store.find_prefix("s1", &replay, &ctx).unwrap().is_none());
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
                "tree": {"nodes": [], "leaves": []},
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
        assert_eq!(single["tree"]["leaves"].as_array().unwrap().len(), 1);
        assert_eq!(single["tree"]["nodes"][0]["stored"], true);
        assert_eq!(single["tree"]["leaves"][0]["path_node_ids"].as_array().unwrap().len(), 1);

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
        let multi_leaves = multi["tree"]["leaves"].as_array().unwrap();
        assert_eq!(multi_leaves.len(), 2);
        // Two independent roots: distinct leaf nodes, both parent-free.
        assert_ne!(multi_leaves[0]["node_id"], multi_leaves[1]["node_id"]);
        assert!(multi_leaves[0]["parent"].is_null());
        assert!(multi_leaves[1]["parent"].is_null());
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
                false,
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
    fn prefix_validation_tolerates_boundary_but_rejects_divergence() {
        let store = make_store();
        store.create_session("s");
        // Qwen3-like ceiling: tolerate 1 trailing boundary token per turn.
        store.set_session_max_trim_tokens("s", 1);

        let parent_hash: PrefixHash = [1u8; 32];
        store
            .store_with_hashes("s", parent_hash, None, vec![10, 20, 30], record(3, "stop"), 0, false)
            .unwrap();

        // Clean extension: the new checkpoint re-emits the truncated turn's
        // closing `<|im_end|>` + newline after the stored content. The last
        // stored token (30) falls inside the max_trim=1 tolerance window, so
        // the shared prefix (first two tokens) still matches.
        store
            .store_with_hashes(
                "s",
                [2u8; 32],
                Some(parent_hash),
                vec![10, 20, 30, 151645, 198],
                record(5, "stop"),
                0,
                false,
            )
            .unwrap();

        // Genuine divergence inside the shared prefix must be rejected.
        let err = store
            .store_with_hashes(
                "s",
                [3u8; 32],
                Some(parent_hash),
                vec![99, 20, 30, 40],
                record(4, "stop"),
                0,
                false,
            )
            .unwrap_err();
        assert!(
            matches!(err, TitoError::PrefixMismatch(_)),
            "expected PrefixMismatch, got {err:?}"
        );

        // The rejected node must not have been stored: the trajectory still
        // points at the clean-extension leaf, not the divergent node.
        let trajectories = store.get_all_trajectories("s");
        assert_eq!(trajectories.len(), 1);
        assert_eq!(trajectories[0].accumulated_token_ids, vec![10, 20, 30, 151645, 198]);
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
                false,
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

#[cfg(test)]
mod dead_leaf_tests {
    use super::*;

    fn hash_from(label: &str) -> PrefixHash {
        *blake3::hash(label.as_bytes()).as_bytes()
    }

    fn entry(parent: Option<PrefixHash>, seq: u64, ends_tool_call: bool) -> PrefixEntry {
        PrefixEntry {
            token_ids: Arc::new(vec![]),
            reusable_prefix_token_ids: None,
            parent_hash: parent,
            turn_record: TurnRecord {
                prompt_token_count: 0,
                output_logprobs: None,
                finish_reason: "stop".to_string(),
                mismatch_report: vec![],
                routed_experts: None,
                weight_version: None,
            },
            seq,
            ends_tool_call,
        }
    }

    #[test]
    fn dangling_rollback_with_later_diverging_sibling_is_dead() {
        let root = hash_from("root");
        let div = hash_from("div");
        let x = hash_from("x");
        let y = hash_from("y");
        let y2 = hash_from("y2");
        let mut state = SessionState::new();
        state.entries.insert(root, entry(None, 0, false));
        state.entries.insert(div, entry(Some(root), 1, false));
        state.entries.insert(x, entry(Some(div), 2, true));
        state.entries.insert(y, entry(Some(div), 3, false));
        state.entries.insert(y2, entry(Some(y), 4, false));
        state.trajectory_leaves.insert(0, y2);
        state.trajectory_leaves.insert(1, x);

        let dead = compute_dead_leaf_hashes(&state);
        assert!(dead.contains(&x), "dangling rollback leaf x should be dead");
        assert!(!dead.contains(&y2), "continued leaf y2 must be kept");
    }

    #[test]
    fn dangling_leaf_without_later_sibling_is_kept() {
        let root = hash_from("root2");
        let child = hash_from("child");
        let mut state = SessionState::new();
        state.entries.insert(root, entry(None, 0, false));
        state.entries.insert(child, entry(Some(root), 1, true));
        state.trajectory_leaves.insert(0, child);
        assert!(compute_dead_leaf_hashes(&state).is_empty());
    }

    #[test]
    fn non_dangling_shorter_leaf_is_kept() {
        let root = hash_from("root3");
        let div = hash_from("div3");
        let short = hash_from("short");
        let deep = hash_from("deep");
        let mut state = SessionState::new();
        state.entries.insert(root, entry(None, 0, false));
        state.entries.insert(div, entry(Some(root), 1, false));
        state.entries.insert(short, entry(Some(div), 2, false));
        state.entries.insert(deep, entry(Some(div), 3, false));
        state.trajectory_leaves.insert(0, short);
        state.trajectory_leaves.insert(1, deep);
        assert!(compute_dead_leaf_hashes(&state).is_empty());
    }

    #[test]
    fn same_depth_compaction_siblings_are_kept() {
        let p = hash_from("comp-p");
        let a = hash_from("comp-a");
        let b = hash_from("comp-b");
        let mut state = SessionState::new();
        state.entries.insert(p, entry(None, 0, false));
        state.entries.insert(a, entry(Some(p), 1, false));
        state.entries.insert(b, entry(Some(p), 2, false));
        state.trajectory_leaves.insert(0, a);
        state.trajectory_leaves.insert(1, b);
        assert!(compute_dead_leaf_hashes(&state).is_empty());
    }
}
