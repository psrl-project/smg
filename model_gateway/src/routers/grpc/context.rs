//! Request context types for gRPC router pipeline
//!
//! This module provides the core context types that flow through the router pipeline,
//! eliminating deep parameter passing chains and providing a single source of truth
//! for request state.

use std::sync::Arc;

use axum::http::HeaderMap;
use llm_tokenizer::{stop::StopSequenceDecoder, traits::Tokenizer, TokenizerRegistry};
use openai_protocol::{
    chat::{ChatCompletionRequest, ChatCompletionResponse},
    classify::{ClassifyRequest, ClassifyResponse},
    completion::{CompletionRequest, CompletionResponse},
    embedding::{EmbeddingRequest, EmbeddingResponse},
    generate::{GenerateRequest, GenerateResponse},
    messages::{CreateMessageRequest, Message},
    responses::ResponsesRequest,
};
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use tool_parser::ParserFactory as ToolParserFactory;
use tracing::debug;

use super::{
    client::GrpcClient,
    epd_encode::EncodeDispatchPlan,
    multimodal::MultimodalComponents,
    proto_wrapper::{
        ProtoEmbedComplete, ProtoEmbedRequest, ProtoGenerateComplete, ProtoGenerateRequest,
        ProtoRequest, ProtoStream,
    },
    routing_loop::partial_rollout::PartialRolloutState,
};
use crate::{
    middleware::TenantRequestMeta,
    worker::{RuntimeType, Worker, WorkerLoadGuard},
};

/// Main request processing context
///
/// This is the single source of truth for all request state as it flows
/// through the pipeline stages. Uses Rust's type system to enforce proper
/// stage ordering at compile time.
pub(crate) struct RequestContext {
    pub input: RequestInput,
    pub components: Arc<SharedComponents>,
    pub state: ProcessingState,
}

/// Immutable request input
pub(crate) struct RequestInput {
    pub request_type: RequestType,
    pub headers: Option<HeaderMap>,
    pub model_id: String,
    pub tenant_request_meta: Option<TenantRequestMeta>,
}

/// Request type variants
/// Using Arc instead of Box to enable cheap cloning for background tasks
pub(crate) enum RequestType {
    Chat(Arc<ChatCompletionRequest>),
    Generate(Arc<GenerateRequest>),
    Completion(Arc<CompletionRequest>),
    Responses(Arc<ResponsesRequest>),
    Embedding(Arc<EmbeddingRequest>),
    Classify(Arc<ClassifyRequest>),
    Messages(Arc<CreateMessageRequest>),
}

impl std::fmt::Display for RequestType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chat(_) => write!(f, "Chat"),
            Self::Generate(_) => write!(f, "Generate"),
            Self::Completion(_) => write!(f, "Completion"),
            Self::Responses(_) => write!(f, "Responses"),
            Self::Embedding(_) => write!(f, "Embedding"),
            Self::Classify(_) => write!(f, "Classify"),
            Self::Messages(_) => write!(f, "Messages"),
        }
    }
}

impl RequestType {
    /// User-supplied `routed_experts_prompt_start`, read from the request's
    /// sampling parameters.  Defaults to 0 when the field is absent.
    ///
    /// Only the vLLM-native [`GenerateRequest`] exposes this knob via its nested
    /// [`SamplingParams`]; OpenAI-shaped chat/completion requests have no
    /// equivalent and always return 0 here, relying on TITO
    /// auto-management for non-zero values.
    pub fn routed_experts_prompt_start(&self) -> u32 {
        match self {
            Self::Generate(req) => req
                .sampling_params
                .as_ref()
                .and_then(|p| p.routed_experts_prompt_start)
                .unwrap_or(0),
            Self::Chat(_)
            | Self::Completion(_)
            | Self::Responses(_)
            | Self::Embedding(_)
            | Self::Classify(_)
            | Self::Messages(_) => 0,
        }
    }
}

impl std::fmt::Display for FinalResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chat(_) => write!(f, "Chat"),
            Self::Generate(_) => write!(f, "Generate"),
            Self::Completion(_) => write!(f, "Completion"),
            Self::Embedding(_) => write!(f, "Embedding"),
            Self::Classify(_) => write!(f, "Classify"),
            Self::Messages(_) => write!(f, "Messages"),
        }
    }
}

/// Shared components (injected once at creation)
pub(crate) struct SharedComponents {
    pub tokenizer_registry: Arc<TokenizerRegistry>,
    pub tool_parser_factory: ToolParserFactory,
    #[expect(dead_code)]
    pub reasoning_parser_factory: ReasoningParserFactory,
    /// Configured tool parser name (from CLI `--tool-call-parser`)
    pub configured_tool_parser: Option<String>,
    /// Multimodal processing components (initialized at router creation)
    pub multimodal: Option<Arc<MultimodalComponents>>,
}

/// Mutable processing state (evolves through pipeline stages)
#[derive(Default)]
pub(crate) struct ProcessingState {
    // Stage 1: Preparation outputs
    pub preparation: Option<PreparationOutput>,

    /// Snapshot of `preparation` saved before request_building consumes it.
    ///
    /// `request_building` takes ownership of `preparation` via `.take()`, so
    /// any partial-rollout loopback iteration would otherwise observe `None`
    /// when re-running `worker_selection` on the next pass
    pub preparation_snapshot: Option<PreparationOutput>,

    /// Resolved tokenizer (set once in preparation, reused in response processing)
    /// This avoids redundant registry lookups across pipeline stages.
    pub tokenizer: Option<Arc<dyn Tokenizer>>,

    // Stage 2: Worker selection outputs
    pub workers: Option<WorkerSelection>,

    // Stage 3: Client acquisition outputs
    pub clients: Option<ClientSelection>,

    // Stage 4: Request building outputs
    pub execution_plan: Option<ExecutionPlan>,

    // Stage 5: Dispatch metadata
    pub dispatch: Option<DispatchMetadata>,

    // Load guard for worker load tracking (created at execution stage)
    pub load_guards: Option<LoadGuards>,

    /// Admit-time token estimate (prompt + response-so-far) for the selected
    /// worker, captured in `WorkerSelectionStage` while `preparation` is still
    /// present. Consumed by the execution stage when minting `LoadGuards` so
    /// the worker's inflight-token counter receives the right estimate. `None`
    /// for request types where no token estimate is meaningful.
    pub admit_token_estimate: Option<usize>,

    // Stage 6: Response processing state
    pub response: ResponseState,

    /// Accumulated partial-rollout state across loopback iterations.
    pub partial_rollout_state: Option<PartialRolloutState>,

    /// Per-loopback overrides injected by `dispatch_entry_with_partial_rollout`
    /// into the next iteration's backend request build.  Empty on iter 1; the
    /// abort branch populates it before re-dispatch so vLLM only captures
    /// routed experts for *new* token positions.
    pub partial_rollout_overrides: PartialRolloutOverrides,

    /// TITO context (set once in preparation, reused in response processing)
    pub tito_context: Option<TitoRequestContext>,
}

/// Sampling-parameter overrides applied on the next loopback dispatch.
///
/// Kept as a small `Default`-able struct so adding future per-iteration
/// override knobs (e.g., for cooperative aborts on different signals) does
/// not churn `ProcessingState` shape.
///
/// All fields are `None` outside the PSRL loopback branch and on iter 1.
#[derive(Default, Debug, Clone)]
pub(crate) struct PartialRolloutOverrides {
    /// Set by `dispatch_entry_with_partial_rollout`'s abort branch
    /// using `first_iter_prompt_start + accumulator.num_tokens()`
    /// so vLLM does not re-capture RE for tokens already covered by prior
    /// iterations.
    pub routed_experts_prompt_start: Option<u32>,
}

/// Execution shape produced by request building and consumed by request execution.
pub(crate) enum ExecutionPlan {
    Single(ProtoRequest),
    PrefillDecode(ProtoGenerateRequest),
    EncodePrefillDecode {
        request: ProtoGenerateRequest,
        encode_dispatch: Option<EncodeDispatchPlan>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExecutionPlanKind {
    Single,
    PrefillDecode,
    EncodePrefillDecode,
}

impl ExecutionPlan {
    pub(crate) fn generate(
        kind: ExecutionPlanKind,
        request: ProtoGenerateRequest,
        encode_dispatch: Option<EncodeDispatchPlan>,
    ) -> Self {
        match kind {
            ExecutionPlanKind::Single => {
                debug_assert!(encode_dispatch.is_none());
                Self::Single(ProtoRequest::Generate(request))
            }
            ExecutionPlanKind::PrefillDecode => {
                debug_assert!(encode_dispatch.is_none());
                Self::PrefillDecode(request)
            }
            ExecutionPlanKind::EncodePrefillDecode => Self::EncodePrefillDecode {
                request,
                encode_dispatch,
            },
        }
    }

    pub(crate) fn embed(request: ProtoEmbedRequest) -> Self {
        Self::Single(ProtoRequest::Embed(request))
    }

    pub(crate) fn request_id(&self) -> &str {
        match self {
            Self::Single(request) => request.request_id(),
            Self::PrefillDecode(request) | Self::EncodePrefillDecode { request, .. } => {
                request.request_id()
            }
        }
    }

    pub(crate) fn request_type(&self) -> &'static str {
        match self {
            Self::Single(ProtoRequest::Generate(_))
            | Self::PrefillDecode(_)
            | Self::EncodePrefillDecode { .. } => "generate",
            Self::Single(ProtoRequest::Embed(_)) => "embed",
        }
    }

    pub(crate) fn mode_label(&self) -> &'static str {
        match self {
            Self::Single(_) => "single",
            Self::PrefillDecode(_) => "prefill_decode",
            Self::EncodePrefillDecode { .. } => "encode_prefill_decode",
        }
    }
}

/// Output from preparation stage (Step 1)
///
/// Each request type produces its own variant, eliminating optional fields
/// that are always None for certain pipelines.
#[derive(Clone)]
pub(crate) enum PreparationOutput {
    Chat {
        token_ids: Vec<u32>,
        processed_messages: super::ProcessedMessages,
        tool_constraints: Option<(String, String)>,
    },
    Messages {
        token_ids: Vec<u32>,
        processed_messages: super::ProcessedMessages,
        tool_constraints: Option<(String, String)>,
    },
    Completion {
        original_text: String,
        token_ids: Vec<u32>,
    },
    Generate {
        original_text: Option<String>,
        token_ids: Vec<u32>,
        multimodal_intermediate: Option<super::multimodal::MultimodalIntermediate>,
    },
    Embedding {
        original_text: String,
        token_ids: Vec<u32>,
    },
    Harmony {
        token_ids: Vec<u32>,
        selection_text: String,
        tool_constraints: Option<(String, String)>,
        /// Request with response_format cleared (when converted to structural tag)
        modified_request: Option<Box<ChatCompletionRequest>>,
        #[expect(dead_code, reason = "stored for future Harmony history tracking")]
        harmony_messages: Vec<super::harmony::HarmonyMessage>,
        harmony_stop_ids: Vec<u32>,
    },
}

impl PreparationOutput {
    /// Token IDs (common to all variants)
    pub fn token_ids(&self) -> &[u32] {
        match self {
            Self::Chat { token_ids, .. }
            | Self::Messages { token_ids, .. }
            | Self::Completion { token_ids, .. }
            | Self::Generate { token_ids, .. }
            | Self::Embedding { token_ids, .. }
            | Self::Harmony { token_ids, .. } => token_ids,
        }
    }

    /// Text for worker routing: original_text for regular pipelines, selection_text for Harmony.
    /// Chat/Messages borrow from processed_messages.text to avoid a redundant clone.
    pub fn routing_text(&self) -> Option<&str> {
        match self {
            Self::Chat {
                processed_messages, ..
            }
            | Self::Messages {
                processed_messages, ..
            } => Some(&processed_messages.text),
            Self::Completion { original_text, .. } | Self::Embedding { original_text, .. } => {
                Some(original_text)
            }
            Self::Generate { original_text, .. } => original_text.as_deref(),
            Self::Harmony { selection_text, .. } => Some(selection_text),
        }
    }
}

#[derive(Clone)]
pub(crate) struct EncodeWorkerAssignment {
    pub item_index: usize,
    pub worker: Arc<dyn Worker>,
}

/// Worker selection (Step 2)
pub(crate) enum WorkerSelection {
    Single {
        worker: Arc<dyn Worker>,
    },
    /// Disaggregated prefill/decode selection. EPD layers per-item encode
    /// assignments on top; plain PD leaves `encode_assignments` unset.
    Disaggregated {
        encode_assignments: Option<Vec<EncodeWorkerAssignment>>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        runtime_type: RuntimeType,
    },
}

/// Client selection (Step 3)
pub(crate) enum ClientSelection {
    Single {
        client: GrpcClient,
    },
    /// Disaggregated prefill/decode scheduler clients. EPD encode workers are
    /// contacted directly from `WorkerSelection::Disaggregated` assignments.
    Disaggregated {
        prefill: GrpcClient,
        decode: GrpcClient,
    },
}

/// Dispatch metadata (Step 5)
#[derive(Clone)]
pub(crate) struct DispatchMetadata {
    pub request_id: String,
    pub model: String,
    pub created: u64,
    pub weight_version: Option<String>,
}

/// Load guards for worker load tracking
/// Automatically decrements load when dropped
pub(crate) enum LoadGuards {
    Single {
        _guard: WorkerLoadGuard,
    },
    /// Disaggregated guards cover the prefill+decode pair. EPD encode workers are
    /// assigned per item; their fire-and-supervise RPCs do not hold load guards.
    Disaggregated {
        _prefill: WorkerLoadGuard,
        _decode: WorkerLoadGuard,
    },
}

impl LoadGuards {
    /// Construct load guards for the selected worker(s), optionally registering
    /// an admit-time token estimate in the selected worker(s)' inflight-token
    /// counter. The estimate flows to load-aware policies (e.g.
    /// throughput_optimal) as an optimistic delta on top of the engine token
    /// base until a fresh snapshot rebases it. Pass `None` to track request
    /// count only.
    ///
    /// For PD (Dual) selection the estimate is registered on the decode worker,
    /// which carries the generation token load; the prefill worker only tracks
    /// request count.
    pub fn with_token_estimate(
        selection: &WorkerSelection,
        headers: Option<&HeaderMap>,
        token_estimate: Option<usize>,
    ) -> Self {
        match selection {
            WorkerSelection::Single { worker } => LoadGuards::Single {
                _guard: match token_estimate {
                    Some(tokens) => {
                        WorkerLoadGuard::with_inflight_tokens(worker.clone(), headers, tokens)
                    }
                    None => WorkerLoadGuard::new(worker.clone(), headers),
                },
            },
            WorkerSelection::Disaggregated {
                prefill, decode, ..
            } => LoadGuards::Disaggregated {
                _prefill: WorkerLoadGuard::new(prefill.clone(), headers),
                _decode: match token_estimate {
                    Some(tokens) => {
                        WorkerLoadGuard::with_inflight_tokens(decode.clone(), headers, tokens)
                    }
                    None => WorkerLoadGuard::new(decode.clone(), headers),
                },
            },
        }
    }

    /// Like [`LoadGuards::with_token_estimate`] but uses
    /// [`WorkerLoadGuard::from_pre_incremented`] — the load counter was already
    /// incremented atomically inside the worker selector to close the TOCTOU
    /// window where all N concurrent dispatch tasks would otherwise read equal
    /// loads and deterministically pick the same worker.
    pub fn from_pre_incremented(
        selection: &WorkerSelection,
        headers: Option<&HeaderMap>,
        token_estimate: Option<usize>,
    ) -> Self {
        match selection {
            WorkerSelection::Single { worker } => LoadGuards::Single {
                _guard: WorkerLoadGuard::from_pre_incremented(
                    worker.clone(),
                    headers,
                    token_estimate,
                ),
            },
            WorkerSelection::Disaggregated {
                prefill, decode, ..
            } => LoadGuards::Disaggregated {
                _prefill: WorkerLoadGuard::from_pre_incremented(prefill.clone(), headers, None),
                _decode: WorkerLoadGuard::from_pre_incremented(
                    decode.clone(),
                    headers,
                    token_estimate,
                ),
            },
        }
    }
}

/// Context set by ChatPreparationStage when the X-SMG-Tito-Session-Id header is present.
/// Consumed by ChatResponseProcessingStage to store generation results.
pub(crate) struct TitoRequestContext {
    pub session_id: String,
    pub request: Arc<ChatCompletionRequest>,
    pub render_context: smg_tito::RenderContext,
    /// Model adapter selected once from server-loaded tokenizer metadata and
    /// reused by merge, validation, and capture.
    pub model_adapter: Arc<dyn smg_tito::model_adapter::ModelAdapter>,
    pub is_tito_hit: bool,
    /// Number of messages matched by TITO prefix (if is_tito_hit is true).
    /// Used for rollback detection: if new request matches fewer messages, we truncate turn_records.
    pub matched_message_num: usize,
    /// Resolved trajectory identifier. In manual mode this comes from the request
    /// header; in auto mode TITO derives it from the matched tree leaf.
    pub trajectory_id: u64,
    /// Prevents concurrent auto-mode branches from claiming the same trajectory.
    pub trajectory_id_reservation: Option<smg_tito::TrajectoryIdReservation>,
    /// Prompt token IDs computed during preparation (set in ChatPreparationStage, read in
    /// ChatResponseProcessingStage for TITO capture).
    /// Consumed by `ChatRequestBuildingStage::execute()` before response processing runs.
    pub prompt_token_ids: Vec<u32>,
    /// Prompt IDs before multimodal anchor expansion, retained for the next
    /// incremental TITO turn. Pure-text requests reuse `prompt_token_ids`.
    pub reusable_prompt_token_ids: Option<Vec<u32>>,
    /// Snapshot of the prefix hash.
    /// The response stage extends this in place with the newly-generated
    /// assistant message and finalizes it to derive the leaf hash,
    /// avoiding a second O(N) walk over the conversation.
    pub running_hasher: smg_tito::PrefixHasher,
    /// Hash at the last assistant boundary in the request messages (i.e. the
    /// parent hash of the node about to be stored).
    pub parent_hash: Option<smg_tito::PrefixHash>,
}

/// Response processing state (Step 6)
#[derive(Default)]
pub(crate) struct ResponseState {
    /// Stop sequence decoder
    pub stop_decoder: Option<StopSequenceDecoder>,

    /// Derived skip_special_tokens for streaming (set in preparation, read in response_processing).
    /// Stored here because PreparationOutput is consumed by request_building before
    /// response_processing runs.
    pub skip_special_tokens: Option<bool>,

    /// Exact prompt IDs dispatched to the backend for an opted-in generate
    /// request. Retained across partial-rollout loopback iterations.
    pub prompt_token_ids: Option<Vec<u32>>,

    /// Execution result (streams from workers)
    pub execution_result: Option<ExecutionResult>,

    /// Final processed response
    pub final_response: Option<FinalResponse>,

    /// Responses API iteration result (Harmony only, for tool loop orchestration)
    pub responses_iteration_result: Option<super::harmony::ResponsesIterationResult>,
}

impl RequestContext {
    /// Create context for chat completion request
    pub fn for_chat(
        request: Arc<ChatCompletionRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Chat(request),
                headers,
                model_id,
                tenant_request_meta: None,
            },
            components,
            state: ProcessingState::default(),
        }
    }

    /// Create context for generate request
    pub fn for_generate(
        request: Arc<GenerateRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Generate(request),
                headers,
                model_id,
                tenant_request_meta: None,
            },
            components,
            state: ProcessingState::default(),
        }
    }

    /// Create context for completion request
    pub fn for_completion(
        request: Arc<CompletionRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Completion(request),
                headers,
                model_id,
                tenant_request_meta: None,
            },
            components,
            state: ProcessingState::default(),
        }
    }

    /// Create context for Responses API request
    pub fn for_responses(
        request: Arc<ResponsesRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Responses(request),
                headers,
                model_id,
                tenant_request_meta: None,
            },
            components,
            state: ProcessingState::default(),
        }
    }

    /// Create context for embedding request
    pub fn for_embedding(
        request: Arc<EmbeddingRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Embedding(request),
                headers,
                model_id,
                tenant_request_meta: None,
            },
            components,
            state: ProcessingState::default(),
        }
    }

    /// Create context for classify request
    pub fn for_classify(
        request: Arc<ClassifyRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Classify(request),
                headers,
                model_id,
                tenant_request_meta: None,
            },
            components,
            state: ProcessingState::default(),
        }
    }

    /// Create context for messages request
    pub fn for_messages(
        request: Arc<CreateMessageRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Messages(request),
                headers,
                model_id,
                tenant_request_meta: None,
            },
            components,
            state: ProcessingState::default(),
        }
    }

    /// Get chat request (panics if not chat)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn chat_request(&self) -> &ChatCompletionRequest {
        match &self.input.request_type {
            RequestType::Chat(req) => req.as_ref(),
            _ => panic!("Expected chat request"),
        }
    }

    /// Get Arc clone of chat request (panics if not chat)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn chat_request_arc(&self) -> Arc<ChatCompletionRequest> {
        match &self.input.request_type {
            RequestType::Chat(req) => Arc::clone(req),
            _ => panic!("Expected chat request"),
        }
    }

    /// Get generate request (panics if not generate)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn generate_request(&self) -> &GenerateRequest {
        match &self.input.request_type {
            RequestType::Generate(req) => req.as_ref(),
            _ => panic!("Expected generate request"),
        }
    }

    /// Get Arc clone of generate request (panics if not generate)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn generate_request_arc(&self) -> Arc<GenerateRequest> {
        match &self.input.request_type {
            RequestType::Generate(req) => Arc::clone(req),
            _ => panic!("Expected generate request"),
        }
    }

    /// Get completion request (panics if not completion)
    #[expect(
        dead_code,
        reason = "ref accessor provided for API completeness alongside Arc accessor"
    )]
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn completion_request(&self) -> &CompletionRequest {
        match &self.input.request_type {
            RequestType::Completion(req) => req.as_ref(),
            _ => panic!("Expected completion request"),
        }
    }

    /// Get Arc clone of completion request (panics if not completion)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn completion_request_arc(&self) -> Arc<CompletionRequest> {
        match &self.input.request_type {
            RequestType::Completion(req) => Arc::clone(req),
            _ => panic!("Expected completion request"),
        }
    }

    /// Get Arc clone of responses request (panics if not responses)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn responses_request_arc(&self) -> Arc<ResponsesRequest> {
        match &self.input.request_type {
            RequestType::Responses(req) => Arc::clone(req),
            _ => panic!("Expected responses request"),
        }
    }

    /// Get messages request (panics if not messages)
    #[expect(
        dead_code,
        reason = "scaffolding for Messages API pipeline, wired in follow-up PR"
    )]
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn messages_request(&self) -> &CreateMessageRequest {
        match &self.input.request_type {
            RequestType::Messages(req) => req.as_ref(),
            _ => panic!("Expected messages request"),
        }
    }

    /// Get Arc clone of messages request (panics if not messages)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn messages_request_arc(&self) -> Arc<CreateMessageRequest> {
        match &self.input.request_type {
            RequestType::Messages(req) => Arc::clone(req),
            _ => panic!("Expected messages request"),
        }
    }

    /// Check if request is streaming
    pub fn is_streaming(&self) -> bool {
        match &self.input.request_type {
            RequestType::Chat(req) => req.stream,
            RequestType::Generate(req) => req.stream,
            RequestType::Completion(req) => req.stream,
            RequestType::Responses(req) => req.stream.unwrap_or(false),
            RequestType::Messages(req) => req.stream.unwrap_or(false),
            RequestType::Embedding(_) => false, // Embeddings are never streaming
            RequestType::Classify(_) => false,  // Classification is never streaming
        }
    }

    /// Get the cached tokenizer, cloning the Arc (cheap 8-byte clone)
    ///
    /// Returns None if tokenizer hasn't been resolved yet.
    /// The tokenizer is resolved once in the preparation stage and cached for reuse.
    pub fn tokenizer_arc(&self) -> Option<Arc<dyn Tokenizer>> {
        self.state.tokenizer.clone()
    }
}

/// Some methods are kept for API completeness even if currently unused.
#[expect(dead_code)]
impl WorkerSelection {
    pub fn is_disaggregated(&self) -> bool {
        matches!(self, Self::Disaggregated { .. })
    }

    pub fn single(&self) -> Option<&Arc<dyn Worker>> {
        match self {
            Self::Single { worker } => Some(worker),
            Self::Disaggregated { .. } => None,
        }
    }

    /// Record circuit breaker outcome for all workers based on HTTP status code.
    pub fn record_outcome(&self, status_code: u16) {
        match self {
            Self::Single { worker } => worker.record_outcome(status_code),
            Self::Disaggregated {
                prefill, decode, ..
            } => {
                // EPD encode dispatch is asynchronous and supervised by
                // RequestExecution; this records only the prefill/decode leg.
                prefill.record_outcome(status_code);
                decode.record_outcome(status_code);
            }
        }
    }

    /// Record circuit breaker outcomes for disaggregated dispatch (individual tracking)
    pub fn record_prefill_decode_outcomes(&self, prefill_status: u16, decode_status: u16) {
        if let Self::Disaggregated {
            prefill, decode, ..
        } = self
        {
            prefill.record_outcome(prefill_status);
            decode.record_outcome(decode_status);
        }
    }

    /// Record circuit breaker outcome for prefill worker only (sequential PD)
    pub fn record_outcome_prefill(&self, status_code: u16) {
        match self {
            Self::Disaggregated { prefill, .. } => {
                prefill.record_outcome(status_code);
            }
            Self::Single { .. } => {
                debug!("record_outcome_prefill called on Single worker selection, ignoring");
            }
        }
    }

    /// Record circuit breaker outcome for decode worker only (sequential PD)
    pub fn record_outcome_decode(&self, status_code: u16) {
        match self {
            Self::Disaggregated { decode, .. } => {
                decode.record_outcome(status_code);
            }
            Self::Single { .. } => {
                debug!("record_outcome_decode called on Single worker selection, ignoring");
            }
        }
    }

    #[expect(clippy::type_complexity)]
    pub fn disaggregated_pair(&self) -> Option<(&Arc<dyn Worker>, &Arc<dyn Worker>)> {
        match self {
            Self::Disaggregated {
                prefill, decode, ..
            } => Some((prefill, decode)),
            Self::Single { .. } => None,
        }
    }

    pub fn prefill_worker(&self) -> Option<&Arc<dyn Worker>> {
        match self {
            Self::Disaggregated { prefill, .. } => Some(prefill),
            Self::Single { .. } => None,
        }
    }

    pub fn decode_worker(&self) -> Option<&Arc<dyn Worker>> {
        match self {
            Self::Disaggregated { decode, .. } => Some(decode),
            Self::Single { .. } => None,
        }
    }

    /// Get the runtime type for disaggregated mode.
    pub fn disaggregated_runtime_type(&self) -> Option<&RuntimeType> {
        match self {
            Self::Disaggregated { runtime_type, .. } => Some(runtime_type),
            Self::Single { .. } => None,
        }
    }

    pub fn encode_assignments(&self) -> Option<&[EncodeWorkerAssignment]> {
        match self {
            Self::Disaggregated {
                encode_assignments, ..
            } => encode_assignments.as_deref(),
            Self::Single { .. } => None,
        }
    }
}

/// Some methods are kept for API completeness even if currently unused.
#[expect(dead_code)]
impl ClientSelection {
    pub fn single(&self) -> Option<&GrpcClient> {
        match self {
            Self::Single { client } => Some(client),
            Self::Disaggregated { .. } => None,
        }
    }

    pub fn single_mut(&mut self) -> Option<&mut GrpcClient> {
        match self {
            Self::Single { client } => Some(client),
            Self::Disaggregated { .. } => None,
        }
    }

    pub fn disaggregated_mut(&mut self) -> Option<(&mut GrpcClient, &mut GrpcClient)> {
        match self {
            Self::Disaggregated { prefill, decode } => Some((prefill, decode)),
            Self::Single { .. } => None,
        }
    }

    pub fn prefill_client(&self) -> Option<&GrpcClient> {
        match self {
            Self::Disaggregated { prefill, .. } => Some(prefill),
            Self::Single { .. } => None,
        }
    }

    pub fn prefill_client_mut(&mut self) -> Option<&mut GrpcClient> {
        match self {
            Self::Disaggregated { prefill, .. } => Some(prefill),
            Self::Single { .. } => None,
        }
    }

    pub fn decode_client(&self) -> Option<&GrpcClient> {
        match self {
            Self::Disaggregated { decode, .. } => Some(decode),
            Self::Single { .. } => None,
        }
    }

    pub fn decode_client_mut(&mut self) -> Option<&mut GrpcClient> {
        match self {
            Self::Disaggregated { decode, .. } => Some(decode),
            Self::Single { .. } => None,
        }
    }
}

/// Result of request execution (streams from workers)
/// Uses ProtoStream to automatically abort on cancellation
pub(crate) enum ExecutionResult {
    Single {
        stream: ProtoStream,
    },
    PrefillDecode {
        prefill: ProtoStream,
        decode: Box<ProtoStream>,
        /// PD timing context, for honest PD TTFT (prefill start to first decode token).
        pd_timing: PdTiming,
    },
    /// Embedding requests return a single response, not a stream
    Embedding {
        response: ProtoEmbedComplete,
    },
    /// Partial-rollout: accumulated complete frame after all loopback iterations finish.
    /// PostExecution response-processing stages handle this identically to a normal
    /// single-stream result — they just call `collect_responses` which returns the
    /// already-assembled `ProtoGenerateComplete` directly.
    Complete(ProtoGenerateComplete),
}

/// Timing context threaded from PD execution into the streaming layer so the
/// first decode token can be measured against prefill start.
#[derive(Clone)]
pub(crate) struct PdTiming {
    /// Monotonic instant the prefill RPC was dispatched.
    pub prefill_start: std::time::Instant,
    /// Backend runtime label (e.g. "sglang", "vllm") for the PD metric set.
    pub runtime: &'static str,
}

/// Final processed response
#[derive(Debug)]
pub(crate) enum FinalResponse {
    Chat(ChatCompletionResponse),
    /// Generate response is a Vec of GenerateResponse (n=1 returns single item, n>1 returns multiple)
    Generate(Vec<GenerateResponse>),
    /// Completion response (OpenAI /v1/completions format)
    Completion(CompletionResponse),
    /// Embedding response
    Embedding(EmbeddingResponse),
    /// Classification response
    Classify(ClassifyResponse),
    /// Messages API response
    Messages(Message),
}
