//! Chat response processing stage: Handles both streaming and non-streaming responses
//!
//! - For streaming: Spawns background task and returns SSE response (early exit)
//! - For non-streaming: Collects all responses and builds final ChatCompletionResponse

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use openai_protocol::chat::{ChatCompletionMessage, ChatMessage, MessageContent};
use smg_tito::{RenderContext, TitoStore};
use tracing::{debug, error, warn};

use crate::{
    routers::{
        error,
        grpc::{
            common::{
                response_collection,
                stages::{PipelineStage, StagePhase},
            },
            context::{FinalResponse, RequestContext, TitoRequestContext},
            proto_wrapper::ProtoGenerateComplete,
            regular::{processor, streaming},
        },
    },
    worker::AttachedBody,
};

/// Chat response processing stage
pub(crate) struct ChatResponseProcessingStage {
    processor: processor::ResponseProcessor,
    streaming_processor: Arc<streaming::StreamingProcessor>,
    tito_store: Option<Arc<TitoStore>>,
}

impl ChatResponseProcessingStage {
    pub fn new(
        processor: processor::ResponseProcessor,
        streaming_processor: Arc<streaming::StreamingProcessor>,
        tito_store: Option<Arc<TitoStore>>,
    ) -> Self {
        Self {
            processor,
            streaming_processor,
            tito_store,
        }
    }
}

#[async_trait]
impl PipelineStage for ChatResponseProcessingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        self.process_chat_response(ctx).await
    }

    fn name(&self) -> &'static str {
        "ChatResponseProcessing"
    }

    fn phase(&self) -> StagePhase {
        StagePhase::PostExecution
    }
}

impl ChatResponseProcessingStage {
    async fn process_chat_response(
        &self,
        ctx: &mut RequestContext,
    ) -> Result<Option<Response>, Response> {
        let is_streaming = ctx.is_streaming();

        // Extract execution result
        let execution_result = ctx.state.response.execution_result.take().ok_or_else(|| {
            error!(
                function = "ChatResponseProcessingStage::execute",
                "No execution result"
            );
            error::internal_error("no_execution_result", "No execution result")
        })?;

        // Get dispatch metadata (needed by both streaming and non-streaming)
        let dispatch = ctx
            .state
            .dispatch
            .as_ref()
            .ok_or_else(|| {
                error!(
                    function = "ChatResponseProcessingStage::execute",
                    "Dispatch metadata not set"
                );
                error::internal_error("dispatch_metadata_not_set", "Dispatch metadata not set")
            })?
            .clone();

        // Take tito_context once — consumed by one of the two paths below.
        let tito_ctx = ctx.state.tito_context.take();

        // Get cached tokenizer (resolved once in preparation stage)
        let tokenizer = ctx.tokenizer_arc().ok_or_else(|| {
            error!(
                function = "ChatResponseProcessingStage::process_chat_response",
                "Tokenizer not cached in context"
            );
            error::internal_error(
                "tokenizer_not_cached",
                "Tokenizer not cached in context - preparation stage may have been skipped",
            )
        })?;

        if is_streaming {
            // Read derived skip_special_tokens set during preparation.
            let skip_special_tokens = ctx
                .state
                .response
                .skip_special_tokens
                .unwrap_or_else(|| ctx.chat_request().skip_special_tokens);

            // Streaming: Use StreamingProcessor and return SSE response
            let response = self.streaming_processor.clone().process_streaming_response(
                execution_result,
                ctx.chat_request_arc(), // Cheap Arc clone (8 bytes)
                dispatch,
                tokenizer,
                skip_special_tokens,
            );

            // Attach load guards to response body for proper RAII lifecycle
            let response = match ctx.state.load_guards.take() {
                Some(guards) => AttachedBody::wrap_response(response, guards),
                None => response,
            };

            return Ok(Some(response));
        }

        // Non-streaming: Delegate to ResponseProcessor
        let request_logprobs = ctx.chat_request().logprobs;

        let chat_request = ctx.chat_request_arc();

        // Single drain — both TITO and response formatting share this Vec.
        let all_completes =
            response_collection::collect_responses(execution_result, request_logprobs).await?;

        // Parse choices once. The same `ChatCompletionMessage` then feeds both TITO
        // storage and the final response: re-parsing here would mint fresh tool-call
        // IDs (`generate_tool_call_id` uses a UUID for non-Kimi models) and break TITO
        // prefix hashing on the next turn.
        let choices = {
            let stop_decoder = ctx.state.response.stop_decoder.as_mut().ok_or_else(|| {
                error!(
                    function = "ChatResponseProcessingStage::execute",
                    "Stop decoder not initialized"
                );
                error::internal_error(
                    "stop_decoder_not_initialized",
                    "Stop decoder not initialized",
                )
            })?;
            // When the request participates in TITO, suppress the
            // OpenAI-shaped `choices[*].logprobs` from the outbound HTTP
            // response: TITO capture reads logprobs straight off the proto in
            // `do_tito_capture_non_streaming`, so the per-token
            // `tokenizer.decode` pass that builds `ChatLogProbsContent` would
            // be pure overhead.
            // Non-TITO callers are unaffected and keep getting the verbose logprobs shape.
            let suppress_logprobs = tito_ctx.is_some();
            self.processor
                .parse_chat_choices(
                    &all_completes,
                    &chat_request,
                    &tokenizer,
                    stop_decoder,
                    suppress_logprobs,
                )
                .await?
        };

        // TITO non-streaming capture: reuse the message we just parsed.
        if let (Some(tc), Some(store)) = (tito_ctx, self.tito_store.as_ref()) {
            let assistant_message = choices.first().map(|c| &c.message).ok_or_else(|| {
                error!(
                    function = "ChatResponseProcessingStage::process_chat_response",
                    session_id = %tc.session_id,
                    "No completion available for TITO assistant capture"
                );
                error::internal_error(
                    "tito_empty_completion",
                    "No completion available for TITO assistant capture",
                )
            })?;
            let dispatched_prompt_start = ctx
                .state
                .partial_rollout_state
                .as_ref()
                .map(|s| s.first_iter_prompt_start)
                .or(ctx
                    .state
                    .partial_rollout_overrides
                    .routed_experts_prompt_start)
                .unwrap_or_else(|| ctx.input.request_type.routed_experts_prompt_start());
            let weight_version = dispatch.weight_version.clone();
            do_tito_capture_non_streaming(
                Arc::clone(store),
                &tc,
                &tokenizer,
                &all_completes,
                assistant_message,
                dispatched_prompt_start,
                weight_version,
            );
        }

        let response = self
            .processor
            .build_chat_response(choices, &all_completes, &dispatch);

        // Store the final response
        ctx.state.response.final_response = Some(FinalResponse::Chat(response));

        Ok(None)
    }
}

/// TITO capture for the non-streaming path.
fn do_tito_capture_non_streaming(
    store: Arc<TitoStore>,
    tito_ctx: &TitoRequestContext,
    tokenizer: &Arc<dyn llm_tokenizer::traits::Tokenizer>,
    all_completes: &[ProtoGenerateComplete],
    assistant_message: &ChatCompletionMessage,
    dispatched_prompt_start: u32,
    weight_version: Option<String>,
) {
    let Some(first_complete) = all_completes.first() else {
        // Caller guarantees a non-empty `choices` list, so this is unreachable.
        // Stay defensive — still log so we notice if assumptions drift.
        error!(
            function = "ChatResponseProcessingStage::do_tito_capture_non_streaming",
            session_id = %tito_ctx.session_id,
            "No completion available for TITO assistant capture"
        );
        return;
    };

    if all_completes.len() > 1 {
        debug!(
            session_id = %tito_ctx.session_id,
            choices = all_completes.len(),
            "TITO capture received multiple choices; using the first choice"
        );
    }

    let request_messages: &[ChatMessage] = tito_ctx.request.messages.as_slice();
    let render_context = &tito_ctx.render_context;

    let model_id = tito_ctx.request.model.as_str();
    let adapter = smg_tito::model_adapter::select_adapter(model_id);
    let max_trim = adapter.max_trim_tokens();
    tracing::debug!(
        session_id = %tito_ctx.session_id,
        model_id = %model_id,
        is_tito_hit = tito_ctx.is_tito_hit,
        adapter_type = std::any::type_name_of_val(&*adapter),
        max_trim_tokens = max_trim,
        "do_tito_capture_non_streaming: processing final response with adapter"
    );
    store.set_session_max_trim_tokens(&tito_ctx.session_id, max_trim);

    let output_ids = first_complete.output_ids();
    let prompt_ids = &tito_ctx.prompt_token_ids;
    let mut full_ids = Vec::with_capacity(prompt_ids.len() + output_ids.len());
    full_ids.extend_from_slice(prompt_ids);
    full_ids.extend_from_slice(output_ids);

    // Build the assistant ChatMessage exactly once: this is the message we
    // hash into the running hasher *and* the message we hand to the
    // diagnostic / mismatch-report paths below.
    //
    // Building it twice (or re-parsing it from the response)
    // would risk minting fresh tool_call IDs and breaking the prefix-hash round trip.
    let new_assistant_message = build_assistant_chat_message(assistant_message);

    // Reuse the prefix hasher captured in preparation to derive the leaf
    // hash without re-walking request_messages.  Clone is mandatory: the
    // hasher state is owned by the immutable ``TitoRequestContext`` and the
    // adapter's `max_trim_tokens` flow ran above without consuming it.
    let mut leaf_hasher = tito_ctx.running_hasher.clone();
    smg_tito::hash_message_into(&mut leaf_hasher, &new_assistant_message);
    let leaf_hash = smg_tito::finalize_hash(&leaf_hasher);
    let parent_hash = tito_ctx.parent_hash;

    let mismatch_report = if store.is_debug() && tito_ctx.is_tito_hit {
        let all_messages = {
            let mut v = Vec::with_capacity(request_messages.len() + 1);
            v.extend_from_slice(request_messages);
            v.push(new_assistant_message);
            v
        };

        let report = build_mismatch_report(
            tito_ctx.is_tito_hit,
            &full_ids,
            &all_messages,
            tokenizer,
            render_context,
            &store,
            model_id,
        );

        debug!(
            session_id = %tito_ctx.session_id,
            total_messages = all_messages.len(),
            assistants = %smg_tito::assistants_diagnostic_summary(&all_messages),
            "TITO store: assistants diagnostic"
        );

        report
    } else {
        vec![]
    };

    tracing::debug!(
        session_id = %tito_ctx.session_id,
        is_tito_hit = tito_ctx.is_tito_hit,
        prompt_len = prompt_ids.len(),
        output_len = output_ids.len(),
        full_ids_len = full_ids.len(),
        mismatch_count = mismatch_report.len(),
        "do_tito_capture_non_streaming: storing tokens"
    );
    let turn_record = extract_turn_record(
        first_complete,
        prompt_ids.len(),
        dispatched_prompt_start,
        weight_version,
        mismatch_report,
    );

    let store_result =
        if let Some(reusable_prompt_ids) = tito_ctx.reusable_prompt_token_ids.as_ref() {
            let mut reusable_full_ids =
                Vec::with_capacity(reusable_prompt_ids.len() + output_ids.len());
            reusable_full_ids.extend_from_slice(reusable_prompt_ids);
            reusable_full_ids.extend_from_slice(output_ids);
            store.store_with_hashes_and_reusable(
                &tito_ctx.session_id,
                leaf_hash,
                parent_hash,
                smg_tito::store::StoredTokenSequences::with_reusable(full_ids, reusable_full_ids),
                turn_record,
                tito_ctx.trajectory_id,
            )
        } else {
            store.store_with_hashes(
                &tito_ctx.session_id,
                leaf_hash,
                parent_hash,
                full_ids,
                turn_record,
                tito_ctx.trajectory_id,
            )
        };

    match store_result {
        Ok(()) => {
            debug!(session_id = %tito_ctx.session_id, "TITO stored generation result");
            // Advance the trajectory's RE offset only on successful store.
            // `prompt_token_count + completion_len - 1` is the absolute
            // position of the last RE row vLLM emitted on this turn — the
            // next turn must start its capture immediately *after* this
            // boundary.
            if !output_ids.is_empty() {
                let captured_upper_bound =
                    (prompt_ids.len() + output_ids.len()).saturating_sub(1) as u32;
                store.advance_routed_experts_offset(
                    &tito_ctx.session_id,
                    tito_ctx.trajectory_id,
                    captured_upper_bound,
                );
            }
        }
        Err(e) => {
            warn!(session_id = %tito_ctx.session_id, error = %e, "TITO store failed (non-fatal)");
        }
    };
}

fn build_mismatch_report(
    is_tito_hit: bool,
    full_ids: &[u32],
    messages: &[ChatMessage],
    tokenizer: &Arc<dyn llm_tokenizer::traits::Tokenizer>,
    render_context: &RenderContext,
    store: &TitoStore,
    model_id: &str,
) -> Vec<smg_tito::MismatchEntry> {
    if !store.is_debug() || !is_tito_hit {
        return vec![];
    }

    let adapter = smg_tito::model_adapter::select_adapter(model_id);
    let assistant_start_str = adapter.assistant_start_str().map(String::from);
    let trim_trailing_ids: std::collections::HashSet<u32> =
        adapter.trailing_token_ids().iter().copied().collect();
    let validator = smg_tito::validator::TokenSeqValidator::new(
        Arc::clone(tokenizer),
        assistant_start_str,
        trim_trailing_ids,
    );
    // full_ids = prompt_ids + output_ids (complete accumulated sequence).
    // Validate against canonical retokenization with add_generation_prompt=false
    // since the assistant turn content is already in `messages`.
    validator.validate(full_ids, messages, false, render_context)
}

/// Build a `TurnRecord` from the selected completed generation.
///
/// Extracts logprobs, finish_reason, and routed-experts metadata once,
/// avoiding per-site boilerplate.
///
/// `dispatched_prompt_start` is the absolute token position vLLM was told
/// to start RE capture from on this turn — the same value that shipped on
/// the wire as `SamplingParams.routed_experts_prompt_start`.
fn extract_turn_record(
    complete: &ProtoGenerateComplete,
    prompt_token_count: usize,
    dispatched_prompt_start: u32,
    weight_version: Option<String>,
    mismatch_report: Vec<smg_tito::MismatchEntry>,
) -> smg_tito::TurnRecord {
    let output_logprobs: Vec<(f32, u32)> = complete
        .output_logprobs()
        .map(|lp| {
            lp.token_logprobs
                .iter()
                .zip(lp.token_ids.iter())
                .map(|(prob, id)| (*prob, *id))
                .collect()
        })
        .unwrap_or_default();
    let finish_reason = complete.finish_reason().to_string();
    let routed_experts = complete.routed_experts().map(|re| {
        let dtype = match re.dtype {
            crate::routers::grpc::proto_wrapper::RoutedExpertsDtype::U8 => {
                smg_tito::TurnRoutedExpertsDtype::U8
            }
            crate::routers::grpc::proto_wrapper::RoutedExpertsDtype::U16 => {
                smg_tito::TurnRoutedExpertsDtype::U16
            }
        };
        smg_tito::TurnRoutedExperts {
            data: Arc::new(re.data.to_vec()),
            num_layers: re.num_layers,
            top_k: re.top_k,
            dtype,
            prompt_start: dispatched_prompt_start,
        }
    });
    smg_tito::TurnRecord {
        prompt_token_count,
        output_logprobs: if output_logprobs.is_empty() {
            None
        } else {
            Some(output_logprobs)
        },
        finish_reason,
        mismatch_report,
        routed_experts,
        weight_version,
    }
}

/// Build the [`ChatMessage::Assistant`] view of a server-parsed
/// [`ChatCompletionMessage`].
///
/// The same `ChatCompletionMessage` must feed both the response that
/// goes back to the client *and* the assistant turn
/// that gets hashed into the TITO prefix tree.
fn build_assistant_chat_message(assistant_message: &ChatCompletionMessage) -> ChatMessage {
    ChatMessage::Assistant {
        content: assistant_message
            .content
            .as_ref()
            .map(|content| MessageContent::Text(content.clone())),
        name: None,
        tool_calls: assistant_message.tool_calls.clone(),
        reasoning_content: assistant_message.reasoning_content.clone(),
    }
}
