//! Chat preparation stage: Filter tools, process messages, tokenize, build constraints

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use llm_multimodal::Modality;
use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{ToolChoice, ToolChoiceValue},
};
use smg_tito::{
    engine::TitoEngine, model_adapter, PrefixLookup, TitoStore, TITO_SESSION_HEADER,
    TITO_TRAJECTORY_ID_HEADER,
};
use tracing::{debug, error, warn};

use crate::routers::{
    error,
    grpc::{
        common::stages::{PipelineStage, StagePhase},
        context::{PreparationOutput, RequestContext, RequestType, TitoRequestContext},
        multimodal, utils, ProcessedMessages,
    },
};

/// Chat preparation stage
///
/// Extracts chat-specific preparation logic from the old unified PreparationStage.
/// This is a direct extraction without architectural changes.
pub(crate) struct ChatPreparationStage {
    tito_store: Option<Arc<TitoStore>>,
}

impl ChatPreparationStage {
    pub fn new(tito_store: Option<Arc<TitoStore>>) -> Self {
        Self { tito_store }
    }
}

#[async_trait]
impl PipelineStage for ChatPreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let request = ctx.chat_request_arc();
        self.prepare_chat(ctx, &request).await?;
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "ChatPreparation"
    }

    fn phase(&self) -> StagePhase {
        StagePhase::Preparation
    }
}

impl ChatPreparationStage {
    async fn prepare_chat(
        &self,
        ctx: &mut RequestContext,
        request: &ChatCompletionRequest,
    ) -> Result<(), Response> {
        // Step 0: Resolve tokenizer from registry (cached for reuse in response processing)
        let tokenizer =
            utils::resolve_tokenizer(ctx, "ChatPreparationStage::prepare_chat").map_err(|e| *e)?;

        // Step 1: Filter tools if needed
        let body_ref = utils::filter_chat_request_by_tool_choice(request);

        // Normalize media once. The same plan drives placeholder resolution,
        // rendering, fetching, preprocessing, and final count validation.
        let media_plan = multimodal::media_plan_chat(&request.messages);
        let (placeholder_tokens, mm_context) = if media_plan.is_empty() {
            (None, None)
        } else if let Some(mm_components) = ctx.components.multimodal.as_ref() {
            let model_id = ctx.input.model_id.clone();
            let entry = ctx
                .components
                .tokenizer_registry
                .get_by_name(&model_id)
                .or_else(|| ctx.components.tokenizer_registry.get_by_id(&model_id));

            let (tokenizer_id, tokenizer_source) = match entry {
                Some(e) => (e.id.clone(), e.source.clone()),
                None => {
                    error!(
                        function = "ChatPreparationStage::execute",
                        model = %model_id,
                        "Tokenizer entry not found for multimodal processing"
                    );
                    return Err(error::bad_request(
                        "multimodal_config_missing",
                        format!("Tokenizer not found for model: {model_id}"),
                    ));
                }
            };

            let placeholders = multimodal::prepare_placeholder_tokens(
                &media_plan,
                &model_id,
                &*tokenizer,
                mm_components,
                &tokenizer_id,
                &tokenizer_source,
            )
            .await
            .map_err(|e| {
                error!(
                    function = "ChatPreparationStage::execute",
                    model = %model_id,
                    error = %e,
                    "Failed to prepare multimodal prompt plan"
                );
                error::bad_request(
                    "invalid_multimodal_request",
                    format!("Invalid multimodal request: {e}"),
                )
            })?;

            (
                Some(placeholders),
                Some((
                    Arc::clone(mm_components),
                    model_id,
                    tokenizer_id,
                    tokenizer_source,
                    media_plan,
                )),
            )
        } else {
            error!(
                function = "ChatPreparationStage::execute",
                "Multimodal content detected but multimodal components not initialized"
            );
            return Err(error::bad_request(
                "multimodal_not_supported",
                "Multimodal content detected but multimodal processing is not available",
            ));
        };

        // TITO always operates on unexpanded anchors. On a hit the reusable
        // prefix is merged with only the appended messages; the complete media
        // plan below then rebuilds bindings/tensors for every historical item.
        let image_placeholder = placeholder_tokens
            .as_ref()
            .and_then(|tokens| tokens.get(Modality::Image));
        let video_placeholder = placeholder_tokens
            .as_ref()
            .and_then(|tokens| tokens.get(Modality::Video));
        let audio_placeholder = placeholder_tokens
            .as_ref()
            .and_then(|tokens| tokens.get(Modality::Audio));
        let tito_token_ids = self.try_tito(
            ctx,
            body_ref.as_ref(),
            &tokenizer,
            image_placeholder,
            video_placeholder,
            audio_placeholder,
        )?;

        let (mut token_ids, processed_messages) = if let Some(ids) = tito_token_ids {
            (
                ids,
                ProcessedMessages {
                    text: String::new(),
                    multimodal_intermediate: None,
                    stop_sequences: body_ref.stop.clone(),
                },
            )
        } else {
            // Process messages and apply chat template
            let processed_messages = match utils::process_chat_messages_with_placeholders(
                &body_ref,
                &*tokenizer,
                placeholder_tokens.as_ref(),
            ) {
                Ok(msgs) => msgs,
                Err(e) => {
                    error!(function = "ChatPreparationStage::execute", error = %e, "Failed to process chat messages");
                    return Err(error::bad_request("process_messages_failed", e));
                }
            };

            // Tokenize the processed text (no special tokens - chat template already handles them)
            let encoding = match utils::encode_blocking(
                tokenizer.clone(),
                processed_messages.text.clone(),
                false,
            )
            .await
            {
                Ok(encoding) => encoding,
                Err(e) => {
                    error!(function = "ChatPreparationStage::execute", error = %e, "Tokenization failed");
                    return Err(error::internal_error(
                        "tokenization_failed",
                        format!("Tokenization failed: {e}"),
                    ));
                }
            };

            (encoding.token_ids().to_vec(), processed_messages)
        };

        if let (Some(placeholders), Some((_, _, _, _, media_plan))) =
            (placeholder_tokens.as_ref(), mm_context.as_ref())
        {
            multimodal::validate_rendered_media_anchors(
                media_plan,
                placeholders,
                &*tokenizer,
                &token_ids,
            )
            .map_err(|error| {
                error!(
                    function = "ChatPreparationStage::execute",
                    %error,
                    "Rendered multimodal anchors do not match request media"
                );
                error::bad_request("multimodal_prompt_contract_mismatch", error.to_string())
            })?;
        }

        // Only multimodal processing can replace media anchors with expanded
        // model tokens. Pure-text TITO reuses `token_ids` without a second copy.
        let reusable_prompt_token_ids = mm_context.as_ref().map(|_| token_ids.clone());

        // Step 4: Full multimodal processing (fetch + preprocess + expand tokens + hash)
        let mut multimodal_intermediate = None;
        if let Some((mm_components, model_id, tokenizer_id, tokenizer_source, media_plan)) =
            mm_context
        {
            match multimodal::process_multimodal_plan(
                media_plan,
                &model_id,
                &*tokenizer,
                token_ids,
                &mm_components,
                &tokenizer_id,
                &tokenizer_source,
            )
            .await
            {
                Ok(output) => {
                    debug!(
                        function = "ChatPreparationStage::execute",
                        expanded_tokens = output.expanded_token_ids.len(),
                        "Multimodal processing complete"
                    );
                    token_ids = output.expanded_token_ids;
                    multimodal_intermediate = Some(output.intermediate);
                }
                Err(e) => {
                    error!(
                        function = "ChatPreparationStage::execute",
                        error = %e,
                        "Multimodal processing failed"
                    );
                    return Err(error::bad_request(
                        "multimodal_processing_failed",
                        format!("Multimodal processing failed: {e}"),
                    ));
                }
            }
        }

        // Step 3.5: Enforce the session's prompt-too-long budget. When the
        // accumulated prompt exceeds the training compaction limit, return an
        // Anthropic `prompt_too_long` error so Claude Code reactively compacts
        // instead of growing past the budget (or hitting the engine overflow
        // with a message Claude Code cannot recognize).
        if let Some(limit) = error::prompt_too_long_limit(ctx.input.headers.as_ref())? {
            let prompt_len = token_ids.len();
            if error::prompt_exceeds_limit(prompt_len, limit) {
                warn!(
                    function = "ChatPreparationStage::execute",
                    prompt_len,
                    limit,
                    "Prompt exceeded compaction budget; returning prompt_too_long for reactive compact"
                );
                return Err(error::prompt_too_long(prompt_len, limit));
            }
        }

        // Step 4: Build tool constraints if needed
        // The tool parser registry handles both structural tag (for native format
        // parsers like Mistral, KimiK2) and generic JSON schema fallback.
        let tool_call_constraint = if let (Some(tools), Some(tool_choice)) =
            (body_ref.tools.as_ref(), request.tool_choice.as_ref())
        {
            ctx.components
                .tool_parser_factory
                .registry()
                .generate_tool_constraint(
                    ctx.components.configured_tool_parser.as_deref(),
                    tools,
                    tool_choice,
                )
                .map_err(|e| {
                    error!(function = "ChatPreparationStage::execute", error = %e, "Invalid tool configuration");
                    error::bad_request(
                        "invalid_tool_configuration",
                        format!("Invalid tool configuration: {e}"),
                    )
                })?
        } else {
            None
        };

        // Derive skip_special_tokens from constraint type:
        // - json_schema: backend forces JSON, no trigger tokens to preserve
        // - structural_tag or no constraint (auto): parser needs trigger tokens
        let skip_special_tokens = match &tool_call_constraint {
            Some(c) if c.is_json_schema() => request.skip_special_tokens,
            _ if request.tools.is_some()
                && !matches!(
                    request.tool_choice,
                    Some(ToolChoice::Value(ToolChoiceValue::None))
                ) =>
            {
                false
            }
            _ => request.skip_special_tokens,
        };

        // Step 5: Create stop sequence decoder (build once, reuse in non-stream)
        let stop_decoder = utils::create_stop_decoder(
            &tokenizer,
            request.stop.as_ref(),
            request.stop_token_ids.as_ref(),
            skip_special_tokens,
            request.no_stop_trim,
            request.ignore_eos,
        );

        let mut processed_messages = processed_messages;
        processed_messages.multimodal_intermediate = multimodal_intermediate;

        // Persist prompt token IDs into tito_context before PreparationOutput is consumed
        // by request_building (which .take()s preparation).
        if let Some(ref mut tc) = ctx.state.tito_context {
            tc.prompt_token_ids.clone_from(&token_ids);
            tc.reusable_prompt_token_ids = reusable_prompt_token_ids;
        }

        // Store results in context
        ctx.state.preparation = Some(PreparationOutput::Chat {
            token_ids,
            processed_messages,
            tool_constraints: tool_call_constraint.map(|c| c.to_tuple()),
        });

        // Store stop decoder and derived skip_special_tokens for response processing.
        // Stored on ResponseState because PreparationOutput is consumed by
        // request_building before response_processing runs.
        ctx.state.response.stop_decoder = Some(stop_decoder);
        ctx.state.response.skip_special_tokens = Some(skip_special_tokens);

        Ok(())
    }

    /// Attempt a TITO prefix lookup and incremental merge
    #[expect(
        clippy::result_large_err,
        reason = "pipeline stages consistently return Axum Response errors"
    )]
    fn try_tito(
        &self,
        ctx: &mut RequestContext,
        request: &ChatCompletionRequest,
        tokenizer: &Arc<dyn llm_tokenizer::traits::Tokenizer>,
        image_placeholder: Option<&str>,
        video_placeholder: Option<&str>,
        audio_placeholder: Option<&str>,
    ) -> Result<Option<Vec<u32>>, Response> {
        let store = match self.tito_store.as_ref() {
            Some(s) => s,
            None => return Ok(None),
        };

        // Only Chat requests participate in TITO (already gated by caller, but be explicit)
        if !matches!(ctx.input.request_type, RequestType::Chat(_)) {
            return Ok(None);
        }

        // Read session-id header
        let session_id = match ctx
            .input
            .headers
            .as_ref()
            .and_then(|h| h.get(TITO_SESSION_HEADER))
            .and_then(|v| v.to_str().ok())
        {
            Some(id) => id.to_owned(),
            None => return Ok(None),
        };

        // Manual mode preserves the existing default-to-zero behavior. Auto mode
        // ignores this value after prefix lookup and resolves from the live leaves.
        let manual_trajectory_id: u64 = ctx
            .input
            .headers
            .as_ref()
            .and_then(|h| h.get(TITO_TRAJECTORY_ID_HEADER))
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let request_arc = Arc::new(request.clone());
        let messages = request.messages.as_slice();
        let render_context = utils::get_render_context_from_request(request, image_placeholder).map(|context| {
            context.with_media_placeholders(
                image_placeholder.map(String::from),
                video_placeholder.map(String::from),
                audio_placeholder.map(String::from),
            )
        }).map_err(|e| {
            error!(function = "ChatPreparationStage::try_tito", error = %e, "Failed to build TITO render context");
            error::bad_request("tito_render_context_failed", e)
        })?;

        // Select once from exact server-side metadata. The same immutable
        // adapter is shared by preparation and capture, avoiding a second
        // special-token lookup/newline encode on every request.
        let adapter: Arc<dyn model_adapter::ModelAdapter> =
            model_adapter::select_adapter_for_tokenizer(&**tokenizer)
                .map(Into::into)
                .map_err(|e| {
                    error!(
                        function = "ChatPreparationStage::try_tito",
                        session_id = %session_id,
                        error = %e,
                        "Failed to select TITO model adapter from tokenizer metadata"
                    );
                    error::bad_request("tito_model_adapter_error", e.to_string())
                })?;

        debug!(
            session_id = %session_id,
            total_messages = messages.len(),
            assistants = %smg_tito::assistants_diagnostic_summary(messages),
            "TITO find_prefix: assistants diagnostic"
        );

        let lookup: PrefixLookup =
            match store.find_prefix_with_lookup(&session_id, messages, &render_context) {
                Ok(lookup) => lookup,
                Err(e) => {
                    warn!(session_id = %session_id, error = %e, "TITO find_prefix error");
                    return Err(error::bad_request(
                        "tito_invalid_appended_messages",
                        e.to_string(),
                    ));
                }
            };

        // Stash the running hasher state and parent hash into the TITO context
        // so the response stage can derive the leaf hash by extending this
        // hasher with the new assistant message.
        let running_hasher = lookup.running_hasher.clone();
        let is_compaction = lookup
            .matched
            .as_ref()
            .is_some_and(|prefix| prefix.has_assistant_in_appended);
        // Compacted prompts are verified as a complete new prompt and become a
        // new root/branch. They must never retain a parent edge to a trajectory
        // whose token stream they rewrote.
        let parent_hash = if lookup.matched.is_some() && !is_compaction {
            lookup.parent_hash
        } else {
            None
        };

        let resolved_trajectory = store
            .resolve_trajectory_id(&session_id, manual_trajectory_id, parent_hash)
            .map_err(|e| {
                error!(
                    function = "ChatPreparationStage::try_tito",
                    session_id = %session_id,
                    error = %e,
                    "Failed to resolve TITO trajectory ID"
                );
                error::internal_error("tito_trajectory_id_resolution_failed", e.to_string())
            })?;
        let trajectory_id = resolved_trajectory.trajectory_id;

        ctx.state.tito_context = Some(TitoRequestContext {
            session_id: session_id.clone(),
            request: request_arc,
            render_context: render_context.clone(),
            model_adapter: Arc::clone(&adapter),
            is_tito_hit: false,
            matched_message_num: 0,
            trajectory_id,
            trajectory_id_reservation: resolved_trajectory.reservation,
            prompt_token_ids: Vec::new(),
            reusable_prompt_token_ids: None,
            running_hasher,
            parent_hash,
        });

        // The gateway picks `prompt_start` for every turn from the
        // trajectory's RE offset store, so each turn captures only
        // the *new* token positions appended since the previous turn.
        let re_prompt_start = store.next_routed_experts_prompt_start(&session_id, trajectory_id);
        ctx.state
            .partial_rollout_overrides
            .routed_experts_prompt_start = Some(re_prompt_start);

        let prefix_match = match lookup.matched {
            Some(pm) => {
                debug!(
                    session_id = %session_id,
                    matched_messages = pm.matched_message_num,
                    prefix_token_len = pm.pretokenized_ids.len(),
                    total_messages = messages.len(),
                    "TITO HIT — found cached prefix"
                );
                if store.is_debug() && pm.has_assistant_in_appended {
                    debug!(
                        session_id = %session_id,
                        matched_messages = pm.matched_message_num,
                        appended_messages = %smg_tito::messages_structure_summary(
                            &messages[pm.matched_message_num..]
                        ),
                        "TITO compacted-context hit — appended assistant turn(s) will be \
                         re-tokenized incrementally and treated as prompt (mask 0)"
                    );
                }
                pm
            }
            None => {
                debug!(
                    session_id = %session_id,
                    total_messages = messages.len(),
                    "TITO MISS — no cached prefix found, falling through to full retokenize"
                );
                return Ok(None);
            }
        };

        let matched_message_num = prefix_match.matched_message_num;
        let prefix_token_len = prefix_match.pretokenized_ids.len();
        if matched_message_num == 0 || prefix_token_len == 0 {
            debug!(
                session_id = %session_id,
                matched_message_num = matched_message_num,
                prefix_token_len = prefix_token_len,
                "TITO pseudo-hit without reusable prefix tokens — falling through to full retokenize"
            );
            ctx.state.tito_context.take();
            return Ok(None);
        }

        debug!(
            session_id = %session_id,
            matched_message_num = matched_message_num,
            prefix_token_len = prefix_token_len,
            "TITO hit — running merge_incremental"
        );

        let appended = &messages[matched_message_num..];
        // Mirror the production render path: project `reasoning_effort` onto the
        // template's thinking toggle so the incremental render's generation
        // prompt / thinking-mode wrapping matches what the full render produces.
        let thinking = openai_protocol::chat::thinking_from_reasoning_effort(
            request.reasoning_effort.as_deref(),
        );
        let merged_ids = match TitoEngine::merge_incremental(
            prefix_match,
            appended,
            &**tokenizer,
            &*adapter,
            &render_context,
            thinking,
        ) {
            Ok(ids) => ids,
            Err(e) => {
                warn!(session_id = %session_id, error = %e, "TITO merge_incremental failed — falling through");
                // A failed token-prefix proof is a genuine TITO miss. Do not
                // let response processing store a node derived from this hit.
                ctx.state.tito_context.take();
                return Ok(None);
            }
        };

        if is_compaction {
            let canonical_ids = match TitoEngine::tokenize_full_prompt(
                messages,
                &**tokenizer,
                &render_context,
                thinking,
            ) {
                Ok(ids) => ids,
                Err(e) => {
                    warn!(session_id = %session_id, error = %e, "TITO compaction verification failed — falling through");
                    ctx.state.tito_context.take();
                    return Ok(None);
                }
            };
            if merged_ids != canonical_ids {
                warn!(
                    session_id = %session_id,
                    merged_tokens = merged_ids.len(),
                    canonical_tokens = canonical_ids.len(),
                    "TITO compacted prompt differs from canonical tokenization — falling through"
                );
                ctx.state.tito_context.take();
                return Ok(None);
            }
        }

        if let Some(ref mut tc) = ctx.state.tito_context {
            tc.is_tito_hit = true;
            tc.matched_message_num = matched_message_num;
        }
        Ok(Some(merged_ids))
    }
}
