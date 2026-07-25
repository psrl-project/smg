use llm_tokenizer::{chat_template::ChatTemplateParams, traits::Tokenizer};
use openai_protocol::chat::{ChatMessage, MessageContent};

use crate::{
    error::TitoError,
    model_adapter::ModelAdapter,
    normalizer::RenderContext,
    store::PrefixMatch,
    validator::{messages_to_template_values, messages_to_template_values_with_context},
};

pub struct TitoEngine;

/// How an appended message segment should be tokenized incrementally.
#[derive(Debug, Clone, Copy)]
enum SegmentKind {
    /// Consecutive `Tool` messages following an assistant tool-call turn.
    ///
    /// Base context: `[dummy_system, dummy_assistant(tool_calls)]`.
    /// This mirrors the template rendering context that tool messages actually
    /// appear in — immediately after an assistant turn that issued the tool calls.
    Tool,
    /// A single `user`, `system`, or `developer` message.
    ///
    /// Base context: `[dummy_system]`.
    UserLike,
}

/// Split `appended` into typed segments for incremental tokenisation.
///
/// Consecutive `Tool` messages are collapsed into one `Tool` segment so that
/// all tool result messages belonging to the same assistant tool-call are
/// tokenized with the same dummy context.  Every non-tool message becomes its
/// own `UserLike` segment.
fn split_into_segments(appended: &[ChatMessage]) -> Vec<(SegmentKind, &[ChatMessage])> {
    let mut segments: Vec<(SegmentKind, &[ChatMessage])> = Vec::new();
    let mut i = 0;
    while i < appended.len() {
        if matches!(appended[i], ChatMessage::Tool { .. }) {
            let start = i;
            while i < appended.len() && matches!(appended[i], ChatMessage::Tool { .. }) {
                i += 1;
            }
            segments.push((SegmentKind::Tool, &appended[start..i]));
        } else {
            segments.push((SegmentKind::UserLike, &appended[i..=i]));
            i += 1;
        }
    }
    segments
}

impl TitoEngine {
    /// Merge a pretokenized prefix with the incremental token IDs for `appended_messages`.
    ///
    /// `appended_messages` is split into typed segments (tool runs vs. user-like messages)
    /// and each segment is tokenized with the appropriate dummy base context, mirroring
    /// how the chat template actually renders those messages in production.
    pub fn merge_incremental(
        prefix_match: PrefixMatch,
        appended_messages: &[ChatMessage],
        tokenizer: &dyn Tokenizer,
        adapter: &dyn ModelAdapter,
        render_context: &RenderContext,
    ) -> Result<Vec<u32>, TitoError> {
        if appended_messages.is_empty() {
            return Ok(adapter.adjust_prefix_boundary(&prefix_match.pretokenized_ids));
        }

        let segments = split_into_segments(appended_messages);
        let num_segments = segments.len();

        let mut all_incremental: Vec<u32> = Vec::new();

        for (seg_idx, (kind, segment)) in segments.iter().enumerate() {
            let add_generation_prompt = seg_idx == num_segments - 1;
            let incremental = tokenize_segment_incremental(
                segment,
                *kind,
                add_generation_prompt,
                adapter,
                tokenizer,
                render_context,
            )?;
            all_incremental.extend_from_slice(&incremental);
        }

        let mut result = adapter.adjust_prefix_boundary(&prefix_match.pretokenized_ids);
        let adjusted_prefix_len = result.len();
        result.extend_from_slice(&all_incremental);

        tracing::debug!(
            prefix_len = prefix_match.pretokenized_ids.len(),
            adjusted_prefix_len = adjusted_prefix_len,
            incremental_len = all_incremental.len(),
            result_len = result.len(),
            segment_count = num_segments,
            matched_messages = prefix_match.matched_message_num,
            adapter_type = std::any::type_name_of_val(adapter),
            "merge_incremental: merge complete"
        );

        Ok(result)
    }
}

/// Tokenize one appended segment and return only the incremental token IDs.
///
/// Uses role-appropriate dummy base context:
/// - Tool segments → `[dummy_system, dummy_assistant(tool_calls)]`
/// - UserLike segments → `[dummy_system]`
///
/// Also validates the append-only invariant: the fully-rendered text must start
/// with the base-only rendered text.  If violated, the chat template is not
/// truly incremental and TITO token IDs would be incorrect.
fn tokenize_segment_incremental(
    segment: &[ChatMessage],
    kind: SegmentKind,
    add_generation_prompt: bool,
    adapter: &dyn ModelAdapter,
    tokenizer: &dyn Tokenizer,
    render_context: &RenderContext,
) -> Result<Vec<u32>, TitoError> {
    let dummy_system = ChatMessage::System {
        content: MessageContent::Text("_".to_string()),
        name: None,
    };

    let base: Vec<ChatMessage> = match kind {
        SegmentKind::Tool => {
            // The dummy assistant mirrors the tool_call_ids that the preceding
            // assistant turn would have emitted, so the template renders
            // `<|observation|>` / `<tool_response>` tokens correctly.
            let dummy_assistant = adapter.build_dummy_assistant(segment);
            vec![dummy_system, dummy_assistant]
        }
        SegmentKind::UserLike => {
            vec![dummy_system]
        }
    };

    let content_format = tokenizer.chat_template_content_format();

    let base_values = messages_to_template_values(&base, content_format, None)
        .map_err(|e| TitoError::EngineFailed(format!("serialize base: {e}")))?;
    let base_params = ChatTemplateParams {
        add_generation_prompt: false,
        tools: render_context.tools_ref(),
        template_kwargs: render_context.template_kwargs_ref(),
        ..Default::default()
    };
    let base_text = tokenizer
        .apply_chat_template(&base_values, base_params)
        .map_err(|e| TitoError::EngineFailed(e.to_string()))?;

    let mut full_msgs = base;
    full_msgs.extend_from_slice(segment);
    let full_values =
        messages_to_template_values_with_context(&full_msgs, content_format, render_context)
            .map_err(|e| TitoError::EngineFailed(format!("serialize full: {e}")))?;
    let full_params = ChatTemplateParams {
        add_generation_prompt,
        tools: render_context.tools_ref(),
        template_kwargs: render_context.template_kwargs_ref(),
        ..Default::default()
    };
    let full_text = tokenizer
        .apply_chat_template(&full_values, full_params)
        .map_err(|e| TitoError::EngineFailed(e.to_string()))?;

    // Append-only invariant: the chat template must not rewrite already-rendered content.
    if !full_text.starts_with(&base_text) {
        tracing::warn!(
            segment_kind = ?kind,
            base_len = base_text.len(),
            full_len = full_text.len(),
            "merge_incremental: chat template violated append-only invariant",
        );
        return Err(TitoError::EngineFailed(
            "chat template violated append-only invariant".to_string(),
        ));
    }

    let full_ids = tokenizer
        .encode(&full_text, false)
        .map_err(|e| TitoError::EngineFailed(e.to_string()))?
        .token_ids()
        .to_vec();

    let base_ids = tokenizer
        .encode(&base_text, false)
        .map_err(|e| TitoError::EngineFailed(e.to_string()))?
        .token_ids()
        .to_vec();

    Ok(full_ids[base_ids.len()..].to_vec())
}
