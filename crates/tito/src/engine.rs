use llm_tokenizer::{chat_template::ChatTemplateParams, traits::Tokenizer};
use openai_protocol::chat::{ChatMessage, MessageContent};

use crate::{
    error::TitoError, model_adapter::ModelAdapter, normalizer::RenderContext, store::PrefixMatch,
    validator::messages_to_template_values_with_context,
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
    /// A single `assistant` message present in the appended slice.
    ///
    /// This occurs when the harness has **compacted** the conversation (e.g.
    /// Claude Code auto-compact): the request's message list contains assistant
    /// turns that this session's model never generated.  For Qwen3-style
    /// templates an assistant turn renders only from its own fields
    /// (`content` / `reasoning_content` / `tool_calls`), so the incremental
    /// slice can be produced with the same base context as a user message.
    ///
    /// Base context: `[dummy_system]`.
    Assistant,
}

/// Split `appended` into typed segments for incremental tokenisation.
///
/// Consecutive `Tool` messages are collapsed into one `Tool` segment so that
/// all tool result messages belonging to the same assistant tool-call are
/// tokenized with the same dummy context.  Every non-tool message becomes its
/// own `UserLike` or `Assistant` segment (assistant turns appear in the
/// appended slice when the harness has compacted the conversation).
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
            let kind = if matches!(appended[i], ChatMessage::Assistant { .. }) {
                SegmentKind::Assistant
            } else {
                SegmentKind::UserLike
            };
            segments.push((kind, &appended[i..=i]));
            i += 1;
        }
    }
    segments
}

impl TitoEngine {
    /// Canonically render and tokenize a complete generation prompt.
    pub fn tokenize_full_prompt(
        messages: &[ChatMessage],
        tokenizer: &dyn Tokenizer,
        render_context: &RenderContext,
        thinking: Option<bool>,
    ) -> Result<Vec<u32>, TitoError> {
        let text = render_append_only(messages, true, tokenizer, render_context, thinking)?;
        encode_ids(tokenizer, &text)
    }

    /// Merge a pretokenized prefix with the incremental token IDs for `appended_messages`.
    ///
    /// `thinking` must mirror the value the production render path used
    /// (`thinking_from_reasoning_effort(request.reasoning_effort)`), otherwise
    /// the generation prompt / thinking-mode wrapping diverges.
    ///
    /// For position-independent templates
    /// (`Tokenizer::chat_template_is_position_dependent()` == false),
    /// `appended_messages` is split into typed segments (tool runs
    /// vs. user-like/assistant messages) and each segment is tokenized with the
    /// appropriate dummy base context.  For position-dependent templates (e.g.
    /// Qwen3.5) the whole appended slice is rendered in one pass behind a fixed
    /// dummy prefix instead (see [`merge_whole_slice`]).
    pub fn merge_incremental(
        prefix_match: PrefixMatch,
        appended_messages: &[ChatMessage],
        tokenizer: &dyn Tokenizer,
        adapter: &dyn ModelAdapter,
        render_context: &RenderContext,
        thinking: Option<bool>,
    ) -> Result<Vec<u32>, TitoError> {
        if appended_messages.is_empty() {
            return Ok(adapter.adjust_prefix_boundary(&prefix_match.pretokenized_ids));
        }

        let position_dependent = tokenizer.chat_template_is_position_dependent();
        let all_incremental = if position_dependent {
            merge_whole_slice(appended_messages, tokenizer, render_context, thinking)?
        } else {
            let segments = split_into_segments(appended_messages);
            let num_segments = segments.len();

            // The `[dummy_system]` base is identical for every UserLike /
            // Assistant segment; render + encode it once and reuse it.
            let dummy_system = ChatMessage::System {
                content: MessageContent::Text("_".to_string()),
                name: None,
            };
            let userlike_base = render_segment_base(
                vec![dummy_system.clone()],
                tokenizer,
                render_context,
                thinking,
            )?;

            let mut all_incremental: Vec<u32> = Vec::new();

            for (seg_idx, (kind, segment)) in segments.iter().enumerate() {
                let add_generation_prompt = seg_idx == num_segments - 1;
                let incremental = match kind {
                    // The dummy assistant mirrors the tool_call_ids that the
                    // preceding assistant turn would have emitted, so the
                    // template renders `<|observation|>` / `<tool_response>`
                    // tokens correctly.
                    SegmentKind::Tool => {
                        let tool_base = render_segment_base(
                            vec![dummy_system.clone(), adapter.build_dummy_assistant(segment)],
                            tokenizer,
                            render_context,
                            thinking,
                        )?;
                        tokenize_segment_incremental(
                            segment,
                            add_generation_prompt,
                            tokenizer,
                            render_context,
                            thinking,
                            &tool_base,
                        )?
                    }
                    SegmentKind::UserLike | SegmentKind::Assistant => tokenize_segment_incremental(
                        segment,
                        add_generation_prompt,
                        tokenizer,
                        render_context,
                        thinking,
                        &userlike_base,
                    )?,
                };
                all_incremental.extend_from_slice(&incremental);
            }
            all_incremental
        };

        let mut result = adapter.adjust_prefix_boundary(&prefix_match.pretokenized_ids);
        let adjusted_prefix_len = result.len();
        result.extend_from_slice(&all_incremental);

        tracing::debug!(
            prefix_len = prefix_match.pretokenized_ids.len(),
            adjusted_prefix_len = adjusted_prefix_len,
            incremental_len = all_incremental.len(),
            result_len = result.len(),
            position_dependent,
            matched_messages = prefix_match.matched_message_num,
            adapter_type = std::any::type_name_of_val(adapter),
            "merge_incremental: merge complete"
        );

        Ok(result)
    }
}

/// Tokenize the entire appended slice in one pass behind a fixed dummy prefix
/// `[dummy_system, dummy_user]`.
///
/// Position-dependent templates decide an assistant turn's rendering from the
/// index of the last *real* user query (e.g. Qwen3.5's `ns.last_query_index`:
/// `loop.index0 > last_query_index` wraps the turn in `<think>`).  That
/// comparison is **translation-invariant** under a fixed-length prefix: the
/// message index and the last-user index shift by the same amount, so every
/// appended message's wrap state in the dummy context equals the production
/// context regardless of the real prefix content.
///
/// The dummy user is mandatory: such templates raise `No user query found`
/// without a real user message, and it provides the baseline
/// `last_query_index`.  The base's rendered bytes are subtracted from the
/// result, so only the appended slice (plus the generation prompt, since every
/// chat request is a generation request) enters the incremental token stream.
fn merge_whole_slice(
    appended_messages: &[ChatMessage],
    tokenizer: &dyn Tokenizer,
    render_context: &RenderContext,
    thinking: Option<bool>,
) -> Result<Vec<u32>, TitoError> {
    let dummy_system = ChatMessage::System {
        content: MessageContent::Text("_".to_string()),
        name: None,
    };
    // A real user query: content must not look like a `<tool_response>` wrapper.
    let dummy_user = ChatMessage::User {
        content: MessageContent::Text("_".to_string()),
        name: None,
    };
    let base = vec![dummy_system, dummy_user];

    let base_text = render_append_only(&base, false, tokenizer, render_context, thinking)?;

    let mut full_msgs = base;
    full_msgs.extend_from_slice(appended_messages);
    let full_text = render_append_only(&full_msgs, true, tokenizer, render_context, thinking)?;

    let full_ids = encode_ids(tokenizer, &full_text)?;
    let base_ids = encode_ids(tokenizer, &base_text)?;

    if !full_ids.starts_with(&base_ids) {
        tracing::warn!(
            base_tokens = base_ids.len(),
            full_tokens = full_ids.len(),
            "merge_whole_slice: token-prefix invariant failed",
        );
        return Err(TitoError::TokenPrefixMismatch {
            path: "whole-slice",
        });
    }

    Ok(full_ids[base_ids.len()..].to_vec())
}

/// A rendered dummy base context: the base messages themselves (prepended to a
/// segment for the full render) and their token IDs (token-prefix validation
/// and incremental slice offset).
struct SegmentBase {
    messages: Vec<ChatMessage>,
    ids: Vec<u32>,
}

/// Render + encode a dummy base context once, so it can be reused across
/// segments that share the same base (e.g. every UserLike/Assistant segment
/// uses `[dummy_system]`).
fn render_segment_base(
    base_messages: Vec<ChatMessage>,
    tokenizer: &dyn Tokenizer,
    render_context: &RenderContext,
    thinking: Option<bool>,
) -> Result<SegmentBase, TitoError> {
    let content_format = tokenizer.chat_template_content_format();
    let values =
        messages_to_template_values_with_context(&base_messages, content_format, render_context)
            .map_err(|e| TitoError::EngineFailed(format!("serialize base: {e}")))?;
    let text = tokenizer
        .apply_chat_template(&values, template_params(false, thinking, render_context))
        .map_err(|e| TitoError::EngineFailed(e.to_string()))?;
    let ids = encode_ids(tokenizer, &text)?;
    Ok(SegmentBase {
        messages: base_messages,
        ids,
    })
}

fn template_params(
    add_generation_prompt: bool,
    thinking: Option<bool>,
    render_context: &RenderContext,
) -> ChatTemplateParams<'_> {
    ChatTemplateParams {
        add_generation_prompt,
        thinking,
        tools: render_context.tools_ref(),
        template_kwargs: render_context.template_kwargs_ref(),
        ..Default::default()
    }
}

fn render_append_only(
    messages: &[ChatMessage],
    add_generation_prompt: bool,
    tokenizer: &dyn Tokenizer,
    render_context: &RenderContext,
    thinking: Option<bool>,
) -> Result<String, TitoError> {
    let content_format = tokenizer.chat_template_content_format();
    let values = messages_to_template_values_with_context(messages, content_format, render_context)
        .map_err(|e| TitoError::EngineFailed(format!("serialize: {e}")))?;
    tokenizer
        .apply_chat_template(
            &values,
            template_params(add_generation_prompt, thinking, render_context),
        )
        .map_err(|e| TitoError::EngineFailed(e.to_string()))
}

fn encode_ids(tokenizer: &dyn Tokenizer, text: &str) -> Result<Vec<u32>, TitoError> {
    tokenizer
        .encode(text, false)
        .map_err(|e| TitoError::EngineFailed(e.to_string()))
        .map(|enc| enc.token_ids().to_vec())
}

/// Tokenize one appended segment against a precomputed dummy base and return
/// only the incremental token IDs.
///
/// Validates the token-prefix invariant: the fully rendered token sequence
/// must start with the base token sequence. If violated, the template is not
/// safely incremental for this segment.
fn tokenize_segment_incremental(
    segment: &[ChatMessage],
    add_generation_prompt: bool,
    tokenizer: &dyn Tokenizer,
    render_context: &RenderContext,
    thinking: Option<bool>,
    base: &SegmentBase,
) -> Result<Vec<u32>, TitoError> {
    let mut full_msgs = base.messages.clone();
    full_msgs.extend_from_slice(segment);
    let full_text = render_append_only(
        &full_msgs,
        add_generation_prompt,
        tokenizer,
        render_context,
        thinking,
    )?;

    let full_ids = encode_ids(tokenizer, &full_text)?;
    if !full_ids.starts_with(&base.ids) {
        tracing::warn!(
            base_tokens = base.ids.len(),
            full_tokens = full_ids.len(),
            "merge_incremental: segment token-prefix invariant failed",
        );
        return Err(TitoError::TokenPrefixMismatch { path: "segment" });
    }
    Ok(full_ids[base.ids.len()..].to_vec())
}

#[cfg(test)]
mod tests {
    use llm_tokenizer::chat_template::ChatTemplateProcessor;
    use openai_protocol::chat::MessageContent;

    use super::*;
    use crate::model_adapter::DefaultAdapter;

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

    #[test]
    fn split_into_segments_classifies_assistant_messages() {
        let appended = vec![
            user_msg("u1"),
            assistant_msg("a1"),
            tool_msg("r1", "c1"),
            tool_msg("r2", "c2"),
            assistant_msg("a2"),
        ];
        let segments = split_into_segments(&appended);
        assert_eq!(segments.len(), 4);
        assert!(matches!(segments[0].0, SegmentKind::UserLike));
        assert!(matches!(segments[1].0, SegmentKind::Assistant));
        assert!(matches!(segments[2].0, SegmentKind::Tool));
        assert_eq!(segments[2].1.len(), 2);
        assert!(matches!(segments[3].0, SegmentKind::Assistant));
    }

    #[test]
    fn split_into_segments_single_assistant() {
        let appended = vec![assistant_msg("a1")];
        let segments = split_into_segments(&appended);
        assert_eq!(segments.len(), 1);
        assert!(matches!(segments[0].0, SegmentKind::Assistant));
    }

    /// The production Qwen3 template used by PSRL
    /// (`examples/mini_swe/config/qwen_no_think_strip.jinja`) renders every
    /// system/user/assistant message independently:
    ///
    ///   `<|im_start|>{role}\n{content}<|im_end|>\n`
    ///
    /// so an assistant turn inside the appended slice (compacted context) can be
    /// tokenized incrementally with the same dummy base as a user message.  This
    /// test proves the byte-level property end-to-end: reconstructing
    /// `prefix + appended` from per-segment dummy-base renders (with the
    /// generation prompt on the final segment) equals the one-shot full render.
    #[test]
    fn qwen_template_incremental_with_assistant_messages_is_byte_exact() {
        let template = r#"
{%- for message in messages %}
    {%- set content = message.content if message.content is string else '' %}
    {%- if message.role == 'system' or message.role == 'user' or message.role == 'assistant' %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' }}
{%- endif %}
"#;
        let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();

        let render = |msgs: &[ChatMessage], add_generation_prompt: bool| -> String {
            let values: Vec<serde_json::Value> = msgs
                .iter()
                .map(|m| serde_json::to_value(m).unwrap())
                .collect();
            processor
                .apply_chat_template(
                    &values,
                    ChatTemplateParams {
                        add_generation_prompt,
                        ..Default::default()
                    },
                )
                .unwrap()
        };

        let prefix = vec![
            ChatMessage::System {
                content: MessageContent::Text("You are a Claude agent".to_string()),
                name: None,
            },
            user_msg("hi"),
            assistant_msg("hello"),
        ];
        let appended = vec![
            user_msg("how are you?"),
            assistant_msg("I am fine"),
            tool_msg("tool_result_here", "call_1"),
            user_msg("great"),
        ];

        let segments = split_into_segments(&appended);
        assert_eq!(segments.len(), 4);
        assert!(matches!(segments[0].0, SegmentKind::UserLike));
        assert!(matches!(segments[1].0, SegmentKind::Assistant));
        assert!(matches!(segments[2].0, SegmentKind::Tool));
        assert!(matches!(segments[3].0, SegmentKind::UserLike));

        let dummy_system = ChatMessage::System {
            content: MessageContent::Text("_".to_string()),
            name: None,
        };
        let mut reconstructed = render(&prefix, false);
        for (idx, (kind, segment)) in segments.iter().enumerate() {
            let add_gen = idx == segments.len() - 1;
            let base: Vec<ChatMessage> = match kind {
                SegmentKind::Tool => vec![
                    dummy_system.clone(),
                    DefaultAdapter.build_dummy_assistant(segment),
                ],
                SegmentKind::UserLike | SegmentKind::Assistant => vec![dummy_system.clone()],
            };
            let base_text = render(&base, false);
            let mut full_msgs = base;
            full_msgs.extend_from_slice(segment);
            let full_text = render(&full_msgs, add_gen);
            assert!(
                full_text.starts_with(&base_text),
                "append-only invariant violated for segment {idx}"
            );
            reconstructed.push_str(&full_text[base_text.len()..]);
        }

        let mut all = prefix;
        all.extend_from_slice(&appended);
        let expected = render(&all, true);
        assert_eq!(reconstructed, expected);
    }
}
