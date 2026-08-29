//! Segment-based TITO token sequence validator.
//!
//! Mirrors the algorithm in miles' `TokenSeqComparator`:
//!
//! 1. Collect special token IDs from the tokenizer (segment boundaries).
//! 2. Trim model stop tokens from both sequence tails.
//! 3. Split each sequence into alternating special/content segments.
//! 4. Structural pre-check: if segment counts or special/content patterns differ,
//!    return a single `special_token_count` mismatch — no per-segment comparison possible.
//! 5. For each aligned segment pair:
//!    - Special segment: compare token IDs → `special_token_type` on mismatch.
//!    - Content segment: decode both sides → compare as text; classify mismatch as
//!      `assistant_text` (expected, non-severe) or `non_assistant_text` (TITO bug).
//!
//! All four mismatch types are reported; the list is empty on a perfect match.

use std::{collections::HashSet, sync::Arc};

use llm_tokenizer::{
    chat_template::{ChatTemplateContentFormat, ChatTemplateParams},
    traits::Tokenizer,
};
use openai_protocol::chat::ChatMessage;
use serde_json::{json, Value};

use crate::{normalizer::RenderContext, store::MismatchEntry};

/// A contiguous run of token IDs — either a single special token or a content run.
struct Segment {
    token_ids: Vec<u32>,
    is_special: bool,
}

/// Validates TITO token sequences against canonical retokenization.
///
/// Constructed once per request with model-specific parameters; `validate` may
/// be called once.
pub struct TokenSeqValidator {
    tokenizer: Arc<dyn Tokenizer>,
    /// IDs treated as segment boundaries (structural special tokens).
    special_ids: HashSet<u32>,
    /// Decoded text prefix identifying assistant content (e.g. `"<|im_start|>assistant"`).
    assistant_start_str: Option<String>,
    /// Token IDs to strip from both sequence tails before comparison.
    trim_trailing_ids: HashSet<u32>,
}

impl TokenSeqValidator {
    /// Create a validator with model-specific parameters.
    pub fn new(
        tokenizer: Arc<dyn Tokenizer>,
        assistant_start_str: Option<String>,
        trim_trailing_ids: HashSet<u32>,
    ) -> Self {
        let special_ids = collect_special_ids(&*tokenizer);
        Self {
            tokenizer,
            special_ids,
            assistant_start_str,
            trim_trailing_ids,
        }
    }

    /// Compare `tito_token_ids` against a fresh full retokenization of `messages`.
    ///
    /// `add_generation_prompt` and `thinking` must mirror the values the
    /// production render path used for the same messages, otherwise the
    /// reference sequence diverges from the rollout render and the validator
    /// reports false mismatches.
    ///
    /// Returns a list of mismatches (empty = perfect match).
    /// Always call this only in debug mode (gate at the call site).
    pub fn validate(
        &self,
        tito_token_ids: &[u32],
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        thinking: Option<bool>,
        render_context: &RenderContext,
    ) -> Vec<MismatchEntry> {
        // Build canonical token IDs via chat-template rendering + encoding.
        let values = match messages_to_template_values_with_context(
            messages,
            self.tokenizer.chat_template_content_format(),
            render_context,
        ) {
            Ok(v) => v,
            Err(_) => return vec![],
        };

        let params = ChatTemplateParams {
            add_generation_prompt,
            thinking,
            tools: render_context.tools_ref(),
            template_kwargs: render_context.template_kwargs_ref(),
            ..Default::default()
        };

        let text = match self.tokenizer.apply_chat_template(&values, params) {
            Ok(t) => t,
            Err(_) => return vec![],
        };

        let reference_ids: Vec<u32> = match self.tokenizer.encode(&text, false) {
            Ok(enc) => enc.token_ids().to_vec(),
            Err(_) => return vec![],
        };

        self.compare_sequences(tito_token_ids, &reference_ids)
    }

    fn compare_sequences(&self, actual_ids: &[u32], expected_ids: &[u32]) -> Vec<MismatchEntry> {
        // Trim trailing stop tokens from both sides.
        let actual_trimmed = trim_trailing(actual_ids, &self.trim_trailing_ids);
        let expected_trimmed = trim_trailing(expected_ids, &self.trim_trailing_ids);

        let act_segs = self.segment_by_special_tokens(actual_trimmed);
        let exp_segs = self.segment_by_special_tokens(expected_trimmed);

        // Structural pre-check.
        if let Some(structural) = self.check_segment_structure(&exp_segs, &act_segs) {
            return vec![structural];
        }

        // Per-segment comparison.
        let mut mismatches = Vec::new();
        for (idx, (exp, act)) in exp_segs.iter().zip(act_segs.iter()).enumerate() {
            // Classify content segments as assistant or not.
            let is_assistant = !exp.is_special
                && self.is_assistant_content(&exp_segs, idx)
                && self.is_assistant_content(&act_segs, idx);

            if let Some(m) = self.compare_single_segment(idx, exp, act, is_assistant) {
                mismatches.push(m);
            }
        }
        mismatches
    }

    /// Split `token_ids` into alternating special / content segments.
    ///
    /// Each special token becomes its own single-ID segment (`is_special=true`).
    /// Consecutive non-special tokens are grouped into content segments.
    fn segment_by_special_tokens(&self, token_ids: &[u32]) -> Vec<Segment> {
        if token_ids.is_empty() {
            return vec![];
        }

        let mut segments: Vec<Segment> = Vec::new();
        let mut current: Vec<u32> = Vec::new();

        for &tid in token_ids {
            if self.special_ids.contains(&tid) {
                if !current.is_empty() {
                    segments.push(Segment {
                        token_ids: std::mem::take(&mut current),
                        is_special: false,
                    });
                }
                segments.push(Segment {
                    token_ids: vec![tid],
                    is_special: true,
                });
            } else {
                current.push(tid);
            }
        }
        if !current.is_empty() {
            segments.push(Segment {
                token_ids: current,
                is_special: false,
            });
        }
        segments
    }

    fn check_segment_structure(
        &self,
        exp_segs: &[Segment],
        act_segs: &[Segment],
    ) -> Option<MismatchEntry> {
        let detail = if exp_segs.len() != act_segs.len() {
            format!(
                "segment count differs: expected {}, got {}",
                exp_segs.len(),
                act_segs.len()
            )
        } else if exp_segs
            .iter()
            .zip(act_segs.iter())
            .any(|(e, a)| e.is_special != a.is_special)
        {
            "segment structure (special/content pattern) differs".to_string()
        } else {
            return None;
        };

        Some(MismatchEntry {
            mismatch_type: "special_token_count".to_string(),
            // usize::MAX is the sentinel for "no specific segment index" (mirrors Python's -1).
            position: usize::MAX,
            detail: format!(
                "{} | expected: {} | actual: {}",
                detail,
                describe_structure(exp_segs, &*self.tokenizer),
                describe_structure(act_segs, &*self.tokenizer),
            ),
        })
    }

    fn compare_single_segment(
        &self,
        idx: usize,
        exp: &Segment,
        act: &Segment,
        is_assistant_content: bool,
    ) -> Option<MismatchEntry> {
        if exp.is_special {
            // Special segments: compare by token ID.
            if exp.token_ids != act.token_ids {
                let exp_text = self.decode_ids(&exp.token_ids);
                let act_text = self.decode_ids(&act.token_ids);
                return Some(MismatchEntry {
                    mismatch_type: "special_token_type".to_string(),
                    position: idx,
                    detail: format!(
                        "expected {:?} ({:?}), got {:?} ({:?})",
                        exp.token_ids, exp_text, act.token_ids, act_text
                    ),
                });
            }
            return None;
        }

        // Content segments: compare decoded text.
        let exp_text = self.decode_ids(&exp.token_ids);
        let act_text = self.decode_ids(&act.token_ids);
        if exp_text == act_text {
            return None;
        }

        let mismatch_type = if is_assistant_content {
            "assistant_text"
        } else {
            "non_assistant_text"
        };

        Some(MismatchEntry {
            mismatch_type: mismatch_type.to_string(),
            position: idx,
            detail: format!("expected {:?}, got {:?}", exp_text, act_text),
        })
    }

    /// Return `true` if the content segment at `idx` belongs to an assistant message.
    ///
    /// Checks that:
    /// - `idx` is a content (non-special) segment.
    /// - The preceding segment (`idx-1`) is a special token.
    /// - The decoded concatenation of the preceding special token and the first ≤20
    ///   tokens of the content segment starts with `assistant_start_str`.
    fn is_assistant_content(&self, segments: &[Segment], idx: usize) -> bool {
        let start_str = match self.assistant_start_str.as_deref() {
            Some(s) => s,
            None => return false,
        };
        if idx == 0 {
            return false;
        }
        let seg = &segments[idx];
        if seg.is_special {
            return false;
        }
        let prev = &segments[idx - 1];
        if !prev.is_special {
            return false;
        }

        let special_text = self.decode_ids(&prev.token_ids);
        let prefix_len = seg.token_ids.len().min(20);
        let content_prefix = self.decode_ids(&seg.token_ids[..prefix_len]);
        (special_text + &content_prefix).starts_with(start_str)
    }

    fn decode_ids(&self, ids: &[u32]) -> String {
        self.tokenizer
            .decode(ids, false)
            .unwrap_or_else(|_| format!("<decode-error:{:?}>", ids))
    }
}

/// Collect token IDs that act as structural segment boundaries.
fn collect_special_ids(tokenizer: &dyn Tokenizer) -> HashSet<u32> {
    let special = tokenizer.get_special_tokens();
    let mut ids: HashSet<u32> = HashSet::new();

    // Named special tokens.
    for opt_token in [
        special.bos_token.as_deref(),
        special.eos_token.as_deref(),
        special.unk_token.as_deref(),
        special.sep_token.as_deref(),
        special.pad_token.as_deref(),
        special.cls_token.as_deref(),
        special.mask_token.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        if let Some(id) = tokenizer.token_to_id(opt_token) {
            ids.insert(id);
        }
    }

    // Additional special tokens (e.g. `<|im_start|>`, `<|im_end|>`, `<|assistant|>`).
    for token in &special.additional_special_tokens {
        if let Some(id) = tokenizer.token_to_id(token) {
            ids.insert(id);
        }
    }

    ids
}

/// Strip trailing token IDs that belong to `to_remove`.
fn trim_trailing<'a>(ids: &'a [u32], to_remove: &HashSet<u32>) -> &'a [u32] {
    if to_remove.is_empty() {
        return ids;
    }
    let mut end = ids.len();
    while end > 0 && to_remove.contains(&ids[end - 1]) {
        end -= 1;
    }
    &ids[..end]
}

/// Build a human-readable description of a segment list (for mismatch details).
fn describe_structure(segments: &[Segment], tokenizer: &dyn Tokenizer) -> String {
    segments
        .iter()
        .map(|s| {
            if s.is_special {
                let text = tokenizer
                    .decode(&s.token_ids, false)
                    .unwrap_or_else(|_| format!("{:?}", s.token_ids));
                format!("[{}]", text)
            } else {
                format!("({} tokens)", s.token_ids.len())
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
pub(crate) fn messages_to_template_values(
    messages: &[ChatMessage],
    content_format: ChatTemplateContentFormat,
    image_placeholder: Option<&str>,
) -> Result<Vec<Value>, serde_json::Error> {
    let context = RenderContext::new(None, None).with_media_placeholders(
        image_placeholder.map(String::from),
        None,
        None,
    );
    messages_to_template_values_with_context(messages, content_format, &context)
}

pub(crate) fn messages_to_template_values_with_context(
    messages: &[ChatMessage],
    content_format: ChatTemplateContentFormat,
    render_context: &RenderContext,
) -> Result<Vec<Value>, serde_json::Error> {
    let mut values = messages
        .iter()
        .map(|message| {
            let mut value = serde_json::to_value(message)?;
            if let Some(obj) = value.as_object_mut() {
                // Mirror the production render path: assistant messages without
                // a content key get an empty-string content so chat templates
                // can test `message['content'] is none` / `is string`
                // consistently between the rollout render and the validator.
                if obj.get("role").and_then(|v| v.as_str()) == Some("assistant")
                    && !obj.contains_key("content")
                {
                    obj.insert("content".to_string(), Value::String(String::new()));
                }
                if let Some(content_value) = obj.get_mut("content") {
                    transform_content_field(content_value, content_format, render_context);
                }
            }
            Ok(value)
        })
        .collect::<Result<Vec<_>, _>>()?;
    process_tool_call_arguments(&mut values);
    Ok(values)
}

fn transform_content_field(
    content_value: &mut Value,
    content_format: ChatTemplateContentFormat,
    render_context: &RenderContext,
) {
    let Some(content_array) = content_value.as_array() else {
        return;
    };

    match content_format {
        ChatTemplateContentFormat::String => {
            // Mirror the production render path exactly (chat_utils.rs
            // `transform_content_field`): media placeholders are emitted BEFORE
            // text and all parts are joined with "\n".  A divergent join/order
            // here would make the validator re-render differ from the rollout
            // render for the same messages, producing false mismatches.
            let mut media_parts: Vec<String> = Vec::new();
            let mut text_parts: Vec<String> = Vec::new();
            for part in content_array {
                let Some(obj) = part.as_object() else {
                    continue;
                };
                let type_str = obj.get("type").and_then(|t| t.as_str());
                match type_str {
                    Some("text") => {
                        if let Some(t) = obj.get("text").and_then(|t| t.as_str()) {
                            text_parts.push(t.to_string());
                        }
                    }
                    Some(
                        "image_url" | "image" | "input_image" | "video_url" | "video" | "audio_url"
                        | "audio" | "input_audio",
                    ) => {
                        if let Some(placeholder) =
                            type_str.and_then(|t| render_context.placeholder_for_part_type(t))
                        {
                            media_parts.push(placeholder.to_string());
                        }
                    }
                    _ => {}
                }
            }

            if !media_parts.is_empty() || !text_parts.is_empty() {
                let ordered: Vec<String> = media_parts.into_iter().chain(text_parts).collect();
                *content_value = Value::String(ordered.join("\n"));
            }
        }
        ChatTemplateContentFormat::OpenAI => {
            let processed_parts: Vec<Value> = content_array
                .iter()
                .map(|part| {
                    part.as_object()
                        .and_then(|obj| obj.get("type")?.as_str())
                        .and_then(|type_str| match type_str {
                            "image_url" => Some(json!({"type": "image"})),
                            "video_url" => Some(json!({"type": "video"})),
                            "audio_url" => Some(json!({"type": "audio"})),
                            _ => None,
                        })
                        .unwrap_or_else(|| part.clone())
                })
                .collect();

            *content_value = Value::Array(processed_parts);
        }
    }
}

fn process_tool_call_arguments(messages: &mut [Value]) {
    for msg in messages {
        let role = msg.get("role").and_then(|v| v.as_str());
        if role != Some("assistant") {
            continue;
        }

        let Some(tool_calls) = msg.get_mut("tool_calls").and_then(|tc| tc.as_array_mut()) else {
            continue;
        };

        for call in tool_calls {
            let Some(function) = call.get_mut("function") else {
                continue;
            };
            let Some(args) = function.get_mut("arguments") else {
                continue;
            };
            let Some(args_str) = args.as_str() else {
                continue;
            };
            if let Ok(parsed) = serde_json::from_str::<Value>(args_str) {
                // Canonicalize (sort keys) so the rendered tool-call block is
                // order-independent, matching the order-insensitive prefix hash
                // (`sort_json_keys` in normalizer.rs).  The model may emit
                // arguments with one key order while Claude Code re-serializes
                // the parsed `input` object with a different order; both must
                // render — and hash — identically for the prefix to HIT.
                *args = crate::normalizer::sort_json_keys(parsed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::chat::MessageContent;

    use super::*;

    #[test]
    fn messages_to_template_values_handles_system_message() {
        use openai_protocol::chat::ChatMessage;

        let messages: Vec<ChatMessage> = vec![ChatMessage::System {
            content: MessageContent::Text("You are helpful".to_string()),
            name: None,
        }];

        let values = messages_to_template_values(
            &messages,
            llm_tokenizer::chat_template::ChatTemplateContentFormat::String,
            None,
        )
        .unwrap();

        assert_eq!(values.len(), 1);
    }

    #[test]
    fn messages_to_template_values_handles_multipart_content() {
        use openai_protocol::{chat::ChatMessage, common::ContentPart};

        let messages: Vec<ChatMessage> = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![ContentPart::Text {
                text: "Hello".to_string(),
            }]),
            name: None,
        }];

        let values = messages_to_template_values(
            &messages,
            llm_tokenizer::chat_template::ChatTemplateContentFormat::String,
            None,
        )
        .unwrap();

        assert_eq!(values.len(), 1);
    }

    #[test]
    fn messages_to_template_values_handles_tool_calls() {
        use openai_protocol::{
            chat::ChatMessage,
            common::{FunctionCallResponse, ToolCall},
        };

        let messages: Vec<ChatMessage> = vec![ChatMessage::Assistant {
            content: None,
            name: None,
            reasoning_content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_1".to_string(),
                tool_type: "function".to_string(),
                function: FunctionCallResponse {
                    name: "my_function".to_string(),
                    arguments: Some(r#"{"key": "value"}"#.to_string()),
                },
            }]),
        }];

        let values = messages_to_template_values(
            &messages,
            llm_tokenizer::chat_template::ChatTemplateContentFormat::String,
            None,
        )
        .unwrap();

        assert_eq!(values.len(), 1);
        // Verify tool_calls are preserved
        let tool_calls = values[0].get("tool_calls");
        assert!(tool_calls.is_some());
    }

    #[test]
    fn messages_to_template_values_handles_developer_message() {
        use openai_protocol::chat::ChatMessage;

        let messages: Vec<ChatMessage> = vec![ChatMessage::Developer {
            content: MessageContent::Text("System directive".to_string()),
            name: None,
            tools: None,
        }];

        let values = messages_to_template_values(
            &messages,
            llm_tokenizer::chat_template::ChatTemplateContentFormat::String,
            None,
        )
        .unwrap();

        assert_eq!(values.len(), 1);
        assert_eq!(
            values[0].get("role").and_then(|r| r.as_str()),
            Some("developer")
        );
    }

    #[test]
    fn messages_to_template_values_handles_tool_message() {
        use openai_protocol::chat::ChatMessage;

        let messages: Vec<ChatMessage> = vec![ChatMessage::Tool {
            content: MessageContent::Text("Tool result".to_string()),
            tool_call_id: "call_1".to_string(),
        }];

        let values = messages_to_template_values(
            &messages,
            llm_tokenizer::chat_template::ChatTemplateContentFormat::String,
            None,
        )
        .unwrap();

        assert_eq!(values.len(), 1);
        assert_eq!(values[0].get("role").and_then(|r| r.as_str()), Some("tool"));
    }

    // -----------------------------------------------------------------------
    // Tests for the new segment-based algorithm helpers
    // -----------------------------------------------------------------------

    #[test]
    fn transform_content_injects_image_placeholder_string_format() {
        use openai_protocol::{chat::ChatMessage, common::ContentPart};

        let messages: Vec<ChatMessage> = vec![ChatMessage::User {
            content: openai_protocol::chat::MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "describe this".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: openai_protocol::common::ImageUrl {
                        url: "http://example.com/img.png".to_string(),
                        detail: None,
                    },
                },
            ]),
            name: None,
        }];

        // Without placeholder: image parts are silently dropped.
        let values_no_placeholder = messages_to_template_values(
            &messages,
            llm_tokenizer::chat_template::ChatTemplateContentFormat::String,
            None,
        )
        .unwrap();
        let content_no_ph = values_no_placeholder[0]
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert_eq!(content_no_ph, "describe this");

        // With placeholder: the image part is hoisted to the front and parts are
        // joined with "\n", mirroring the production render path.
        let values_with_placeholder = messages_to_template_values(
            &messages,
            llm_tokenizer::chat_template::ChatTemplateContentFormat::String,
            Some("<|image|>"),
        )
        .unwrap();
        let content_with_ph = values_with_placeholder[0]
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert_eq!(content_with_ph, "<|image|>\ndescribe this");
    }

    #[test]
    fn string_format_injects_all_modality_placeholders() {
        use openai_protocol::{
            chat::{ChatMessage, MessageContent},
            common::{AudioUrl, ContentPart, ImageUrl, VideoUrl},
        };
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "image".into(),
                        detail: None,
                    },
                },
                ContentPart::VideoUrl {
                    video_url: VideoUrl {
                        url: "video".into(),
                    },
                },
                ContentPart::AudioUrl {
                    audio_url: AudioUrl {
                        url: "audio".into(),
                    },
                },
            ]),
            name: None,
        }];
        let context = RenderContext::new(None, None).with_media_placeholders(
            Some("<image>".into()),
            Some("<video>".into()),
            Some("<audio>".into()),
        );
        let values = messages_to_template_values_with_context(
            &messages,
            ChatTemplateContentFormat::String,
            &context,
        )
        .unwrap();
        assert_eq!(values[0]["content"], "<image>\n<video>\n<audio>");
    }

    #[test]
    fn trim_trailing_removes_matching_tokens() {
        let ids = [1u32, 2, 3, 4, 4];
        let to_remove: HashSet<u32> = [4u32].into();
        assert_eq!(trim_trailing(&ids, &to_remove), &[1, 2, 3]);
    }

    #[test]
    fn trim_trailing_removes_nothing_when_no_match() {
        let ids = [1u32, 2, 3];
        let to_remove: HashSet<u32> = [99u32].into();
        assert_eq!(trim_trailing(&ids, &to_remove), &[1, 2, 3]);
    }

    #[test]
    fn trim_trailing_empty_set_is_identity() {
        let ids = [1u32, 2, 3];
        assert_eq!(trim_trailing(&ids, &HashSet::new()), &[1, 2, 3]);
    }

    #[test]
    fn trim_trailing_can_remove_all_tokens() {
        let ids = [5u32, 5, 5];
        let to_remove: HashSet<u32> = [5u32].into();
        assert_eq!(trim_trailing(&ids, &to_remove), &[] as &[u32]);
    }

    #[test]
    fn process_tool_call_arguments_canonicalizes_key_order() {
        use openai_protocol::{
            chat::ChatMessage,
            common::{FunctionCallResponse, ToolCall},
        };

        // Same logical arguments, different JSON key order (as the model emits
        // vs. what Claude Code re-serializes).  `process_tool_call_arguments`
        // must canonicalize so both render to the same tool-call block, matching
        // the order-insensitive prefix hash.
        let mk_asst = |args: &str| ChatMessage::Assistant {
            content: None,
            name: None,
            reasoning_content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_1".to_string(),
                tool_type: "function".to_string(),
                function: FunctionCallResponse {
                    name: "Edit".to_string(),
                    arguments: Some(args.to_string()),
                },
            }]),
        };

        let render_args = |args: &str| -> Value {
            let messages = vec![mk_asst(args)];
            let values = messages_to_template_values(
                &messages,
                llm_tokenizer::chat_template::ChatTemplateContentFormat::String,
                None,
            )
            .unwrap();
            values[0]["tool_calls"][0]["function"]["arguments"].clone()
        };

        let a = render_args(r#"{"file_path":"a.py","old_string":{"z":1,"a":2},"new_string":"b"}"#);
        let b = render_args(r#"{"new_string":"b","file_path":"a.py","old_string":{"a":2,"z":1}}"#);
        assert_eq!(a, b, "tool-call arguments must render canonically");
        assert_eq!(
            a,
            serde_json::json!({"file_path":"a.py","old_string":{"a":2,"z":1},"new_string":"b"})
        );
    }
}
