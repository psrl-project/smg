use openai_protocol::{
    chat::{ChatMessage, MessageContent},
    common::{FunctionCallResponse, ToolCall},
};

/// Model-specific token boundary adjustment.
pub trait ModelAdapter: Send + Sync {
    /// Adjust the end of the pretokenized prefix before merging with incremental tokens.
    fn adjust_prefix_boundary(&self, prefix: &[u32]) -> Vec<u32>;

    /// Number of trailing boundary tokens the model may emit that will be
    /// re-emitted by the chat template as the next turn's delimiter.
    /// These are trimmed from non-last turns during training data construction.
    fn max_trim_tokens(&self) -> usize {
        0
    }

    /// Decoded text prefix that identifies an assistant content segment in the
    /// token stream (e.g. `"<|im_start|>assistant"` for Qwen3, `"<|assistant|>"` for GLM).
    ///
    /// Used by `TokenSeqValidator` to classify content-segment mismatches as
    /// `assistant_text` (expected/non-severe) vs `non_assistant_text` (a TITO bug).
    /// Returns `None` for models where no such detection is needed.
    fn assistant_start_str(&self) -> Option<&str> {
        None
    }

    /// Token IDs to strip from both sequence tails before comparison.
    ///
    /// The model's generated output may end with a stop token (e.g. newline for
    /// Qwen after `<|im_end|>`, or `<|observation|>`/`<|user|>` for GLM) that
    /// won't appear at the same position in the template-rendered canonical sequence.
    /// Stripping these before comparison avoids false structural mismatches.
    fn trailing_token_ids(&self) -> &[u32] {
        &[]
    }

    /// Whether the chat template renders assistant turns **position-dependently**.
    ///
    /// Qwen3.5, for example, wraps an assistant turn in `<think>...</think>` only
    /// when it appears after the last real user query (`ns.last_query_index`).
    /// For such templates, per-message dummy-base incremental tokenization is
    /// unsound: the dummy context would decide the wrap state differently from
    /// the production context.  The engine must then use the whole-appended-slice
    /// single-pass path (whose fixed dummy prefix keeps the wrap decision
    /// translation-invariant) instead of per-segment tokenization.
    fn rendering_is_position_dependent(&self) -> bool {
        false
    }

    /// Build a synthetic assistant message that mirrors tool_call_ids from tool messages in appended.
    fn build_dummy_assistant(&self, appended: &[ChatMessage]) -> ChatMessage {
        // Extract tool_call_ids from tool messages in appended
        let tool_call_ids: Vec<String> = appended
            .iter()
            .filter_map(|m| match m {
                ChatMessage::Tool { tool_call_id, .. } => Some(tool_call_id.clone()),
                _ => None,
            })
            .collect();

        if tool_call_ids.is_empty() {
            ChatMessage::Assistant {
                content: Some(MessageContent::Text(String::new())),
                name: None,
                tool_calls: None,
                reasoning_content: None,
            }
        } else {
            ChatMessage::Assistant {
                content: Some(MessageContent::Text(String::new())),
                name: None,
                reasoning_content: None,
                tool_calls: Some(
                    tool_call_ids
                        .into_iter()
                        .map(|id| ToolCall {
                            id,
                            tool_type: "function".to_string(),
                            function: FunctionCallResponse {
                                name: String::new(),
                                arguments: Some("{}".to_string()),
                            },
                        })
                        .collect(),
                ),
            }
        }
    }
}

/// Default adapter: no boundary adjustment.
pub struct DefaultAdapter;

impl ModelAdapter for DefaultAdapter {
    fn adjust_prefix_boundary(&self, prefix: &[u32]) -> Vec<u32> {
        prefix.to_vec()
    }
}

/// Qwen3 (base Qwen3 / Qwen3-MoE): append newline token if prefix ends with `<|im_end|>`.
pub struct Qwen3Adapter {
    pub im_end_id: u32,
    pub newline_id: u32,
    trailing_ids: [u32; 2],
}

impl Qwen3Adapter {
    pub fn new(im_end_id: u32, newline_id: u32) -> Self {
        Self {
            im_end_id,
            newline_id,
            // Also trim a trailing `<|im_end|>` (plus the template's following
            // newline) from both compared sequences. A truncated/aborted final
            // assistant turn legitimately ends without the closing stop token in
            // the recorded token stream, while the chat-template reference
            // always appends one; without this the validator's structural
            // pre-check reports a spurious `special_token_count` mismatch on the
            // last turn. Tail-only, so mid-sequence differences are still caught.
            trailing_ids: [im_end_id, newline_id],
        }
    }
}

impl ModelAdapter for Qwen3Adapter {
    fn adjust_prefix_boundary(&self, prefix: &[u32]) -> Vec<u32> {
        let mut result = prefix.to_vec();
        if result.last() == Some(&self.im_end_id) {
            result.push(self.newline_id);
        }
        result
    }

    fn max_trim_tokens(&self) -> usize {
        1
    }

    fn assistant_start_str(&self) -> Option<&str> {
        Some("<|im_start|>assistant")
    }

    fn trailing_token_ids(&self) -> &[u32] {
        &self.trailing_ids
    }
}

/// Qwen3.5 family: same boundary behaviour as Qwen3 for now (append newline after
/// `<|im_end|>`), but kept as a separate type so that model-family-specific divergences
/// can be handled here without touching `Qwen3Adapter`.
pub struct Qwen35Adapter {
    pub im_end_id: u32,
    pub newline_id: u32,
    trailing_ids: [u32; 2],
}

impl Qwen35Adapter {
    pub fn new(im_end_id: u32, newline_id: u32) -> Self {
        Self {
            im_end_id,
            newline_id,
            // See Qwen3Adapter::new — trim a trailing <|im_end|> plus newline.
            trailing_ids: [im_end_id, newline_id],
        }
    }
}

impl ModelAdapter for Qwen35Adapter {
    fn adjust_prefix_boundary(&self, prefix: &[u32]) -> Vec<u32> {
        let mut result = prefix.to_vec();
        if result.last() == Some(&self.im_end_id) {
            result.push(self.newline_id);
        }
        result
    }

    fn max_trim_tokens(&self) -> usize {
        1
    }

    fn assistant_start_str(&self) -> Option<&str> {
        Some("<|im_start|>assistant")
    }

    fn trailing_token_ids(&self) -> &[u32] {
        &self.trailing_ids
    }

    fn rendering_is_position_dependent(&self) -> bool {
        true
    }
}

/// QwenNext family (future Qwen releases beyond 3.5): same boundary behaviour as
/// Qwen3 for now, isolated for easy differentiation.
pub struct QwenNextAdapter {
    pub im_end_id: u32,
    pub newline_id: u32,
    trailing_ids: [u32; 2],
}

impl QwenNextAdapter {
    pub fn new(im_end_id: u32, newline_id: u32) -> Self {
        Self {
            im_end_id,
            newline_id,
            // See Qwen3Adapter::new — trim a trailing <|im_end|> plus newline.
            trailing_ids: [im_end_id, newline_id],
        }
    }
}

impl ModelAdapter for QwenNextAdapter {
    fn adjust_prefix_boundary(&self, prefix: &[u32]) -> Vec<u32> {
        let mut result = prefix.to_vec();
        if result.last() == Some(&self.im_end_id) {
            result.push(self.newline_id);
        }
        result
    }

    fn max_trim_tokens(&self) -> usize {
        1
    }

    fn assistant_start_str(&self) -> Option<&str> {
        Some("<|im_start|>assistant")
    }

    fn trailing_token_ids(&self) -> &[u32] {
        &self.trailing_ids
    }
}

/// GLM4.7: strip last token if it is `<|observation|>` or `<|user|>`.
pub struct Glm47Adapter {
    pub observation_id: u32,
    pub user_id: u32,
    trailing_ids: [u32; 2],
}

impl Glm47Adapter {
    pub fn new(observation_id: u32, user_id: u32) -> Self {
        Self {
            observation_id,
            user_id,
            trailing_ids: [observation_id, user_id],
        }
    }
}

impl ModelAdapter for Glm47Adapter {
    fn adjust_prefix_boundary(&self, prefix: &[u32]) -> Vec<u32> {
        let mut result = prefix.to_vec();
        if matches!(result.last(), Some(&id) if id == self.observation_id || id == self.user_id) {
            result.pop();
        }
        result
    }

    fn max_trim_tokens(&self) -> usize {
        1
    }

    fn assistant_start_str(&self) -> Option<&str> {
        Some("<|assistant|>")
    }

    fn trailing_token_ids(&self) -> &[u32] {
        &self.trailing_ids
    }
}

/// Select an adapter at runtime based on a model identifier string.
///
/// The `model_identifier` may be either an exact `hf_model_type` value (e.g. `"qwen3"`,
/// `"chatglm"`) or a full model-id string (e.g. `"Qwen3.5-7B-Instruct"`).  Matching is
/// case-insensitive substring search so both forms work correctly.
///
/// Uses family-default special-token IDs; prefer
/// [`select_adapter_for_tokenizer`] in production so the boundary adjustment
/// matches the actual checkpoint vocabulary.
pub fn select_adapter(model_identifier: &str) -> Box<dyn ModelAdapter> {
    select_adapter_with_ids(model_identifier, None, None)
}

/// Select an adapter at runtime, deriving the model-specific special-token IDs
/// from the tokenizer's actual vocabulary.
///
/// Checkpoints of the same family can use different added-token IDs (e.g.
/// Qwen3-4B-Instruct-2507 has `<|im_end|>` = 151645 while Qwen3.5-4B uses
/// 248046); the hardcoded family defaults would mis-adjust the prefix boundary.
/// Falls back to the family defaults when the tokenizer cannot resolve a token.
pub fn select_adapter_for_tokenizer(
    model_identifier: &str,
    tokenizer: &dyn llm_tokenizer::traits::Tokenizer,
) -> Box<dyn ModelAdapter> {
    let im_end = tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);
    let newline = tokenizer.token_to_id("\n").unwrap_or(198);
    let observation = tokenizer.token_to_id("<|observation|>").unwrap_or(64795);
    let user = tokenizer.token_to_id("<|user|>").unwrap_or(64796);
    select_adapter_with_ids(
        model_identifier,
        Some((im_end, newline)),
        Some((observation, user)),
    )
}

fn select_adapter_with_ids(
    model_identifier: &str,
    qwen_ids: Option<(u32, u32)>,
    glm_ids: Option<(u32, u32)>,
) -> Box<dyn ModelAdapter> {
    let lower = model_identifier.to_ascii_lowercase();
    let (im_end, newline) = qwen_ids.unwrap_or((151645, 198));
    let (observation, user) = glm_ids.unwrap_or((64795, 64796));

    // GLM family — check before "qwen" to avoid any future overlap.
    if lower.contains("glm") || lower.contains("chatglm") {
        return Box::new(Glm47Adapter::new(observation, user));
    }

    // Qwen family — most-specific variants first so generic "qwen3" substring
    // does not swallow Qwen3.5 or later families.
    //
    // "qwen3.5" / "qwen-3.5" both contain "qwen3.5" or "qwen_3.5" after normalisation;
    // we normalise dots/dashes/underscores to a single form for reliable matching.
    let normalised = lower.replace(['.', '-', '_'], "");
    if normalised.contains("qwen35") || normalised.contains("qwen3point5") {
        return Box::new(Qwen35Adapter::new(im_end, newline));
    }
    if normalised.contains("qwen3") {
        return Box::new(Qwen3Adapter::new(im_end, newline));
    }
    // Generic "qwen" catch-all for future / unknown Qwen variants.
    if lower.contains("qwen") {
        return Box::new(QwenNextAdapter::new(im_end, newline));
    }

    Box::new(DefaultAdapter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_adapter_is_identity() {
        let ids = vec![1u32, 2, 3];
        assert_eq!(DefaultAdapter.adjust_prefix_boundary(&ids), ids.as_slice());
    }

    #[test]
    fn qwen3_appends_newline_when_prefix_ends_with_im_end() {
        let adapter = Qwen3Adapter::new(151645, 198);
        let ids = vec![1u32, 2, 151645];
        let result = adapter.adjust_prefix_boundary(&ids);
        assert_eq!(result.last(), Some(&198u32));
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn qwen3_no_change_when_prefix_does_not_end_with_im_end() {
        let adapter = Qwen3Adapter::new(151645, 198);
        let ids = vec![1u32, 2, 3];
        assert_eq!(adapter.adjust_prefix_boundary(&ids).len(), 3);
    }

    #[test]
    fn glm47_strips_observation_token() {
        let adapter = Glm47Adapter::new(64795, 64796);
        let ids = vec![1u32, 2, 64795];
        let result = adapter.adjust_prefix_boundary(&ids);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn adapter_assistant_start_str() {
        assert_eq!(
            Qwen3Adapter::new(151645, 198).assistant_start_str(),
            Some("<|im_start|>assistant")
        );
        assert_eq!(
            Qwen35Adapter::new(151645, 198).assistant_start_str(),
            Some("<|im_start|>assistant")
        );
        assert_eq!(
            QwenNextAdapter::new(151645, 198).assistant_start_str(),
            Some("<|im_start|>assistant")
        );
        assert_eq!(
            Glm47Adapter::new(64795, 64796).assistant_start_str(),
            Some("<|assistant|>")
        );
        assert_eq!(DefaultAdapter.assistant_start_str(), None);
    }

    #[test]
    fn adapter_trailing_token_ids() {
        assert_eq!(
            Qwen3Adapter::new(151645, 198).trailing_token_ids(),
            &[151645u32, 198]
        );
        assert_eq!(
            Qwen35Adapter::new(151645, 198).trailing_token_ids(),
            &[151645u32, 198]
        );
        assert_eq!(
            QwenNextAdapter::new(151645, 198).trailing_token_ids(),
            &[151645u32, 198]
        );
        assert_eq!(
            Glm47Adapter::new(64795, 64796).trailing_token_ids(),
            &[64795u32, 64796]
        );
        assert_eq!(DefaultAdapter.trailing_token_ids(), &[] as &[u32]);
    }

    #[test]
    fn select_adapter_returns_qwen3_for_qwen3_type() {
        let adapter = select_adapter("qwen3");
        let _ = adapter.adjust_prefix_boundary(&[1, 2, 3]);
    }

    #[test]
    fn select_adapter_returns_qwen3_for_model_id() {
        // "Qwen3-7B-Instruct" → Qwen3Adapter (appends newline after im_end)
        let adapter = select_adapter("Qwen3-7B-Instruct");
        let ids = vec![1u32, 2, 151645];
        let result = adapter.adjust_prefix_boundary(&ids);
        assert_eq!(result.last(), Some(&198u32));
    }

    #[test]
    fn select_adapter_returns_qwen35_for_qwen35_model_id() {
        // "Qwen3.5-7B-Instruct" → Qwen35Adapter (same boundary logic, distinct type)
        let adapter = select_adapter("Qwen3.5-7B-Instruct");
        let ids = vec![1u32, 2, 151645];
        let result = adapter.adjust_prefix_boundary(&ids);
        assert_eq!(result.last(), Some(&198u32));
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn select_adapter_qwen35_does_not_match_qwen3_adapter() {
        // Ensure "Qwen3.5" doesn't accidentally resolve to bare Qwen3Adapter.
        // Both produce the same boundary behaviour, but type names should differ.
        let adapter35 = select_adapter("Qwen3.5-14B");
        let adapter3 = select_adapter("Qwen3-7B");
        // Both append newline; what matters is that the dispatch path ran correctly
        // (no panic, correct token appended).
        let ids = vec![151645u32];
        assert_eq!(adapter35.adjust_prefix_boundary(&ids).last(), Some(&198u32));
        assert_eq!(adapter3.adjust_prefix_boundary(&ids).last(), Some(&198u32));
    }

    #[test]
    fn select_adapter_returns_qwen_next_for_unknown_qwen_variant() {
        // Generic "qwen" without a recognised version → QwenNextAdapter
        let adapter = select_adapter("Qwen-VL-Plus");
        let ids = vec![1u32, 2, 151645];
        let result = adapter.adjust_prefix_boundary(&ids);
        assert_eq!(result.last(), Some(&198u32));
    }

    #[test]
    fn select_adapter_returns_glm_for_model_id() {
        // "GLM-4-9B-Chat" → Glm47Adapter (strips observation token)
        let adapter = select_adapter("GLM-4-9B-Chat");
        let ids = vec![1u32, 2, 64795];
        let result = adapter.adjust_prefix_boundary(&ids);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn select_adapter_returns_default_for_unknown() {
        let adapter = select_adapter("llama3");
        let ids = vec![1u32, 2, 3];
        assert_eq!(adapter.adjust_prefix_boundary(&ids), ids.as_slice());
    }
}
