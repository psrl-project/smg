use openai_protocol::{
    chat::{ChatMessage, MessageContent},
    common::{FunctionCallResponse, ToolCall},
};

use crate::error::TitoError;

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

/// Select an adapter at runtime, deriving the model-specific special-token IDs
/// from the tokenizer's actual vocabulary and the exact server-side
/// `config.json::model_type`.
///
/// Checkpoints of the same family can use different added-token IDs (e.g.
/// Qwen3-4B-Instruct-2507 has `<|im_end|>` = 151645 while Qwen3.5-4B uses
/// 248046), so missing required tokens are typed errors rather than guessed IDs.
pub fn select_adapter_for_tokenizer(
    tokenizer: &dyn llm_tokenizer::traits::Tokenizer,
) -> Result<Box<dyn ModelAdapter>, TitoError> {
    let model_type = tokenizer
        .model_type()
        .ok_or(TitoError::ModelTypeUnavailable)?;
    select_adapter(model_type, tokenizer)
}

fn required_token_id(
    tokenizer: &dyn llm_tokenizer::traits::Tokenizer,
    model_type: &str,
    token: &'static str,
) -> Result<u32, TitoError> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| TitoError::RequiredTokenMissing {
            model_type: model_type.to_owned(),
            token,
        })
}

fn required_encoded_token_id(
    tokenizer: &dyn llm_tokenizer::traits::Tokenizer,
    model_type: &str,
    token: &'static str,
) -> Result<u32, TitoError> {
    let encoding =
        tokenizer
            .encode(token, false)
            .map_err(|_| TitoError::RequiredTokenEncoding {
                model_type: model_type.to_owned(),
                token,
            })?;
    match encoding.token_ids() {
        [id] => Ok(*id),
        _ => Err(TitoError::RequiredTokenEncoding {
            model_type: model_type.to_owned(),
            token,
        }),
    }
}

fn select_adapter(
    model_type: &str,
    tokenizer: &dyn llm_tokenizer::traits::Tokenizer,
) -> Result<Box<dyn ModelAdapter>, TitoError> {
    match model_type {
        "qwen2" | "qwen2_moe" | "qwen2_vl" | "qwen2_5_vl" | "qwen3" | "qwen3_moe" | "qwen3_vl"
        | "qwen3_vl_moe" => {
            let im_end = required_token_id(tokenizer, model_type, "<|im_end|>")?;
            let newline = required_encoded_token_id(tokenizer, model_type, "\n")?;
            Ok(Box::new(Qwen3Adapter::new(im_end, newline)))
        }
        "qwen3_5" | "qwen3_5_moe" => {
            let im_end = required_token_id(tokenizer, model_type, "<|im_end|>")?;
            let newline = required_encoded_token_id(tokenizer, model_type, "\n")?;
            Ok(Box::new(Qwen35Adapter::new(im_end, newline)))
        }
        "glm4_moe" | "glm_moe_dsa" | "glm4v_moe" => {
            let observation = required_token_id(tokenizer, model_type, "<|observation|>")?;
            let user = required_token_id(tokenizer, model_type, "<|user|>")?;
            Ok(Box::new(Glm47Adapter::new(observation, user)))
        }
        _ => Ok(Box::new(DefaultAdapter)),
    }
}

#[cfg(test)]
mod tests {
    use llm_tokenizer::mock::MockTokenizer;

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
    fn adapter_selection_requires_server_model_type() {
        let error = match select_adapter_for_tokenizer(&MockTokenizer::new()) {
            Ok(_) => panic!("missing model_type must fail"),
            Err(error) => error,
        };
        assert!(matches!(error, TitoError::ModelTypeUnavailable));
    }

    #[test]
    fn model_alias_is_not_used_as_hf_model_type() {
        let tokenizer = MockTokenizer::new();
        let adapter = select_adapter("Qwen3-7B-Instruct", &tokenizer).unwrap();
        let ids = [1, 2, 1002];
        assert_eq!(adapter.adjust_prefix_boundary(&ids), ids);
    }

    #[test]
    fn missing_required_token_is_typed_error() {
        let error = match select_adapter("qwen3", &MockTokenizer::new()) {
            Ok(_) => panic!("missing newline token must fail"),
            Err(error) => error,
        };
        assert!(matches!(error, TitoError::RequiredTokenEncoding { .. }));
    }
}
