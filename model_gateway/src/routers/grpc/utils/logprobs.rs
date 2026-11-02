//! Proto-to-OpenAI logprob conversion functions.

use std::sync::Arc;

use llm_tokenizer::traits::Tokenizer;
use openai_protocol::common::{ChatLogProbs, ChatLogProbsContent, TopLogProb};

use crate::routers::grpc::proto_wrapper::{ProtoInputLogProbs, ProtoOutputLogProbs};

/// Convert OutputLogProbs to OpenAI ChatLogProbs format
///
/// Generic over the token decoding strategy. The `decode_token` closure maps a
/// single token ID to its text representation.
pub(crate) fn convert_proto_logprobs(
    proto_logprobs: &ProtoOutputLogProbs,
    decode_token: impl Fn(u32) -> String,
) -> ChatLogProbs {
    let mut content_items = Vec::with_capacity(proto_logprobs.token_logprobs.len());

    for (i, &logprob) in proto_logprobs.token_logprobs.iter().enumerate() {
        let token_id = proto_logprobs.token_ids.get(i).copied().unwrap_or(0);
        let token_text = decode_token(token_id);
        let bytes = Some(token_text.as_bytes().to_vec());

        // Build top_logprobs for this position
        let top_logprobs = if let Some(top_logprobs_entry) = proto_logprobs.top_logprobs.get(i) {
            top_logprobs_entry
                .values
                .iter()
                .enumerate()
                .filter_map(|(j, &top_logprob)| {
                    top_logprobs_entry.token_ids.get(j).map(|&tid| {
                        let text = decode_token(tid);
                        let bytes = Some(text.as_bytes().to_vec());
                        TopLogProb {
                            token: text,
                            logprob: top_logprob,
                            bytes,
                        }
                    })
                })
                .collect()
        } else {
            Vec::new()
        };

        content_items.push(ChatLogProbsContent {
            token: token_text,
            logprob,
            bytes,
            top_logprobs,
        });
    }

    ChatLogProbs::Detailed {
        content: (!content_items.is_empty()).then_some(content_items),
    }
}

/// Convert OutputLogProbs to OpenAI ChatLogProbs format using a Tokenizer
pub(crate) fn convert_proto_to_openai_logprobs(
    proto_logprobs: &ProtoOutputLogProbs,
    tokenizer: &Arc<dyn Tokenizer>,
) -> ChatLogProbs {
    convert_proto_logprobs(proto_logprobs, |token_id| {
        tokenizer
            .decode(&[token_id], false)
            .unwrap_or_else(|_| format!("<token_{token_id}>"))
    })
}

/// Convert OutputLogProbs to Generate format Vec<Vec<Option<f64>>>
///
/// Generate format: [[logprob, token_id, ...], [logprob, token_id, ...], ...]
/// Each inner vec contains [logprob (f64), token_id (u32), ...]
pub(crate) fn convert_generate_output_logprobs(
    proto_logprobs: &ProtoOutputLogProbs,
) -> Vec<Vec<Option<f64>>> {
    proto_logprobs
        .token_logprobs
        .iter()
        .zip(proto_logprobs.token_ids.iter())
        .map(|(&logprob, &token_id)| vec![Some(logprob as f64), Some(token_id as f64)])
        .collect()
}

/// Convert InputLogProbs to Generate format Vec<Vec<Option<f64>>>
///
/// Generate format: [[logprob, token_id, ...], [logprob, token_id, ...], ...]
/// First token has null logprob: [[null, token_id], [logprob, token_id], ...]
pub(crate) fn convert_generate_input_logprobs(
    proto_logprobs: &ProtoInputLogProbs,
) -> Vec<Vec<Option<f64>>> {
    proto_logprobs
        .token_logprobs
        .iter()
        .zip(proto_logprobs.token_ids.iter())
        .map(|(&token_logprob, &token_id)| {
            // token_logprob is Option<f32> in unified type
            let logprob_value = token_logprob.map(|v| v as f64);
            vec![logprob_value, Some(token_id as f64)]
        })
        .collect()
}

/// Preserve raw prompt alternatives without decoding token IDs.
pub(crate) fn convert_generate_input_top_logprobs(
    proto_logprobs: &ProtoInputLogProbs,
) -> Vec<Vec<(f64, u32)>> {
    proto_logprobs
        .token_ids
        .iter()
        .enumerate()
        .map(|(position, _)| {
            if position == 0 {
                return Vec::new();
            }
            proto_logprobs
                .top_logprobs
                .get(position)
                .map(|top| {
                    top.values
                        .iter()
                        .zip(&top.token_ids)
                        .map(|(&probability, &token)| (f64::from(probability), token))
                        .collect()
                })
                .unwrap_or_default()
        })
        .collect()
}

#[cfg(test)]
mod prompt_top_logprobs_tests {
    use super::*;
    use crate::routers::grpc::proto_wrapper::ProtoTopLogProbs;

    #[test]
    fn preserves_raw_ids_and_first_token_placeholder() {
        let prompt = ProtoInputLogProbs {
            token_logprobs: vec![None, Some(-0.5), Some(-0.25)],
            token_ids: vec![12, 99, 7],
            top_logprobs: vec![
                ProtoTopLogProbs {
                    values: vec![],
                    token_ids: vec![],
                },
                ProtoTopLogProbs {
                    values: vec![-0.5, -1.0],
                    token_ids: vec![99, 42],
                },
                ProtoTopLogProbs {
                    values: vec![-0.25, -2.0],
                    token_ids: vec![7, 88],
                },
            ],
        };
        let alternatives = convert_generate_input_top_logprobs(&prompt);
        assert_eq!(
            alternatives,
            vec![
                vec![],
                vec![(-0.5, 99), (-1.0, 42)],
                vec![(-0.25, 7), (-2.0, 88)]
            ]
        );
        let json = serde_json::to_value(alternatives).unwrap();
        assert_eq!(json[0], serde_json::json!([]));
        assert_eq!(json[1][0], serde_json::json!([-0.5, 99]));
        let sampled = convert_generate_input_logprobs(&prompt);
        assert_eq!(sampled[0], vec![None, Some(12.0)]);
        assert_eq!(sampled[2], vec![Some(-0.25), Some(7.0)]);
    }

    #[test]
    fn missing_alternatives_remain_empty() {
        let prompt = ProtoInputLogProbs {
            token_logprobs: vec![None, Some(-0.5)],
            token_ids: vec![12, 99],
            top_logprobs: vec![],
        };
        assert_eq!(
            convert_generate_input_top_logprobs(&prompt),
            vec![vec![], vec![]]
        );
    }
}
