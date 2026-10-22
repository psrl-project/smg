//! Completion preparation stage: resolve prompt, tokenize, create stop decoder.
//!
//! This is the `/v1/completions` Stage 1 equivalent. It intentionally builds on top of
//! the native completion pipeline typing introduced in PR #840. It keeps
//! `CompletionRequest` native in the request context instead of laundering it
//! through `GenerateRequest`.

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use llm_tokenizer::traits::Tokenizer;
use openai_protocol::completion::CompletionPrompt;

use crate::routers::{
    error,
    grpc::{
        common::stages::{PipelineStage, StagePhase},
        context::{PreparationOutput, RequestContext},
        utils,
    },
};

pub(crate) struct CompletionPreparationStage;

#[async_trait]
impl PipelineStage for CompletionPreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let request = ctx.completion_request_arc();

        let tokenizer =
            utils::resolve_tokenizer(ctx, "CompletionPreparationStage::execute").map_err(|e| *e)?;

        let (prompt_text, token_ids) =
            resolve_completion_prompt(tokenizer.clone(), &request.prompt)
                .await
                .map_err(|message| error::bad_request("invalid_prompt", message))?;

        let stop_decoder = utils::create_stop_decoder(
            &tokenizer,
            request.stop.as_ref(),
            request.stop_token_ids.as_ref(),
            request.skip_special_tokens,
            request.no_stop_trim,
            request.ignore_eos,
        );

        ctx.state.preparation = Some(PreparationOutput::Completion {
            original_text: prompt_text,
            token_ids,
        });

        ctx.state.response.stop_decoder = Some(stop_decoder);

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "CompletionPreparation"
    }

    fn phase(&self) -> StagePhase {
        StagePhase::Preparation
    }
}

async fn resolve_completion_prompt(
    tokenizer: Arc<dyn Tokenizer>,
    prompt: &CompletionPrompt,
) -> Result<(String, Vec<u32>), String> {
    match prompt {
        CompletionPrompt::String(text) => {
            let encoding = utils::encode_blocking(tokenizer, text.clone(), false)
                .await
                .map_err(|error| format!("Tokenization failed: {error}"))?;
            Ok((text.clone(), encoding.token_ids().to_vec()))
        }
        CompletionPrompt::TokenIds(ids) => {
            let token_ids = ids.clone();
            tokio::task::spawn_blocking(move || {
                let text = tokenizer
                    .decode(&token_ids, false)
                    .map_err(|error| format!("Prompt decoding failed: {error}"))?;
                Ok((text, token_ids))
            })
            .await
            .map_err(|error| format!("Prompt decoding task failed: {error}"))?
        }
        CompletionPrompt::Array(_) | CompletionPrompt::TokenIdBatch(_) => {
            Err("Batched prompts are not supported for gRPC /v1/completions.".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use llm_tokenizer::MockTokenizer;

    use super::*;

    #[tokio::test]
    async fn completion_preparation_preserves_special_and_space_tokens() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
        let ids = vec![1000, 1, 5, 2, 1002];
        let (_, prepared) =
            resolve_completion_prompt(tokenizer, &CompletionPrompt::TokenIds(ids.clone()))
                .await
                .unwrap();
        assert_eq!(prepared, ids);
    }

    #[tokio::test]
    async fn completion_preparation_keeps_text_tokenization() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
        let (text, ids) =
            resolve_completion_prompt(tokenizer, &CompletionPrompt::String("Hello world".into()))
                .await
                .unwrap();
        assert_eq!(text, "Hello world");
        assert_eq!(ids, vec![1, 2]);
    }
}
