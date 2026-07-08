//! Generate preparation stage: Resolve input, tokenize, create stop decoder

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use llm_multimodal::Modality;
use llm_tokenizer::traits::Tokenizer;
use openai_protocol::{common::InputIds, generate::GenerateRequest};
use tracing::{debug, error};

use crate::routers::{
    error,
    grpc::{
        common::stages::{PipelineStage, StagePhase},
        context::{PreparationOutput, RequestContext},
        multimodal, utils,
    },
};

/// Generate preparation stage
///
/// Extracts generate-specific preparation logic from the old unified PreparationStage.
/// This is a direct extraction without architectural changes.
pub(crate) struct GeneratePreparationStage;

#[async_trait]
impl PipelineStage for GeneratePreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let request = ctx.generate_request_arc();
        self.prepare_generate(ctx, &request).await?;
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "GeneratePreparation"
    }

    fn phase(&self) -> StagePhase {
        StagePhase::Preparation
    }
}

impl GeneratePreparationStage {
    async fn prepare_generate(
        &self,
        ctx: &mut RequestContext,
        request: &GenerateRequest,
    ) -> Result<(), Response> {
        // Resolve tokenizer from registry (cached for reuse in response processing)
        let tokenizer = utils::resolve_tokenizer(ctx, "GeneratePreparationStage::prepare_generate")
            .map_err(|e| *e)?;

        let (original_text, mut token_ids) = match self
            .resolve_generate_input(request, &tokenizer)
            .await
        {
            Ok(res) => res,
            Err(msg) => {
                error!(function = "GeneratePreparationStage::execute", error = %msg, "Failed to resolve generate input");
                return Err(error::bad_request("resolve_input_failed", msg));
            }
        };

        let mut multimodal_intermediate = None;
        if multimodal::has_multimodal_content_generate(request) {
            if multimodal::has_unsupported_generate_modality(request) {
                return Err(error::bad_request(
                    "multimodal_modality_not_supported",
                    "SMG /generate gRPC multimodal fast path currently supports image inputs only",
                ));
            }

            let mm_components = ctx.components.multimodal.as_ref().ok_or_else(|| {
                error!(
                    function = "GeneratePreparationStage::execute",
                    "Multimodal content detected but multimodal components not initialized"
                );
                error::bad_request(
                    "multimodal_not_supported",
                    "Multimodal content detected but multimodal processing is not available",
                )
            })?;

            let model_id = ctx.input.model_id.clone();
            let entry = ctx
                .components
                .tokenizer_registry
                .get_by_name(&model_id)
                .or_else(|| ctx.components.tokenizer_registry.get_by_id(&model_id))
                .ok_or_else(|| {
                    error!(
                        function = "GeneratePreparationStage::execute",
                        model = %model_id,
                        "Tokenizer entry not found for multimodal processing"
                    );
                    error::bad_request(
                        "multimodal_config_missing",
                        format!("Tokenizer not found for model: {model_id}"),
                    )
                })?;

            multimodal::resolve_placeholder_token(
                &model_id,
                &*tokenizer,
                mm_components,
                &entry.id,
                &entry.source,
                Modality::Image,
            )
            .await
            .map_err(|e| {
                error!(
                    function = "GeneratePreparationStage::execute",
                    model = %model_id,
                    error = %e,
                    "Failed to resolve multimodal placeholder token"
                );
                error::internal_error(
                    "multimodal_placeholder_resolution_failed",
                    format!("Failed to resolve multimodal placeholder token: {e}"),
                )
            })?;

            match multimodal::process_multimodal_generate(
                request,
                &model_id,
                &*tokenizer,
                token_ids,
                mm_components,
                &entry.id,
                &entry.source,
            )
            .await
            {
                Ok(output) => {
                    debug!(
                        function = "GeneratePreparationStage::execute",
                        expanded_tokens = output.expanded_token_ids.len(),
                        "Generate multimodal processing complete"
                    );
                    token_ids = output.expanded_token_ids;
                    multimodal_intermediate = Some(output.intermediate);
                }
                Err(e) => {
                    error!(
                        function = "GeneratePreparationStage::execute",
                        error = %e,
                        "Generate multimodal processing failed"
                    );
                    return Err(error::bad_request(
                        "multimodal_processing_failed",
                        format!("Multimodal processing failed: {e}"),
                    ));
                }
            }
        }

        // Create stop sequence decoder for generate requests
        let params = request.sampling_params.as_ref();
        let stop_decoder = utils::create_stop_decoder(
            &tokenizer,
            params.and_then(|p| p.stop.as_ref()),
            params.and_then(|p| p.stop_token_ids.as_ref()),
            params.and_then(|p| p.skip_special_tokens).unwrap_or(true),
            params.and_then(|p| p.no_stop_trim).unwrap_or(false),
            params.and_then(|p| p.ignore_eos).unwrap_or(false),
        );

        ctx.state.preparation = Some(PreparationOutput::Generate {
            original_text,
            token_ids,
            multimodal_intermediate,
        });

        // Store stop decoder
        ctx.state.response.stop_decoder = Some(stop_decoder);

        Ok(())
    }

    async fn resolve_generate_input(
        &self,
        request: &GenerateRequest,
        tokenizer: &Arc<dyn Tokenizer>,
    ) -> Result<(Option<String>, Vec<u32>), String> {
        if let Some(text) = &request.text {
            return self
                .tokenize_single_text(tokenizer, text)
                .await
                .map(|(original, ids)| (Some(original), ids));
        }

        // Handle input_ids - validate and convert
        if let Some(input_ids) = &request.input_ids {
            return match input_ids {
                InputIds::Single(ids) => ids
                    .iter()
                    .map(|&id| u32::try_from(id))
                    .collect::<Result<Vec<u32>, _>>()
                    .map(|converted| (None, converted))
                    .map_err(|_| "input_ids must be non-negative".to_string()),
                InputIds::Batch(_) => {
                    Err("Batch input_ids are not supported over gRPC generate yet".to_string())
                }
            };
        }

        Err("Either `text` or `input_ids` must be provided".to_string())
    }

    async fn tokenize_single_text(
        &self,
        tokenizer: &Arc<dyn Tokenizer>,
        text: &str,
    ) -> Result<(String, Vec<u32>), String> {
        // Don't add special tokens - raw text generation uses text as-is
        let encoding = utils::encode_blocking(tokenizer.clone(), text.to_string(), false)
            .await
            .map_err(|e| format!("Tokenization failed: {e}"))?;
        Ok((text.to_string(), encoding.token_ids().to_vec()))
    }
}
