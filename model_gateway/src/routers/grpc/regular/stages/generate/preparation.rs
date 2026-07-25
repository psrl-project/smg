//! Generate preparation stage: Resolve input, tokenize, create stop decoder

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use llm_multimodal::Modality;
use llm_tokenizer::traits::Tokenizer;
use openai_protocol::{
    common::InputIds,
    generate::{GenerateRequest, MultimodalTokenMode},
};
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
        let media_plan = multimodal::media_plan_generate(request);
        if request.preprocessed_mm_inputs.is_some() && media_plan.is_empty() {
            return Err(error::bad_request(
                "invalid_preprocessed_mm_inputs",
                "preprocessed_mm_inputs require at least one valid image_data reference",
            ));
        }
        if let Some(preprocessing) = &request.image_preprocessing {
            let image_count = media_plan.count(Modality::Image);
            if preprocessing.resize_targets.len() != image_count {
                return Err(error::bad_request(
                    "invalid_image_preprocessing",
                    format!(
                        "image_preprocessing has {} resize targets for {image_count} images",
                        preprocessing.resize_targets.len()
                    ),
                ));
            }
        }
        if !media_plan.is_empty() {
            if request.video_data.is_some() || request.audio_data.is_some() {
                return Err(error::bad_request(
                    "multimodal_modality_not_supported",
                    "/generate currently supports image_data only",
                ));
            }
            let components = ctx.components.multimodal.as_ref().ok_or_else(|| {
                error::bad_request(
                    "multimodal_not_supported",
                    "Multimodal processing is not configured",
                )
            })?;
            let model_id = &ctx.input.model_id;
            let entry = ctx
                .components
                .tokenizer_registry
                .get_by_name(model_id)
                .or_else(|| ctx.components.tokenizer_registry.get_by_id(model_id))
                .ok_or_else(|| {
                    error::bad_request(
                        "multimodal_config_missing",
                        format!("Tokenizer not found for model: {model_id}"),
                    )
                })?;
            let result = if request.preprocessed_mm_inputs.is_some() {
                multimodal::process_multimodal_plan_python_preprocessed(
                    media_plan,
                    model_id,
                    &*tokenizer,
                    token_ids,
                    components,
                    &entry.id,
                    &entry.source,
                )
                .await
            } else if let Some(preprocessing) = &request.image_preprocessing {
                multimodal::process_multimodal_plan_with_image_preprocessing(
                    media_plan,
                    model_id,
                    &*tokenizer,
                    token_ids,
                    components,
                    &entry.id,
                    &entry.source,
                    preprocessing.resize_targets.clone(),
                )
                .await
            } else if matches!(
                request.multimodal_token_mode,
                Some(MultimodalTokenMode::Preexpanded)
            ) {
                multimodal::process_multimodal_plan_preexpanded(
                    media_plan,
                    model_id,
                    &*tokenizer,
                    token_ids,
                    components,
                    &entry.id,
                    &entry.source,
                )
                .await
            } else {
                multimodal::process_multimodal_plan(
                    media_plan,
                    model_id,
                    &*tokenizer,
                    token_ids,
                    components,
                    &entry.id,
                    &entry.source,
                )
                .await
            }
            .map_err(|e| {
                error::bad_request(
                    "multimodal_processing_failed",
                    format!("Multimodal processing failed: {e}"),
                )
            })?;
            let result = if let Some(preprocessed) = &request.preprocessed_mm_inputs {
                multimodal::MultimodalOutput {
                    expanded_token_ids: result.expanded_token_ids,
                    intermediate: multimodal::apply_python_preprocessed_inputs(
                        result.intermediate,
                        preprocessed,
                    )
                    .map_err(|e| {
                        error::bad_request("invalid_preprocessed_mm_inputs", e.to_string())
                    })?,
                }
            } else {
                result
            };
            debug!(
                expanded_tokens = result.expanded_token_ids.len(),
                "Generate multimodal processing complete"
            );
            token_ids = result.expanded_token_ids;
            multimodal_intermediate = Some(result.intermediate);
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

        ctx.state.response.prompt_token_ids =
            request.return_prompt_token_ids.then(|| token_ids.clone());

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
