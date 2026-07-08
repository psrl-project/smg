//! Generate request building stage: Build proto GenerateRequest for generate requests

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;
use uuid::Uuid;

use crate::routers::{
    error,
    grpc::{
        common::stages::{helpers, PipelineStage},
        context::{
            ClientSelection, ExecutionPlan, ExecutionPlanKind, PreparationOutput, RequestContext,
        },
        multimodal::assemble_multimodal_data,
    },
};

/// Generate request building stage
///
/// Extracts generate-specific request building logic from the old unified RequestBuildingStage.
pub(crate) struct GenerateRequestBuildingStage {
    inject_pd_metadata: bool,
    plan_kind: ExecutionPlanKind,
}

impl GenerateRequestBuildingStage {
    pub fn new(inject_pd_metadata: bool, plan_kind: ExecutionPlanKind) -> Self {
        Self {
            inject_pd_metadata,
            plan_kind,
        }
    }
}

#[async_trait]
impl PipelineStage for GenerateRequestBuildingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let prep = ctx.state.preparation.take().ok_or_else(|| {
            error!(
                function = "GenerateRequestBuildingStage::execute",
                "Preparation not completed"
            );
            error::internal_error("preparation_not_completed", "Preparation not completed")
        })?;

        let clients = ctx.state.clients.as_ref().ok_or_else(|| {
            error!(
                function = "GenerateRequestBuildingStage::execute",
                "Client acquisition not completed"
            );
            error::internal_error(
                "client_acquisition_not_completed",
                "Client acquisition not completed",
            )
        })?;

        let generate_request = ctx.generate_request_arc();

        // Get client for building request (use prefill client in disaggregated mode)
        let builder_client = match clients {
            ClientSelection::Single { client } => client,
            ClientSelection::Disaggregated { prefill, .. } => prefill,
        };

        // Build generate request. PD retries re-run this stage, and a NIXL-tagged
        // prefill keeps the request_id alive on the worker until the KV lease
        // expires, so client rids get a unique per-attempt engine id in PD mode.
        let disaggregated = matches!(clients, ClientSelection::Disaggregated { .. });
        let request_id = match generate_request.rid.clone() {
            Some(rid) if disaggregated => format!("{rid}-{}", Uuid::now_v7()),
            Some(rid) => rid,
            None => format!("gen-{}", Uuid::now_v7()),
        };

        let PreparationOutput::Generate {
            original_text,
            token_ids,
            multimodal_intermediate,
        } = prep
        else {
            debug_assert!(false, "pipeline guarantees Generate variant");
            return Err(error::internal_error(
                "wrong_preparation_type",
                "Expected Generate preparation output",
            ));
        };

        let multimodal_data = if let Some(intermediate) = multimodal_intermediate {
            Some(
                assemble_multimodal_data(intermediate, builder_client, ctx.state.workers.as_ref())
                    .await
                    .map_err(|e| {
                error!(function = "GenerateRequestBuildingStage::execute", error = %e, "Failed to assemble multimodal request");
                error::bad_request("multimodal_not_supported", format!("{e}"))
                    })?,
            )
        } else {
            None
        };

        // Build proto request using centralized dispatch
        let mut proto_request = builder_client
            .build_generate_request(
                request_id,
                &generate_request,
                original_text,
                token_ids,
                multimodal_data,
            )
            .map_err(|e| {
                error!(function = "GenerateRequestBuildingStage::execute", error = %e, "Failed to build generate request");
                error::bad_request("build_request_failed", e)
            })?;

        helpers::apply_sampling_defaults_to_generate_request(
            &mut proto_request,
            &ctx.input.request_type,
            ctx.state.workers.as_ref(),
        );

        if self.inject_pd_metadata {
            if let Some(workers) = ctx.state.workers.as_ref() {
                helpers::maybe_inject_pd_metadata(&mut proto_request, workers);
            }
        }

        // EPD: inject the prefill->decode KV rendezvous for backends that carry it
        // in the request. No-op unless the selected workers are TokenSpeed EPD.
        if let Some(workers) = ctx.state.workers.as_ref() {
            helpers::maybe_inject_pd_rendezvous(&mut proto_request, workers);
        }

        ctx.state.execution_plan =
            Some(ExecutionPlan::generate(self.plan_kind, proto_request, None));
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "GenerateRequestBuilding"
    }
}
