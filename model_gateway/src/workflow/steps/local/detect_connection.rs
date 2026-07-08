//! Connection mode detection step.
//!
//! Determines whether a worker communicates via HTTP or gRPC.
//! This step only answers "HTTP or gRPC?" — backend runtime detection
//! (sglang vs vllm vs trtllm) is handled by the separate DetectBackendStep.

use async_trait::async_trait;
use tracing::debug;
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    worker::ConnectionMode,
    workflow::{
        data::{WorkerKind, WorkerWorkflowData},
        steps::util::{try_grpc_reachable, try_http_reachable},
    },
};

/// Step 1: Verify the declared connection mode (HTTP vs gRPC).
///
/// Validates that the worker is reachable via the `connection_mode` explicitly
/// provided in the registration payload. Auto-detection is not performed —
/// callers must supply the mode manually.
/// Does NOT detect backend runtime — that's handled by DetectBackendStep.
pub struct DetectConnectionModeStep;

#[async_trait]
impl StepExecutor<WorkerWorkflowData> for DetectConnectionModeStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        if context.data.worker_kind != Some(WorkerKind::Local) {
            return Ok(StepResult::Skip);
        }

        let config = &context.data.config;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;

        debug!(
            "Detecting connection mode for {} (timeout: {:?}s, max_attempts: {})",
            config.url, config.health.timeout_secs, config.max_connection_attempts
        );

        let url = config.url.clone();
        let timeout = config
            .health
            .timeout_secs
            .unwrap_or(app_context.router_config.health_check.timeout_secs);
        let client = &app_context.client;

        let connection_mode = config.connection_mode;

        // Verify the declared connection mode.
        match connection_mode {
            ConnectionMode::Http => {
                try_http_reachable(&url, timeout, client)
                    .await
                    .map_err(|e| WorkflowError::StepFailed {
                        step_id: StepId::new("detect_connection_mode"),
                        message: format!(
                            "HTTP health check failed for {} \
                                (connection_mode=http was explicitly declared in the \
                                registration payload): {e}",
                            config.url
                        ),
                    })?;
                debug!(
                    "{} confirmed reachable via HTTP (as declared in payload)",
                    config.url
                );
            }
            ConnectionMode::Grpc => {
                try_grpc_reachable(&url, timeout)
                    .await
                    .map_err(|e| WorkflowError::StepFailed {
                        step_id: StepId::new("detect_connection_mode"),
                        message: format!(
                            "gRPC health check failed for {} \
                                (connection_mode=grpc was explicitly declared in the \
                                registration payload): {e}",
                            config.url
                        ),
                    })?;
                debug!(
                    "{} confirmed reachable via gRPC (as declared in payload)",
                    config.url
                );
            }
        }

        context.data.connection_mode = Some(connection_mode);
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }
}
