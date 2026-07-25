//! Worker selection stage: Select appropriate worker(s) based on routing mode

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use tracing::{error, warn};

use super::{worker_selector::WorkerSelectorStrategy, PipelineStage, StagePhase};
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    policies::{PolicyRegistry, SelectWorkerInfo},
    routers::{
        error,
        grpc::{
            context::{LoadGuards, RequestContext, WorkerSelection},
            routing_loop::metadata::parse_routing_request_meta_from_context,
        },
    },
    worker::{ConnectionMode, RuntimeType, Worker, WorkerRegistry, WorkerType, UNKNOWN_MODEL_ID},
};

/// Result type for PD worker pair selection: (prefill, decode, runtime_type)
type PdWorkerPair = (Arc<dyn Worker>, Arc<dyn Worker>, RuntimeType);

/// Internal representation of the two selection modes.
enum WorkerSelectionMode {
    /// Regular (single-worker) mode: delegates to a pluggable strategy.
    Regular {
        strategy: Arc<dyn WorkerSelectorStrategy>,
    },
    /// PD (prefill-decode) mode: always uses policy-based naive selection.
    PrefillDecode {
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
    },
}

/// Worker selection stage: selects appropriate worker(s) for the current routing mode.
pub(crate) struct WorkerSelectionStage {
    inner: WorkerSelectionMode,
}

impl WorkerSelectionStage {
    /// Construct a Regular-mode stage that delegates to the given strategy.
    pub fn new_regular(strategy: Arc<dyn WorkerSelectorStrategy>) -> Self {
        Self {
            inner: WorkerSelectionMode::Regular { strategy },
        }
    }

    /// Construct a PD-mode stage using policy-based naive selection.
    pub fn new_pd(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
    ) -> Self {
        Self {
            inner: WorkerSelectionMode::PrefillDecode {
                worker_registry,
                policy_registry,
            },
        }
    }
}

#[async_trait]
impl PipelineStage for WorkerSelectionStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let prep = ctx.state.preparation.as_ref().ok_or_else(|| {
            error!(
                function = "WorkerSelectionStage::execute",
                "Preparation stage not completed"
            );
            error::internal_error(
                "preparation_stage_not_completed",
                "Preparation stage not completed",
            )
        })?;

        let text = prep.routing_text();

        // Get tokens for PrefixHash policy support
        let ids = prep.token_ids();
        let tokens = if ids.is_empty() { None } else { Some(ids) };

        let headers = ctx.input.headers.as_ref();

        let model_id = ctx.input.model_id.as_str();
        let workers = match &self.inner {
            WorkerSelectionMode::Regular { strategy } => {
                let routing_meta = parse_routing_request_meta_from_context(ctx);

                match strategy
                    .select_single_worker(model_id, text, tokens, headers, routing_meta.as_ref())
                    .await
                {
                    Some(w) => WorkerSelection::Single { worker: w },
                    None => {
                        // No worker available — the routing loop will re-enqueue
                        // this request; log at debug level only to avoid noise.
                        tracing::debug!(
                            function = "WorkerSelectionStage::execute",
                            mode = "Regular",
                            model_id = %model_id,
                            "No available workers for model; request will be re-enqueued"
                        );
                        return Err(error::model_not_found(model_id));
                    }
                }
            }
            WorkerSelectionMode::PrefillDecode {
                worker_registry,
                policy_registry,
            } => {
                match select_pd_pair(
                    model_id,
                    text,
                    tokens,
                    headers,
                    worker_registry,
                    policy_registry,
                ) {
                    Some((prefill, decode, runtime_type)) => WorkerSelection::Dual {
                        prefill,
                        decode,
                        runtime_type,
                    },
                    None => {
                        // No PD pair available — the routing loop will re-enqueue
                        // this request; log at debug level only to avoid noise.
                        tracing::debug!(
                            function = "WorkerSelectionStage::execute",
                            mode = "PrefillDecode",
                            model_id = %model_id,
                            "No available PD worker pairs for model; request will be re-enqueued"
                        );
                        return Err(error::model_not_found(model_id));
                    }
                }
            }
        };

        // Capture the admit-time token estimate while `preparation` is still
        // present (request_building consumes it via `.take()` before the
        // execution stage runs). Prompt tokens + response-tokens-so-far gives
        // the request's current token footprint; the execution stage feeds it
        // into the worker's inflight-token counter when minting LoadGuards so
        // load-aware policies account for just-admitted requests between
        // engine snapshots. Computed before the mutable `ctx.state.workers`
        // write so the immutable `ids`/`ctx` borrows end first.
        let response_so_far = parse_routing_request_meta_from_context(ctx)
            .and_then(|meta| meta.response_token_count)
            .unwrap_or(0);
        let admit_token_estimate = ids.len() + response_so_far;

        ctx.state.workers = Some(workers);
        ctx.state.admit_token_estimate = Some(admit_token_estimate);
        // Use from_pre_incremented for Regular mode: the selector already
        // called increment_load() atomically with select_worker() to prevent
        // the TOCTOU race. For PD mode the load was not pre-incremented, so
        // it falls through to with_token_estimate via the else branch below.
        ctx.state.load_guards = Some(match &self.inner {
            WorkerSelectionMode::Regular { .. } => LoadGuards::from_pre_incremented(
                ctx.state.workers.as_ref().unwrap(),
                ctx.input.headers.as_ref(),
                ctx.state.admit_token_estimate,
            ),
            WorkerSelectionMode::PrefillDecode { .. } => LoadGuards::with_token_estimate(
                ctx.state.workers.as_ref().unwrap(),
                ctx.input.headers.as_ref(),
                ctx.state.admit_token_estimate,
            ),
        });
        Ok(None)
    }

    async fn commit(&self, ctx: &mut RequestContext) -> Result<(), Response> {
        let WorkerSelectionMode::Regular { strategy } = &self.inner else {
            return Ok(());
        };
        let Some(WorkerSelection::Single { worker }) = ctx.state.workers.as_ref() else {
            return Err(error::internal_error(
                "worker_selection_not_completed",
                "Worker selection must complete before commit",
            ));
        };
        let tokens = ctx
            .state
            .preparation
            .as_ref()
            .or(ctx.state.preparation_snapshot.as_ref())
            .map(|preparation| preparation.token_ids())
            .filter(|ids| !ids.is_empty());
        let routing_meta = parse_routing_request_meta_from_context(ctx);
        let pinned_version = strategy
            .commit_single_worker(
                ctx.input.model_id.as_str(),
                tokens,
                routing_meta.as_ref(),
                worker,
            )
            .await?;

        // Persist a freshly pinned version tag back into the request headers so
        // downstream re-routes (partial-rollout loopback, later agent turns)
        // parse the pinned version instead of the original `-1`.
        if let Some(version) = pinned_version {
            if let Ok(value) = http::HeaderValue::from_str(&version.to_string()) {
                ctx.input
                    .headers
                    .get_or_insert_with(Default::default)
                    .insert("x-version-tag", value);
            }
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "WorkerSelection"
    }

    fn phase(&self) -> StagePhase {
        StagePhase::WorkerSelection
    }
}

fn select_pd_pair(
    model_id: &str,
    text: Option<&str>,
    tokens: Option<&[u32]>,
    headers: Option<&http::HeaderMap>,
    worker_registry: &Arc<WorkerRegistry>,
    policy_registry: &Arc<PolicyRegistry>,
) -> Option<PdWorkerPair> {
    // Treat "unknown" model as wildcard (match any worker)
    let model_filter = if model_id == UNKNOWN_MODEL_ID {
        None
    } else {
        Some(model_id)
    };

    let all_workers = worker_registry.get_workers_filtered(
        model_filter,
        None,
        Some(ConnectionMode::Grpc), // Match any gRPC worker
        None,                       // any runtime type
        false,
    );

    let (all_prefill, all_decode): (Vec<_>, Vec<_>) =
        all_workers
            .into_iter()
            .fold((Vec::new(), Vec::new()), |mut acc, w| {
                if w.is_available() {
                    match w.metadata().spec.worker_type {
                        WorkerType::Prefill => acc.0.push(w),
                        WorkerType::Decode => acc.1.push(w),
                        WorkerType::Regular => {}
                    }
                }
                acc
            });

    if all_prefill.is_empty() {
        warn!("No available prefill workers");
        return None;
    }

    if all_decode.is_empty() {
        warn!("No available decode workers");
        return None;
    }

    // Determine the runtime type from prefill workers.
    // All workers in a PD pair must use the same runtime.
    let first_runtime = all_prefill.first()?.metadata().spec.runtime_type;

    // Check for mixed runtimes in both prefill and decode pools
    let prefill_mixed = all_prefill
        .iter()
        .skip(1)
        .any(|w| w.metadata().spec.runtime_type != first_runtime);
    let decode_mixed = all_decode
        .iter()
        .any(|w| w.metadata().spec.runtime_type != first_runtime);

    if prefill_mixed || decode_mixed {
        warn!(
            "Mixed runtime types in PD workers (prefill_mixed={}, decode_mixed={}). Using {:?}.",
            prefill_mixed, decode_mixed, first_runtime
        );
    }

    let target_runtime = first_runtime;

    // Filter both pools to the target runtime
    let available_prefill: Vec<_> = all_prefill
        .into_iter()
        .filter(|w| w.metadata().spec.runtime_type == target_runtime)
        .collect();
    let available_decode: Vec<_> = all_decode
        .into_iter()
        .filter(|w| w.metadata().spec.runtime_type == target_runtime)
        .collect();

    if available_prefill.is_empty() || available_decode.is_empty() {
        warn!("No available PD pair for runtime {:?}", target_runtime);
        return None;
    }

    // Select using policies (PD mode always uses naive policy-based selection)
    let policy = policy_registry.get_policy_or_default(model_id);

    // Get cached hash ring for consistent hashing (O(log n) lookup)
    let hash_ring = worker_registry.get_hash_ring(model_id);

    let info = SelectWorkerInfo {
        request_text: text,
        tokens,
        headers,
        hash_ring,
        response_token_count: None,
        priority_groups: None,
        score_trace: None,
    };
    let prefill_idx = policy.select_worker(&available_prefill, &info)?;
    let decode_idx = policy.select_worker(&available_decode, &info)?;

    let policy_name = policy.name();

    // Record worker selection metrics for both prefill and decode
    Metrics::record_worker_selection(
        metrics_labels::WORKER_PREFILL,
        metrics_labels::CONNECTION_GRPC,
        model_id,
        policy_name,
    );
    Metrics::record_worker_selection(
        metrics_labels::WORKER_DECODE,
        metrics_labels::CONNECTION_GRPC,
        model_id,
        policy_name,
    );

    Some((
        available_prefill[prefill_idx].clone(),
        available_decode[decode_idx].clone(),
        target_runtime,
    ))
}
