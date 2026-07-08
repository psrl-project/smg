//! Worker selection stage: selects appropriate worker(s) based on routing mode.

use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    http::{HeaderMap, HeaderValue},
    response::Response,
};
use tracing::{error, warn};

use super::{worker_selector::WorkerSelectorStrategy, PipelineStage, StagePhase};
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    policies::{LoadBalancingPolicy, PolicyRegistry, SelectWorkerInfo, WorkerLeg},
    routers::{
        error,
        grpc::{
            context::{
                EncodeWorkerAssignment, LoadGuards, PreparationOutput, RequestContext,
                WorkerSelection,
            },
            multimodal::{self, MultimodalIntermediate},
            routing_loop::metadata::parse_routing_request_meta_from_context,
        },
    },
    worker::{
        ConnectionMode, HashRing, RuntimeType, Worker, WorkerRegistry, WorkerType, UNKNOWN_MODEL_ID,
    },
};

type PdWorkerPair = (Arc<dyn Worker>, Arc<dyn Worker>, RuntimeType);
type EncodePrefillDecodeWorkerSelection = (
    Vec<EncodeWorkerAssignment>,
    Arc<dyn Worker>,
    Arc<dyn Worker>,
    RuntimeType,
);

/// Public construction mode used by pipeline builders for multi-leg selection.
pub(crate) enum WorkerSelectionMode {
    EncodePrefillDecode,
}

enum WorkerSelectionInner {
    Regular {
        strategy: Arc<dyn WorkerSelectorStrategy>,
    },
    PrefillDecode {
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
    },
    EncodePrefillDecode {
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
    },
}

/// Worker selection stage.
pub(crate) struct WorkerSelectionStage {
    inner: WorkerSelectionInner,
}

impl WorkerSelectionStage {
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        mode: WorkerSelectionMode,
    ) -> Self {
        match mode {
            WorkerSelectionMode::EncodePrefillDecode => Self {
                inner: WorkerSelectionInner::EncodePrefillDecode {
                    worker_registry,
                    policy_registry,
                },
            },
        }
    }

    pub fn new_regular(strategy: Arc<dyn WorkerSelectorStrategy>) -> Self {
        Self {
            inner: WorkerSelectionInner::Regular { strategy },
        }
    }

    pub fn new_pd(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
    ) -> Self {
        Self {
            inner: WorkerSelectionInner::PrefillDecode {
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
        let ids = prep.token_ids();
        let tokens = if ids.is_empty() { None } else { Some(ids) };
        let headers = ctx.input.headers.as_ref();
        let model_id = ctx.input.model_id.as_str();
        let routing_meta = parse_routing_request_meta_from_context(ctx);
        let response_so_far = routing_meta
            .as_ref()
            .and_then(|meta| meta.response_token_count)
            .unwrap_or(0);

        let workers = match &self.inner {
            WorkerSelectionInner::Regular { strategy } => {
                match strategy
                    .select_single_worker(model_id, text, tokens, headers, routing_meta.as_ref())
                    .await
                {
                    Some(worker) => WorkerSelection::Single { worker },
                    None => {
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
            WorkerSelectionInner::PrefillDecode {
                worker_registry,
                policy_registry,
            } => {
                match select_pd_pair(
                    model_id,
                    text,
                    tokens,
                    headers,
                    response_so_far,
                    worker_registry,
                    policy_registry,
                ) {
                    Some((prefill, decode, runtime_type)) => WorkerSelection::Disaggregated {
                        encode_assignments: None,
                        prefill,
                        decode,
                        runtime_type,
                    },
                    None => {
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
            WorkerSelectionInner::EncodePrefillDecode {
                worker_registry,
                policy_registry,
            } => {
                let encode_item_hashes = match encode_item_hashes(prep) {
                    Ok(hashes) => hashes,
                    Err(err) => {
                        error!(
                            function = "WorkerSelectionStage::execute",
                            error = %err,
                            "Failed to derive encode item routing hashes"
                        );
                        return Err(error::internal_error(
                            "encode_routing_hash_failed",
                            format!("Failed to derive encode routing hashes: {err}"),
                        ));
                    }
                };

                match select_encode_prefill_decode_workers(
                    model_id,
                    text,
                    tokens,
                    headers,
                    response_so_far,
                    &encode_item_hashes,
                    worker_registry,
                    policy_registry,
                ) {
                    Some((encode_assignments, prefill, decode, runtime_type)) => {
                        WorkerSelection::Disaggregated {
                            encode_assignments: if encode_assignments.is_empty() {
                                None
                            } else {
                                Some(encode_assignments)
                            },
                            prefill,
                            decode,
                            runtime_type,
                        }
                    }
                    None => {
                        tracing::debug!(
                            function = "WorkerSelectionStage::execute",
                            mode = "EncodePrefillDecode",
                            model_id = %model_id,
                            "No available encode/prefill/decode worker set; request will be re-enqueued"
                        );
                        return Err(error::model_not_found(model_id));
                    }
                }
            }
        };

        let admit_token_estimate = ids.len() + response_so_far;
        ctx.state.workers = Some(workers);
        ctx.state.admit_token_estimate = Some(admit_token_estimate);
        ctx.state.load_guards = Some(match &self.inner {
            WorkerSelectionInner::Regular { .. } => LoadGuards::from_pre_incremented(
                ctx.state.workers.as_ref().expect("workers just set"),
                ctx.input.headers.as_ref(),
                ctx.state.admit_token_estimate,
            ),
            WorkerSelectionInner::PrefillDecode { .. }
            | WorkerSelectionInner::EncodePrefillDecode { .. } => LoadGuards::with_token_estimate(
                ctx.state.workers.as_ref().expect("workers just set"),
                ctx.input.headers.as_ref(),
                ctx.state.admit_token_estimate,
            ),
        });

        Ok(None)
    }

    async fn commit(&self, ctx: &mut RequestContext) -> Result<(), Response> {
        let WorkerSelectionInner::Regular { strategy } = &self.inner else {
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

        if let Some(version) = pinned_version {
            if let Ok(value) = HeaderValue::from_str(&version.to_string()) {
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

fn model_filter(model_id: &str) -> Option<&str> {
    (model_id != UNKNOWN_MODEL_ID).then_some(model_id)
}

fn select_pd_pair(
    model_id: &str,
    text: Option<&str>,
    tokens: Option<&[u32]>,
    headers: Option<&HeaderMap>,
    response_token_count: usize,
    worker_registry: &WorkerRegistry,
    policy_registry: &PolicyRegistry,
) -> Option<PdWorkerPair> {
    let all_workers = worker_registry.get_workers_filtered(
        model_filter(model_id),
        None,
        Some(ConnectionMode::Grpc),
        None,
        false,
    );

    let (all_prefill, all_decode): (Vec<_>, Vec<_>) =
        all_workers
            .into_iter()
            .fold((Vec::new(), Vec::new()), |mut acc, worker| {
                if worker.is_available() {
                    match worker.metadata().spec.worker_type {
                        WorkerType::Prefill => acc.0.push(worker),
                        WorkerType::Decode => acc.1.push(worker),
                        WorkerType::Regular | WorkerType::Encode => {}
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

    let first_runtime = all_prefill.first()?.metadata().spec.runtime_type;
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

    let policy = policy_registry.get_policy_or_default(model_id);
    let hash_ring = worker_registry.get_hash_ring(model_id);
    let mut info = SelectWorkerInfo {
        request_text: text,
        tokens,
        headers,
        hash_ring,
        response_token_count: Some(response_token_count),
        priority_groups: None,
        leg: WorkerLeg::Prefill,
    };
    let prefill_idx = policy_registry.select_worker(&policy, &available_prefill, &info)?;
    info.leg = WorkerLeg::Decode;
    let decode_idx = policy_registry.select_worker(&policy, &available_decode, &info)?;

    let policy_name = policy.name();
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
        Arc::clone(&available_prefill[prefill_idx]),
        Arc::clone(&available_decode[decode_idx]),
        target_runtime,
    ))
}

fn select_encode_prefill_decode_workers(
    model_id: &str,
    text: Option<&str>,
    tokens: Option<&[u32]>,
    headers: Option<&HeaderMap>,
    response_token_count: usize,
    encode_item_hashes: &[Vec<u8>],
    worker_registry: &WorkerRegistry,
    policy_registry: &PolicyRegistry,
) -> Option<EncodePrefillDecodeWorkerSelection> {
    let all_workers = worker_registry.get_workers_filtered(
        model_filter(model_id),
        None,
        Some(ConnectionMode::Grpc),
        None,
        false,
    );

    let (all_encode, all_prefill, all_decode): (Vec<_>, Vec<_>, Vec<_>) = all_workers
        .into_iter()
        .fold((Vec::new(), Vec::new(), Vec::new()), |mut acc, worker| {
            if worker.is_available() {
                match worker.metadata().spec.worker_type {
                    WorkerType::Encode => acc.0.push(worker),
                    WorkerType::Prefill => acc.1.push(worker),
                    WorkerType::Decode => acc.2.push(worker),
                    WorkerType::Regular => {}
                }
            }
            acc
        });

    let needs_encode = !encode_item_hashes.is_empty();
    if needs_encode && all_encode.is_empty() {
        warn!("No available encode workers");
        return None;
    }
    if all_prefill.is_empty() {
        warn!("No available prefill workers");
        return None;
    }
    if all_decode.is_empty() {
        warn!("No available decode workers");
        return None;
    }

    let Some(target_runtime) = all_prefill
        .iter()
        .map(|w| w.metadata().spec.runtime_type)
        .find(|runtime| {
            all_decode
                .iter()
                .any(|w| w.metadata().spec.runtime_type == *runtime)
                && (!needs_encode
                    || all_encode
                        .iter()
                        .any(|w| w.metadata().spec.runtime_type == *runtime))
        })
    else {
        warn!("No available encode/prefill/decode worker set with a shared runtime");
        return None;
    };

    let mixed = all_prefill
        .iter()
        .chain(all_decode.iter())
        .any(|w| w.metadata().spec.runtime_type != target_runtime)
        || (needs_encode
            && all_encode
                .iter()
                .any(|w| w.metadata().spec.runtime_type != target_runtime));
    if mixed {
        warn!(
            "Mixed runtime types in encode/prefill/decode workers. Using {:?}.",
            target_runtime
        );
    }

    let available_encode: Vec<_> = all_encode
        .into_iter()
        .filter(|w| w.metadata().spec.runtime_type == target_runtime)
        .collect();
    let available_prefill: Vec<_> = all_prefill
        .into_iter()
        .filter(|w| w.metadata().spec.runtime_type == target_runtime)
        .collect();
    let available_decode: Vec<_> = all_decode
        .into_iter()
        .filter(|w| w.metadata().spec.runtime_type == target_runtime)
        .collect();
    if (needs_encode && available_encode.is_empty())
        || available_prefill.is_empty()
        || available_decode.is_empty()
    {
        warn!(
            "No available encode/prefill/decode worker set for runtime {:?}",
            target_runtime
        );
        return None;
    }

    let encode_policy = policy_registry.get_encode_policy();
    let prefill_policy = policy_registry.get_prefill_policy();
    let decode_policy = policy_registry.get_decode_policy();
    let hash_ring = worker_registry.get_hash_ring(model_id);
    let mut info = SelectWorkerInfo {
        request_text: text,
        tokens,
        headers,
        hash_ring: hash_ring.clone(),
        response_token_count: Some(response_token_count),
        priority_groups: None,
        leg: WorkerLeg::Prefill,
    };
    let prefill_idx = policy_registry.select_worker(&prefill_policy, &available_prefill, &info)?;
    info.leg = WorkerLeg::Decode;
    let decode_idx = policy_registry.select_worker(&decode_policy, &available_decode, &info)?;

    let encode_assignments = assign_encode_workers(
        &available_encode,
        encode_item_hashes,
        model_id,
        encode_policy.as_ref(),
        hash_ring,
    )?;

    Metrics::record_worker_selection(
        metrics_labels::WORKER_PREFILL,
        metrics_labels::CONNECTION_GRPC,
        model_id,
        prefill_policy.name(),
    );
    Metrics::record_worker_selection(
        metrics_labels::WORKER_DECODE,
        metrics_labels::CONNECTION_GRPC,
        model_id,
        decode_policy.name(),
    );

    Some((
        encode_assignments,
        Arc::clone(&available_prefill[prefill_idx]),
        Arc::clone(&available_decode[decode_idx]),
        target_runtime,
    ))
}

fn encode_item_hashes(prep: &PreparationOutput) -> anyhow::Result<Vec<Vec<u8>>> {
    let intermediate = match prep {
        PreparationOutput::Chat {
            processed_messages, ..
        }
        | PreparationOutput::Messages {
            processed_messages, ..
        } => processed_messages.multimodal_intermediate.as_ref(),
        _ => None,
    };
    let Some(MultimodalIntermediate::Precomputed(precomputed)) = intermediate else {
        return Ok(Vec::new());
    };
    multimodal::precomputed_encode_routing_hashes(precomputed)
}

fn assign_encode_workers(
    encode_workers: &[Arc<dyn Worker>],
    item_hashes: &[Vec<u8>],
    model_id: &str,
    policy: &dyn LoadBalancingPolicy,
    hash_ring: Option<Arc<HashRing>>,
) -> Option<Vec<EncodeWorkerAssignment>> {
    if item_hashes.is_empty() {
        return Some(Vec::new());
    }

    item_hashes
        .iter()
        .enumerate()
        .map(|(item_index, content_hash)| {
            let routing_headers = encode_routing_headers(content_hash);
            let info = SelectWorkerInfo {
                request_text: None,
                tokens: None,
                headers: Some(&routing_headers),
                hash_ring: hash_ring.clone(),
                response_token_count: None,
                priority_groups: None,
                leg: WorkerLeg::Single,
            };
            let worker_idx = policy.select_worker(encode_workers, &info)?;
            let worker = Arc::clone(&encode_workers[worker_idx]);
            Metrics::record_worker_selection(
                metrics_labels::WORKER_ENCODE,
                metrics_labels::CONNECTION_GRPC,
                model_id,
                policy.name(),
            );
            Some(EncodeWorkerAssignment { item_index, worker })
        })
        .collect()
}

fn encode_routing_headers(content_hash: &[u8]) -> HeaderMap {
    let mut headers = HeaderMap::new();
    let key = hex_encode(content_hash);
    if let Ok(value) = HeaderValue::from_str(&key) {
        headers.insert("x-smg-routing-key", value);
    }
    headers
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}
