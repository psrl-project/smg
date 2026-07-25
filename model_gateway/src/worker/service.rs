//! Worker Service - Business logic layer for worker operations
//!
//! This module provides a clean separation between HTTP concerns (in routers)
//! and business logic for worker management. The service orchestrates
//! WorkerRegistry and JobQueue operations.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use openai_protocol::worker::{WorkerErrorResponse, WorkerInfo, WorkerSpec, WorkerUpdateRequest};
use serde_json::json;
use tracing::warn;

use crate::{
    config::RouterConfig,
    policies::PolicyRegistry,
    routers::grpc::routing_loop::runtime::InstanceVersionMap,
    worker::{
        registry::WorkerId, worker::worker_to_info, EngineStatsUpdateOutcome, Worker,
        WorkerRegistry, WorkerRoutingControlRequest, WorkerRoutingControlResult,
        WorkerRoutingControlResultItem, WorkerRoutingControlTargetRequest,
        WorkerStatsUpdateRequest, WorkerStatsUpdateResult, WorkerStatsUpdateResultItem,
        WorkerWeightVersionUpdateRequest, WorkerWeightVersionUpdateResult,
        WorkerWeightVersionUpdateResultItem,
    },
    workflow::{Job, JobQueue},
};

type ReplicaWeightVersionUpdate = (WorkerId, Arc<dyn Worker>, usize, u64);

/// Error types for worker service operations
#[derive(Debug)]
pub enum WorkerServiceError {
    /// Worker with given ID was not found
    NotFound { worker_id: String },
    /// Invalid worker ID format.
    InvalidId { raw: String, message: String },
    /// Bad request (e.g., URL mismatch in PUT)
    BadRequest { message: String },
    /// Worker with this URL already exists (duplicate POST)
    Conflict { url: String, worker_id: WorkerId },
    /// Job queue not initialized
    QueueNotInitialized,
    /// Failed to submit job to queue
    QueueSubmitFailed { message: String },
}

impl WorkerServiceError {
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::NotFound { .. } => "WORKER_NOT_FOUND",
            Self::InvalidId { .. } => "BAD_REQUEST",
            Self::BadRequest { .. } => "BAD_REQUEST",
            Self::Conflict { .. } => "WORKER_ALREADY_EXISTS",
            Self::QueueNotInitialized => "INTERNAL_SERVER_ERROR",
            Self::QueueSubmitFailed { .. } => "INTERNAL_SERVER_ERROR",
        }
    }

    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::NotFound { .. } => StatusCode::NOT_FOUND,
            Self::InvalidId { .. } => StatusCode::BAD_REQUEST,
            Self::BadRequest { .. } => StatusCode::BAD_REQUEST,
            Self::Conflict { .. } => StatusCode::CONFLICT,
            Self::QueueNotInitialized => StatusCode::INTERNAL_SERVER_ERROR,
            Self::QueueSubmitFailed { .. } => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl std::fmt::Display for WorkerServiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound { worker_id } => write!(f, "Worker {worker_id} not found"),
            Self::InvalidId { raw, message } => {
                write!(f, "Invalid worker_id '{raw}'. Error: {message}")
            }
            Self::BadRequest { message } => write!(f, "{message}"),
            Self::Conflict { url, worker_id } => {
                let id = worker_id.as_str();
                write!(
                    f,
                    "Worker already exists at URL '{url}' with ID {id}. \
                    Use PUT /workers/{id} to replace or PATCH /workers/{id} to update."
                )
            }
            Self::QueueNotInitialized => write!(f, "Job queue not initialized"),
            Self::QueueSubmitFailed { message } => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for WorkerServiceError {}

impl IntoResponse for WorkerServiceError {
    fn into_response(self) -> Response {
        let error = WorkerErrorResponse {
            error: self.to_string(),
            code: self.error_code().to_string(),
        };
        (self.status_code(), Json(error)).into_response()
    }
}

/// Result of creating a worker (async job submission)
#[derive(Debug)]
pub struct CreateWorkerResult {
    pub worker_id: WorkerId,
    pub url: String,
    pub location: String,
}

impl IntoResponse for CreateWorkerResult {
    fn into_response(self) -> Response {
        let response = json!({
            "status": "accepted",
            "worker_id": self.worker_id.as_str(),
            "url": self.url,
            "location": self.location,
            "message": "Worker addition queued for background processing"
        });
        (
            StatusCode::ACCEPTED,
            [(http::header::LOCATION, self.location)],
            Json(response),
        )
            .into_response()
    }
}

/// Result of deleting a worker (async job submission)
#[derive(Debug)]
pub struct DeleteWorkerResult {
    pub worker_id: WorkerId,
    pub url: String,
}

impl IntoResponse for DeleteWorkerResult {
    fn into_response(self) -> Response {
        let response = json!({
            "status": "accepted",
            "worker_id": self.worker_id.as_str(),
            "message": "Worker removal queued for background processing"
        });
        (StatusCode::ACCEPTED, Json(response)).into_response()
    }
}

/// Result of updating a worker (async job submission)
#[derive(Debug)]
pub struct UpdateWorkerResult {
    pub worker_id: WorkerId,
    pub url: String,
}

impl IntoResponse for UpdateWorkerResult {
    fn into_response(self) -> Response {
        let response = json!({
            "status": "accepted",
            "worker_id": self.worker_id.as_str(),
            "message": "Worker update queued for background processing"
        });
        (StatusCode::ACCEPTED, Json(response)).into_response()
    }
}

/// Result of listing workers
#[derive(Debug)]
pub struct ListWorkersResult {
    pub workers: Vec<WorkerInfo>,
    pub total: usize,
    pub prefill_count: usize,
    pub decode_count: usize,
    pub regular_count: usize,
}

impl IntoResponse for ListWorkersResult {
    fn into_response(self) -> Response {
        let response = json!({
            "workers": self.workers,
            "total": self.total,
            "stats": {
                "prefill_count": self.prefill_count,
                "decode_count": self.decode_count,
                "regular_count": self.regular_count,
            }
        });
        Json(response).into_response()
    }
}

/// Wrapper for WorkerInfo to implement IntoResponse
pub struct GetWorkerResponse(pub WorkerInfo);

impl IntoResponse for GetWorkerResponse {
    fn into_response(self) -> Response {
        Json(self.0).into_response()
    }
}

/// Type alias for a list of resolved (WorkerId, Worker) pairs.
type ResolvedWorkers = Vec<(WorkerId, Arc<dyn Worker>)>;

/// Worker Service - Orchestrates worker business logic
///
/// This service provides a clean API for worker operations, separating
/// business logic from HTTP concerns. Handlers in server.rs become thin
/// wrappers that translate between HTTP and this service.
pub struct WorkerService {
    worker_registry: Arc<WorkerRegistry>,
    job_queue: Arc<std::sync::OnceLock<Arc<JobQueue>>>,
    router_config: RouterConfig,
    instance_to_version_after_sync: InstanceVersionMap,
    /// Policy registry, used to notify throughput-optimal policies
    /// when fresh engine stats arrive.
    policy_registry: Option<Arc<PolicyRegistry>>,
}

impl WorkerService {
    /// Create a new WorkerService
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        job_queue: Arc<std::sync::OnceLock<Arc<JobQueue>>>,
        router_config: RouterConfig,
        instance_to_version_after_sync: InstanceVersionMap,
    ) -> Self {
        Self {
            worker_registry,
            job_queue,
            router_config,
            instance_to_version_after_sync,
            policy_registry: None,
        }
    }

    pub fn instance_to_version_map(&self) -> &InstanceVersionMap {
        &self.instance_to_version_after_sync
    }

    /// Remove version-map entries for a set of workers.
    ///
    /// Called after workers are deregistered from the registry so that stale
    /// `(base_worker_id, dp_rank)` → version entries do not accumulate
    /// indefinitely in `instance_to_version_after_sync`.
    pub fn remove_version_entries(&self, workers: &[Arc<dyn Worker>]) {
        for worker in workers {
            let base_worker_id = self
                .worker_registry
                .reserve_id_for_url(worker.base_url())
                .as_str()
                .to_string();
            let key = (base_worker_id, worker.dp_rank().unwrap_or(0));
            self.instance_to_version_after_sync.remove(&key);
        }
    }

    /// Attach a policy registry so the service can notify throughput-optimal
    /// policies when engine stats are applied.
    pub fn with_policy_registry(mut self, policy_registry: Arc<PolicyRegistry>) -> Self {
        self.policy_registry = Some(policy_registry);
        self
    }

    /// Parse and validate a worker ID string.
    pub fn parse_worker_id(raw: &str) -> Result<WorkerId, WorkerServiceError> {
        if raw.trim().is_empty() {
            return Err(WorkerServiceError::InvalidId {
                raw: raw.to_string(),
                message: "worker_id cannot be empty".to_string(),
            });
        }
        Ok(WorkerId::from_string(raw.to_string()))
    }

    /// Get the job queue, returning an error if not initialized
    fn get_job_queue(&self) -> Result<&Arc<JobQueue>, WorkerServiceError> {
        self.job_queue
            .get()
            .ok_or(WorkerServiceError::QueueNotInitialized)
    }

    pub async fn create_worker(
        &self,
        config: WorkerSpec,
    ) -> Result<CreateWorkerResult, WorkerServiceError> {
        if self.router_config.api_key.is_some() && config.api_key.is_none() {
            warn!(
                "Adding worker {} without API key while router has API key configured. \
                Worker will be accessible without authentication. \
                If the worker requires the same API key as the router, please specify it explicitly.",
                config.url
            );
        }

        let worker_url = config.url.clone();

        // Reserve (or retrieve) a stable ID for the 202 response.
        // If this URL already has an active worker, reject with 409.
        let worker_id = if let Some(ref manual_id) = config.id {
            let manual_worker_id = Self::parse_worker_id(manual_id)?;
            if let Some(existing_url) = self.worker_registry.get_url_by_id(&manual_worker_id) {
                if existing_url != worker_url {
                    return Err(WorkerServiceError::Conflict {
                        url: existing_url,
                        worker_id: manual_worker_id,
                    });
                }
            }

            let reserved_id = self
                .worker_registry
                .reserve_id_for_url_as(&worker_url, manual_worker_id.clone());
            if reserved_id != manual_worker_id {
                return Err(WorkerServiceError::Conflict {
                    url: worker_url,
                    worker_id: reserved_id,
                });
            }
            manual_worker_id
        } else {
            self.worker_registry.reserve_id_for_url(&worker_url)
        };
        if self.worker_registry.get(&worker_id).is_some() {
            return Err(WorkerServiceError::Conflict {
                url: worker_url,
                worker_id,
            });
        }

        let job = Job::AddWorker {
            config: Box::new(config),
        };

        self.get_job_queue()?
            .submit(job)
            .await
            .map_err(|e| WorkerServiceError::QueueSubmitFailed { message: e })?;

        let location = format!("/workers/{}", worker_id.as_str());

        Ok(CreateWorkerResult {
            worker_id,
            url: worker_url,
            location,
        })
    }

    /// Replace a worker by ID (full replace, re-runs registration workflow)
    pub async fn replace_worker(
        &self,
        worker_id_raw: &str,
        config: WorkerSpec,
    ) -> Result<UpdateWorkerResult, WorkerServiceError> {
        let worker_id = Self::parse_worker_id(worker_id_raw)?;

        let url = self
            .worker_registry
            .get_url_by_id(&worker_id)
            .ok_or_else(|| WorkerServiceError::NotFound {
                worker_id: worker_id_raw.to_string(),
            })?;

        // Validate that the URL in the request body matches the existing worker.
        // URL changes are not supported via replace — use DELETE + POST instead.
        if config.url != url {
            return Err(WorkerServiceError::BadRequest {
                message: format!(
                    "URL mismatch: worker has URL '{url}' but request body has '{}'. \
                    URL changes are not supported via PUT. Use DELETE + POST instead.",
                    config.url
                ),
            });
        }

        // Re-run the full registration workflow (model discovery, etc.)
        // The workflow uses register_or_replace() which does overwrite-then-diff.
        let job = Job::AddWorker {
            config: Box::new(config),
        };

        let job_queue = self.get_job_queue()?;
        job_queue
            .submit(job)
            .await
            .map_err(|e| WorkerServiceError::QueueSubmitFailed { message: e })?;

        Ok(UpdateWorkerResult { worker_id, url })
    }

    /// List all workers with their info
    pub fn list_workers(&self) -> ListWorkersResult {
        let workers = self.worker_registry.get_all_with_ids();
        let worker_infos: Vec<WorkerInfo> = workers
            .iter()
            .map(|(worker_id, worker)| {
                let mut info = worker_to_info(worker);
                info.id = worker_id.as_str().to_string();
                info
            })
            .collect();

        let stats = self.worker_registry.stats();

        ListWorkersResult {
            workers: worker_infos,
            total: stats.total_workers,
            prefill_count: stats.prefill_workers,
            decode_count: stats.decode_workers,
            regular_count: stats.regular_workers,
        }
    }

    pub fn get_worker(&self, worker_id_raw: &str) -> Result<GetWorkerResponse, WorkerServiceError> {
        let worker_id = Self::parse_worker_id(worker_id_raw)?;
        let job_queue = self.get_job_queue()?;

        if let Some(worker) = self.worker_registry.get(&worker_id) {
            let worker_url = worker.url().to_string();
            let mut worker_info = worker_to_info(&worker);
            worker_info.id = worker_id.as_str().to_string();
            if let Some(status) = job_queue.get_status(&worker_url) {
                worker_info.job_status = Some(status);
            }
            return Ok(GetWorkerResponse(worker_info));
        }

        if let Some(worker_url) = self.worker_registry.get_url_by_id(&worker_id) {
            if let Some(status) = job_queue.get_status(&worker_url) {
                return Ok(GetWorkerResponse(WorkerInfo::pending(
                    worker_id.as_str(),
                    worker_url,
                    Some(status),
                )));
            }
        }

        Err(WorkerServiceError::NotFound {
            worker_id: worker_id_raw.to_string(),
        })
    }

    /// Delete a worker by ID (submits async job)
    pub async fn delete_worker(
        &self,
        worker_id_raw: &str,
    ) -> Result<DeleteWorkerResult, WorkerServiceError> {
        let worker_id = Self::parse_worker_id(worker_id_raw)?;

        let url = self
            .worker_registry
            .get_url_by_id(&worker_id)
            .ok_or_else(|| WorkerServiceError::NotFound {
                worker_id: worker_id_raw.to_string(),
            })?;

        let job = Job::RemoveWorker {
            url: url.clone(),
            expected_revision: None,
        };

        let job_queue = self.get_job_queue()?;
        job_queue
            .submit(job)
            .await
            .map_err(|e| WorkerServiceError::QueueSubmitFailed { message: e })?;

        Ok(DeleteWorkerResult { worker_id, url })
    }

    /// Update a worker by ID (submits async job)
    pub async fn update_worker(
        &self,
        worker_id_raw: &str,
        update: WorkerUpdateRequest,
    ) -> Result<UpdateWorkerResult, WorkerServiceError> {
        let worker_id = Self::parse_worker_id(worker_id_raw)?;

        let url = self
            .worker_registry
            .get_url_by_id(&worker_id)
            .ok_or_else(|| WorkerServiceError::NotFound {
                worker_id: worker_id_raw.to_string(),
            })?;

        let job = Job::UpdateWorker {
            url: url.clone(),
            update: Box::new(update),
        };

        let job_queue = self.get_job_queue()?;
        job_queue
            .submit(job)
            .await
            .map_err(|e| WorkerServiceError::QueueSubmitFailed { message: e })?;

        Ok(UpdateWorkerResult { worker_id, url })
    }

    pub fn update_worker_stats(&self, update: WorkerStatsUpdateRequest) -> WorkerStatsUpdateResult {
        let total = update.len();
        let mut results = Vec::with_capacity(total);
        let staleness_threshold_ms = self.router_config.engine_stats_staleness_threshold_ms;

        // Process each update
        for item in update {
            match self.resolve_worker_by_id_and_dp(&item.worker_id, item.dp_rank) {
                Ok((worker_id, worker)) => {
                    let url = worker.url().to_string();
                    let dp_rank = item.dp_rank;
                    let outcome = worker.update_engine_stats(item.stats, staleness_threshold_ms);

                    // Agreement-gated rebase of the worker's inflight-token
                    // estimates. When a fresh snapshot's request count agrees
                    // with the locally-tracked load counter, the engine has
                    // caught up to every admitted request, so the snapshot's
                    // token totals already incorporate them — clear the local
                    // estimates to avoid double-counting. When they disagree
                    // (requests still in transit to the engine), keep the
                    // estimates so load-aware policies still see them. This
                    // mirrors psrl_agent's `is_staled` count-agreement gate and
                    // replaces the old unconditional optimistic-delta reset.
                    if matches!(outcome, EngineStatsUpdateOutcome::Applied)
                        && worker.engine_stats().waiting_and_running_queue_size() == worker.load()
                    {
                        worker.rebase_inflight_tokens();
                    }

                    results.push(Self::build_stats_update_result(
                        &worker_id, url, dp_rank, outcome,
                    ));
                }
                Err(err) => {
                    results.push(WorkerStatsUpdateResultItem {
                        status: "rejected".to_string(),
                        worker_id: item.worker_id,
                        url: String::new(),
                        applied: false,
                        dp_rank: item.dp_rank,
                        stale_reason: Some(err.to_string()),
                    });
                }
            }
        }

        // Count the results
        let updated = results.iter().filter(|r| r.status == "updated").count();
        let stale_ignored = results
            .iter()
            .filter(|r| r.status == "stale_ignored")
            .count();
        let rejected = results.iter().filter(|r| r.status == "rejected").count();

        WorkerStatsUpdateResult {
            total,
            updated,
            stale_ignored,
            rejected,
            results,
        }
    }

    pub fn update_worker_weight_version(
        &self,
        update: WorkerWeightVersionUpdateRequest,
    ) -> WorkerWeightVersionUpdateResult {
        let total = update.len();
        let mut results = Vec::with_capacity(total);
        let mut replica_updates: HashMap<String, Vec<ReplicaWeightVersionUpdate>> = HashMap::new();
        let mut replica_ranks: HashMap<String, HashSet<usize>> = HashMap::new();

        for worker in self.worker_registry.get_all() {
            if let Some(dp_rank) = worker.dp_rank() {
                let base_worker_id = self
                    .worker_registry
                    .reserve_id_for_url(worker.base_url())
                    .as_str()
                    .to_string();
                replica_ranks
                    .entry(base_worker_id)
                    .or_default()
                    .insert(dp_rank);
            }
        }

        for item in update {
            let weight_version = item.weight_version;
            match self.resolve_worker_by_id_and_dp(&item.worker_id, item.dp_rank) {
                Ok((worker_id, worker)) => {
                    let base_worker_id = self
                        .worker_registry
                        .reserve_id_for_url(worker.base_url())
                        .as_str()
                        .to_string();
                    replica_updates.entry(base_worker_id).or_default().push((
                        worker_id,
                        worker.clone(),
                        worker.dp_rank().unwrap_or(0),
                        weight_version,
                    ));
                }
                Err(err) => {
                    results.push(WorkerWeightVersionUpdateResultItem {
                        status: "rejected".to_string(),
                        worker_id: item.worker_id,
                        url: String::new(),
                        weight_version,
                        dp_rank: item.dp_rank,
                        reason: Some(err.to_string()),
                    });
                }
            }
        }

        for (base_worker_id, items) in replica_updates {
            let expected_ranks = replica_ranks.get(&base_worker_id);
            let requested_ranks: HashSet<usize> =
                items.iter().map(|(_, _, rank, _)| *rank).collect();
            let versions: HashSet<u64> = items.iter().map(|(_, _, _, version)| *version).collect();
            let all_supported = items
                .iter()
                .all(|(_, worker, _, _)| worker.supports_dyn_weight_version_update());
            let reject_reason = if expected_ranks.is_some_and(|ranks| requested_ranks != *ranks) {
                Some(format!(
                    "replica update must cover all DP ranks: expected {expected_ranks:?}, got {requested_ranks:?}"
                ))
            } else if requested_ranks.len() != items.len() {
                Some("replica update contains duplicate DP ranks".to_string())
            } else if versions.len() != 1 {
                Some(
                    "all DP ranks in a replica update must use the same weight version".to_string(),
                )
            } else if !all_supported {
                Some("worker does not support dynamic weight version updates".to_string())
            } else {
                None
            };

            if let Some(reason) = reject_reason {
                for (worker_id, worker, rank, weight_version) in items {
                    results.push(WorkerWeightVersionUpdateResultItem {
                        status: "rejected".to_string(),
                        worker_id: worker_id.as_str().to_string(),
                        url: worker.url().to_string(),
                        weight_version,
                        dp_rank: worker.dp_rank().map(|_| rank),
                        reason: Some(reason.clone()),
                    });
                }
                continue;
            }

            for (worker_id, worker, rank, weight_version) in items {
                let updated = worker.update_dyn_weight_version(weight_version);
                debug_assert!(updated, "support check must make update infallible");
                // Model-version bump: the engine interrupts/clears its run+wait
                // queues, so the router's speculative occupancy is now stale.
                // Reset the load counter and drop inflight-token estimates,
                // otherwise the agreement-gated rebase (`waiting_and_running ==
                // load()`) can never re-fire and the KV estimate stays wedged.
                worker.reset_load();
                worker.rebase_inflight_tokens();
                // Drop the policy's stale per-instance speculative state
                // (cache-affinity / hypothetical-KV trees) for this instance.
                if let Some(registry) = self.policy_registry.as_ref() {
                    registry
                        .get_policy_or_default(worker.model_id())
                        .on_version_bump(worker.url());
                }
                if let Ok(version_tag) = i64::try_from(weight_version) {
                    self.instance_to_version_after_sync
                        .insert((base_worker_id.clone(), rank), version_tag);
                }
                results.push(WorkerWeightVersionUpdateResultItem {
                    status: "updated".to_string(),
                    worker_id: worker_id.as_str().to_string(),
                    url: worker.url().to_string(),
                    weight_version,
                    dp_rank: worker.dp_rank().map(|_| rank),
                    reason: None,
                });
            }
        }

        // Count the number of updated and rejected results
        let updated = results.iter().filter(|r| r.status == "updated").count();
        let rejected = results.iter().filter(|r| r.status == "rejected").count();

        WorkerWeightVersionUpdateResult {
            total,
            updated,
            rejected,
            results,
        }
    }

    pub fn pause_workers(&self, update: WorkerRoutingControlRequest) -> WorkerRoutingControlResult {
        self.apply_worker_routing_control(update, true)
    }

    pub fn resume_workers(
        &self,
        update: WorkerRoutingControlRequest,
    ) -> WorkerRoutingControlResult {
        self.apply_worker_routing_control(update, false)
    }

    fn apply_worker_routing_control(
        &self,
        update: WorkerRoutingControlRequest,
        paused: bool,
    ) -> WorkerRoutingControlResult {
        let mut results = Vec::new();
        let mut seen_worker_ids = HashSet::new();

        for target in update {
            let base_worker_id = target.base_worker_id.clone();
            let requested_ranks = target
                .dp_rank
                .as_ref()
                .and_then(|input| input.to_ranks().ok());

            match self.resolve_routing_control_target(&target) {
                Ok(workers) => {
                    for (worker_id, worker) in workers {
                        if !seen_worker_ids.insert(worker_id.as_str().to_string()) {
                            continue;
                        }

                        let updated = worker.set_paused(paused);
                        results.push(WorkerRoutingControlResultItem {
                            status: if updated { "updated" } else { "rejected" }.to_string(),
                            worker_id: worker_id.as_str().to_string(),
                            url: worker.url().to_string(),
                            paused: worker.is_paused(),
                            base_worker_id: base_worker_id.clone(),
                            dp_rank: worker.dp_rank().or_else(|| {
                                requested_ranks
                                    .as_ref()
                                    .and_then(|ranks| (ranks.len() == 1).then_some(ranks[0]))
                            }),
                            reason: (!updated)
                                .then(|| "worker does not support routing pause state".to_string()),
                        });
                    }
                }
                Err(err) => {
                    results.push(WorkerRoutingControlResultItem {
                        status: "rejected".to_string(),
                        worker_id: target
                            .worker_id
                            .or_else(|| target.base_worker_id.clone())
                            .unwrap_or_default(),
                        url: String::new(),
                        paused,
                        base_worker_id: target.base_worker_id,
                        dp_rank: requested_ranks
                            .as_ref()
                            .and_then(|ranks| (ranks.len() == 1).then_some(ranks[0])),
                        reason: Some(err.to_string()),
                    });
                }
            }
        }

        let updated = results.iter().filter(|r| r.status == "updated").count();
        let rejected = results.iter().filter(|r| r.status == "rejected").count();

        WorkerRoutingControlResult {
            action: if paused { "paused" } else { "resumed" }.to_string(),
            total: results.len(),
            updated,
            rejected,
            results,
        }
    }

    fn resolve_routing_control_target(
        &self,
        target: &WorkerRoutingControlTargetRequest,
    ) -> Result<ResolvedWorkers, WorkerServiceError> {
        match (
            target.worker_id.as_deref(),
            target.base_worker_id.as_deref(),
            target.dp_rank.as_ref(),
        ) {
            (Some(worker_id_raw), None, None) => {
                let worker_id = Self::parse_worker_id(worker_id_raw)?;
                let worker = self.worker_registry.get(&worker_id).ok_or_else(|| {
                    WorkerServiceError::NotFound {
                        worker_id: worker_id_raw.to_string(),
                    }
                })?;
                Ok(vec![(worker_id, worker)])
            }
            (Some(base_worker_id_raw), None, Some(dp_rank_input)) => {
                let ranks = dp_rank_input
                    .to_ranks()
                    .map_err(|message| WorkerServiceError::BadRequest { message })?;
                ranks
                    .into_iter()
                    .map(|rank| self.resolve_worker_by_id_and_dp(base_worker_id_raw, Some(rank)))
                    .collect()
            }
            (None, Some(base_worker_id_raw), None) => {
                let base_worker_id = Self::parse_worker_id(base_worker_id_raw)?;
                let base_url = self
                    .worker_registry
                    .get_url_by_id(&base_worker_id)
                    .ok_or_else(|| WorkerServiceError::NotFound {
                        worker_id: base_worker_id_raw.to_string(),
                    })?;
                let prefix = format!("{base_url}@");

                let mut workers: Vec<_> = self
                    .worker_registry
                    .get_all_with_ids()
                    .into_iter()
                    .filter(|(_, worker)| worker.url().starts_with(&prefix))
                    .collect();

                if workers.is_empty() {
                    if let Some(worker) = self.worker_registry.get_by_url(&base_url) {
                        workers.push((base_worker_id, worker));
                    }
                }

                if workers.is_empty() {
                    return Err(WorkerServiceError::NotFound {
                        worker_id: base_worker_id_raw.to_string(),
                    });
                }

                Ok(workers)
            }
            (None, Some(base_worker_id_raw), Some(dp_rank_input)) => {
                let ranks = dp_rank_input
                    .to_ranks()
                    .map_err(|message| WorkerServiceError::BadRequest { message })?;
                ranks
                    .into_iter()
                    .map(|rank| self.resolve_worker_by_id_and_dp(base_worker_id_raw, Some(rank)))
                    .collect()
            }
            _ => Err(WorkerServiceError::BadRequest {
                message: "target must specify one mode: worker_id, worker_id with dp_rank, \
                     base_worker_id, or base_worker_id with dp_rank"
                    .to_string(),
            }),
        }
    }

    fn resolve_worker_by_id_and_dp(
        &self,
        worker_id_raw: &str,
        dp_rank: Option<usize>,
    ) -> Result<(WorkerId, Arc<dyn Worker>), WorkerServiceError> {
        let worker_id = Self::parse_worker_id(worker_id_raw)?;
        let Some(dp_rank) = dp_rank else {
            let worker = self.worker_registry.get(&worker_id).ok_or_else(|| {
                WorkerServiceError::NotFound {
                    worker_id: worker_id_raw.to_string(),
                }
            })?;
            if worker.is_dp_aware() {
                return Err(WorkerServiceError::BadRequest {
                    message: format!(
                        "worker_id '{worker_id_raw}' points to a DP worker; include the base worker_id and dp_rank instead"
                    ),
                });
            }
            return Ok((worker_id, worker));
        };

        if self
            .worker_registry
            .get(&worker_id)
            .is_some_and(|worker| worker.is_dp_aware())
        {
            return Err(WorkerServiceError::BadRequest {
                message: format!(
                    "worker_id '{worker_id_raw}' points to a DP worker; use the base worker_id with dp_rank"
                ),
            });
        }

        let base_url = self
            .worker_registry
            .get_url_by_id(&worker_id)
            .ok_or_else(|| WorkerServiceError::NotFound {
                worker_id: worker_id_raw.to_string(),
            })?;
        let target_url = format!("{base_url}@{dp_rank}");
        let target_id = self
            .worker_registry
            .get_id_by_url(&target_url)
            .ok_or_else(|| WorkerServiceError::NotFound {
                worker_id: format!("{worker_id_raw}@{dp_rank}"),
            })?;
        let worker =
            self.worker_registry
                .get(&target_id)
                .ok_or_else(|| WorkerServiceError::NotFound {
                    worker_id: target_id.as_str().to_string(),
                })?;
        if worker.dp_rank() != Some(dp_rank) {
            return Err(WorkerServiceError::BadRequest {
                message: format!("worker at URL '{}' is not DP rank {dp_rank}", worker.url()),
            });
        }

        Ok((target_id, worker))
    }

    fn build_stats_update_result(
        worker_id: &WorkerId,
        url: String,
        dp_rank: Option<usize>,
        outcome: EngineStatsUpdateOutcome,
    ) -> WorkerStatsUpdateResultItem {
        match outcome {
            EngineStatsUpdateOutcome::Applied => WorkerStatsUpdateResultItem {
                status: "updated".to_string(),
                worker_id: worker_id.as_str().to_string(),
                url,
                applied: true,
                dp_rank,
                stale_reason: None,
            },
            EngineStatsUpdateOutcome::Stale { reason } => WorkerStatsUpdateResultItem {
                status: "stale_ignored".to_string(),
                worker_id: worker_id.as_str().to_string(),
                url,
                applied: false,
                dp_rank,
                stale_reason: Some(reason),
            },
            EngineStatsUpdateOutcome::Rejected { reason } => WorkerStatsUpdateResultItem {
                status: "rejected".to_string(),
                worker_id: worker_id.as_str().to_string(),
                url,
                applied: false,
                dp_rank,
                stale_reason: Some(reason),
            },
        }
    }
}
