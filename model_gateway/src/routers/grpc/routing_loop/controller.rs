//! Control-plane handlers for the request routing loop.

use std::sync::Arc;

use axum::{
    extract::{Query, State},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};

use super::runtime::{RequestMigrationTarget, RoutingLoopRuntime};
use crate::{routers::error, server::AppState};

fn runtime_from_state(state: &AppState) -> Result<Arc<RoutingLoopRuntime>, Box<Response>> {
    state.context.routing_loop_runtime.clone().ok_or_else(|| {
        Box::new(error::not_found(
            "routing_loop_disabled",
            "Routing loop is not enabled",
        ))
    })
}

// ── pause ─────────────────────────────────────────────────────────────────────

/// Query params for `POST /routing_loop/pause`.
#[derive(Deserialize)]
pub(crate) struct PauseParams {
    /// When `true`, wait until all admitted decisions and commit/dispatch
    /// handoffs have exited before returning.
    ///
    /// The response is the pause acknowledgement used by the sync coordinator.
    wait: Option<bool>,
}

pub(crate) async fn pause_routing_loop(
    State(state): State<Arc<AppState>>,
    Query(params): Query<PauseParams>,
) -> Response {
    match runtime_from_state(&state) {
        Ok(runtime) => {
            runtime.pause();

            if params.wait.unwrap_or(false) {
                runtime.wait_for_pause_barrier().await;
            }

            Json(runtime.status().await).into_response()
        }
        Err(response) => *response,
    }
}

pub(crate) async fn resume_routing_loop(State(state): State<Arc<AppState>>) -> Response {
    match runtime_from_state(&state) {
        Ok(runtime) => {
            runtime.resume();
            Json(runtime.status().await).into_response()
        }
        Err(response) => *response,
    }
}

pub(crate) async fn routing_loop_status(State(state): State<Arc<AppState>>) -> Response {
    match runtime_from_state(&state) {
        Ok(runtime) => Json(runtime.status().await).into_response(),
        Err(response) => *response,
    }
}

pub(crate) async fn routing_loop_snapshot(State(state): State<Arc<AppState>>) -> Response {
    match runtime_from_state(&state) {
        Ok(runtime) => Json(runtime.queued_request_snapshot().await).into_response(),
        Err(response) => *response,
    }
}

#[derive(Deserialize)]
pub(crate) struct MigrationQuery {
    request_id: String,
}

pub(crate) async fn prepare_request_migration(
    State(state): State<Arc<AppState>>, Json(target): Json<RequestMigrationTarget>,
) -> Response {
    match runtime_from_state(&state) {
        Ok(runtime) => {
            runtime.prepare_migration(target);
            Json(serde_json::json!({"prepared": true})).into_response()
        }
        Err(response) => *response,
    }
}

pub(crate) async fn request_migration_status(
    State(state): State<Arc<AppState>>, Query(query): Query<MigrationQuery>,
) -> Response {
    match runtime_from_state(&state) {
        Ok(runtime) => Json(runtime.migration_status(&query.request_id)).into_response(),
        Err(response) => *response,
    }
}

pub(crate) async fn finish_request_migration(
    State(state): State<Arc<AppState>>, Json(query): Json<MigrationQuery>,
) -> Response {
    match runtime_from_state(&state) {
        Ok(runtime) => {
            runtime.finish_migration(&query.request_id);
            Json(serde_json::json!({"finished": true})).into_response()
        }
        Err(response) => *response,
    }
}

/// Query params for `GET /routing_loop/filter`.
#[derive(Deserialize)]
pub(crate) struct FilterParams {
    version_tag: i64,
}

/// `GET /routing_loop/filter?version_tag=N`
///
/// Returns the metadata of all queued requests whose `version_tag ≤ N`.
/// Used by the Python coordinator's `_check_should_sync` first step to
/// determine whether all old-version requests have drained from the queue.
pub(crate) async fn routing_loop_filter(
    State(state): State<Arc<AppState>>,
    Query(params): Query<FilterParams>,
) -> Response {
    match runtime_from_state(&state) {
        Ok(runtime) => {
            let filtered = runtime
                .filter_queue_by_version_tag(params.version_tag)
                .await;
            Json(filtered).into_response()
        }
        Err(response) => *response,
    }
}

/// Per-worker stats returned by `GET /workers/stats`.
#[derive(Serialize)]
pub(crate) struct WorkerStats {
    base_worker_id: String,
    dp_rank: usize,
    version_tag: i64,
    /// Number of prompt groups currently pinned to this worker instance.
    running_requests: usize,
}

/// `GET /workers/stats`
///
/// Returns an array of per-worker statistics: synced version tag and the count
/// of active prompt groups routed to each instance.  Used by the Python
/// coordinator for migration decisions.
pub(crate) async fn workers_stats(State(state): State<Arc<AppState>>) -> Response {
    match runtime_from_state(&state) {
        Ok(runtime) => {
            let stats: Vec<WorkerStats> = runtime
                .instance_to_version_after_sync
                .iter()
                .map(|entry| {
                    let (base_worker_id, dp_rank) = entry.key().clone();
                    let version_tag = *entry.value();
                    // Count prompt groups whose pinned instance equals this worker.
                    let running_requests = runtime
                        .prompt_to_pinned_instance
                        .iter()
                        .filter(|e| *e.value() == (base_worker_id.clone(), dp_rank))
                        .count();
                    WorkerStats {
                        base_worker_id,
                        dp_rank,
                        version_tag,
                        running_requests,
                    }
                })
                .collect();
            Json(stats).into_response()
        }
        Err(response) => *response,
    }
}
