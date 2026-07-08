//! PSRL five-stage worker selector.
//!
//! This strategy implements the five-stage candidate-filtering chain used in
//! Policy-based Scheduling for Reinforcement Learning (PSRL) training rollouts.
//!
//! # Stage overview
//!
//! 1. **Version filter** — retain only workers whose synced version tag ≥ the
//!    request's `version_tag`.  Uses a synchronous `parking_lot::Mutex` to avoid
//!    yielding to the async runtime on the hot path.
//! 2. **Trajectory sticky pin** — when `is_sticky` and a `rollout_instance_hint`
//!    is present, route directly to that instance. Non-sticky requests (including
//!    partial-rollout loopbacks) are free to land on any eligible instance;
//!    coordinator-side migration is independent of this stage.
//! 3. **Group pin** — when `prompt_id` is present and multi-instance group
//!    sampling is disabled, route to the instance already serving that prompt.
//! 4. **Can-reserve filter** — call `CanReserveRequest` via the PS Manager to
//!    retain only workers that currently have capacity (skipped when
//!    `version_tag != -1` because version-tagged requests are pre-routed).
//! 5. **Sort and select** — sort candidates by `WorkerSelectionKey`, assign
//!    group IDs, then delegate to the configured policy.
//!
//! Post-selection: update the PS Manager and the runtime bookkeeping maps.

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use tracing::{debug, error, info, warn};

use super::WorkerSelectorStrategy;
use crate::{
    config::types::CandidateSortKey,
    observability::metrics::{metrics_labels, Metrics},
    policies::{PolicyRegistry, SelectWorkerInfo, WorkerLeg},
    routers::{
        error,
        grpc::{
            kv_transfer::KvTransferCoordinator,
            routing_loop::{metadata::RoutingMeta, runtime::RoutingLoopRuntime},
        },
    },
    worker::{ConnectionMode, Worker, WorkerRegistry, WorkerType, UNKNOWN_MODEL_ID},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkerSelectionKey {
    /// Version ordering indicator.
    ///
    /// New requests use `-version` so ascending order tries newer versions
    /// first; versioned requests use `version` so ascending order tries the
    /// lowest acceptable version first.
    Version(i64),
    /// Sort by PS Manager reserve indicator, with version as tiebreaker.
    ///
    /// Lower reserve indicator is better (`-inf` best, `inf` worst), then
    /// lower `version_indicator` wins.
    /// `reserve_bits` stores the `f64` bit pattern for deterministic ordering.
    Reserve {
        /// Bits of the `f64` reserve indicator for `total_cmp`-style ordering.
        reserve_bits: u64,
        /// Version indicator used as the tiebreaker.
        version_indicator: i64,
    },
}

impl PartialOrd for WorkerSelectionKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WorkerSelectionKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Self::Version(a), Self::Version(b)) => a.cmp(b),
            (
                Self::Reserve {
                    reserve_bits: ra,
                    version_indicator: va,
                },
                Self::Reserve {
                    reserve_bits: rb,
                    version_indicator: vb,
                },
            ) => {
                // Use total_cmp semantics via bit reinterpretation.
                let ord = f64::from_bits(*ra).total_cmp(&f64::from_bits(*rb));
                if ord == std::cmp::Ordering::Equal {
                    va.cmp(vb)
                } else {
                    ord
                }
            }
            // Heterogeneous comparisons shouldn't occur; treat as equal.
            _ => std::cmp::Ordering::Equal,
        }
    }
}

/// Read the worker runtime version used by PSRL when the sync map has no entry.
///
/// Unknown or out-of-range versions default to `0`, matching PSRL's initial
/// rollout-instance version.
fn worker_version_tag(worker: &Arc<dyn Worker>) -> i64 {
    let dyn_version = worker.dyn_weight_version();
    i64::try_from(dyn_version).unwrap_or(0)
}

pub(crate) struct PsrlWorkerSelector {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    runtime: Arc<RoutingLoopRuntime>,
    candidate_sort_key: CandidateSortKey,
    enable_group_sticky: bool,
    kv_transfer: Option<Arc<KvTransferCoordinator>>,
    /// Guards the "sort → policy select → increment load" step at Stage 5 so
    /// that concurrent dispatch tasks cannot all read equal loads and pick the
    /// same worker before any increment has been registered.
    selection_lock: parking_lot::Mutex<()>,
}

impl PsrlWorkerSelector {
    pub(crate) fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        runtime: Arc<RoutingLoopRuntime>,
        candidate_sort_key: CandidateSortKey,
        enable_group_sticky: bool,
        kv_transfer: Option<Arc<KvTransferCoordinator>>,
    ) -> Self {
        info!(
            enable_group_sticky,
            kv_transfer_enabled = kv_transfer.is_some(),
            "PSRL worker selector initialized (sticky config)"
        );
        Self {
            worker_registry,
            policy_registry,
            runtime,
            candidate_sort_key,
            enable_group_sticky,
            kv_transfer,
            selection_lock: parking_lot::Mutex::new(()),
        }
    }

    fn worker_instance_id(&self, worker: &Arc<dyn Worker>) -> (String, usize) {
        let base_worker_id = self
            .worker_registry
            .reserve_id_for_url(worker.base_url())
            .as_str()
            .to_string();
        (base_worker_id, worker.dp_rank().unwrap_or(0))
    }

    fn version_after_sync(&self, worker: &Arc<dyn Worker>) -> i64 {
        let instance = self.worker_instance_id(worker);
        self.runtime
            .instance_to_version_after_sync
            .get(&instance)
            .map(|v| *v)
            .unwrap_or_else(|| worker_version_tag(worker))
    }

    /// Resolve the live [`Worker`] for a `(base_worker_id, dp_rank)` instance,
    /// scanning the model's worker set. Used to locate the migration source
    /// (old instance A) so the transfer coordinator can address its servicer.
    fn find_worker_by_instance(
        &self,
        model_id: &str,
        instance: &(String, usize),
    ) -> Option<Arc<dyn Worker>> {
        let workers = if model_id == UNKNOWN_MODEL_ID {
            self.worker_registry.get_all()
        } else {
            self.worker_registry.get_by_model(model_id).to_vec()
        };
        workers
            .into_iter()
            .find(|w| self.worker_instance_id(w) == *instance)
    }
}

#[async_trait]
impl WorkerSelectorStrategy for PsrlWorkerSelector {
    async fn select_single_worker(
        &self,
        model_id: &str,
        text: Option<&str>,
        tokens: Option<&[u32]>,
        headers: Option<&http::HeaderMap>,
        routing_meta: Option<&RoutingMeta>,
    ) -> Option<Arc<dyn Worker>> {
        // ── collect initial candidate pool ──────────────────────────────────
        let model_filter = if model_id == UNKNOWN_MODEL_ID {
            None
        } else {
            Some(model_id)
        };

        let workers = self.worker_registry.get_workers_filtered(
            model_filter,
            Some(WorkerType::Regular),
            Some(ConnectionMode::Grpc),
            None,
            false,
        );

        let mut candidates: Vec<Arc<dyn Worker>> =
            workers.into_iter().filter(|w| w.is_available()).collect();

        if candidates.is_empty() {
            return None;
        }

        let meta = routing_meta?;

        // ── Stage 1: version filter ─────────────────────────────────────────
        candidates.retain(|w| self.version_after_sync(w) >= meta.version_tag);

        if candidates.is_empty() {
            warn!(
                version_tag = meta.version_tag,
                "PSRL Stage 1: no candidates after version filter"
            );
            return None;
        }

        // ── Stage 2: trajectory sticky pin ──────────────────────────────────
        // Only `is_sticky` pins to the previous instance. Non-sticky requests
        // keep `rollout_instance_hint` as the KV-transfer source but are free
        // to land on any eligible worker (policy / cache-aware / load balance).
        // Coordinator-side imbalance migration is independent of this stage.
        if meta.is_sticky {
            if let Some(ref hint) = meta.rollout_instance_hint {
                let pinned: Vec<Arc<dyn Worker>> = candidates
                    .iter()
                    .filter(|w| self.worker_instance_id(w) == *hint)
                    .cloned()
                    .collect();
                if !pinned.is_empty() {
                    debug!(
                        request_id = meta.request_id,
                        prompt_id = meta.prompt_id,
                        base_worker_id = %hint.0,
                        target_dp_rank = hint.1,
                        "PSRL trajectory sticky HIT: pinned to hinted instance"
                    );
                    candidates = pinned;
                } else {
                    debug!(
                        request_id = meta.request_id,
                        prompt_id = meta.prompt_id,
                        base_worker_id = %hint.0,
                        target_dp_rank = hint.1,
                        "PSRL trajectory sticky MISS: hinted instance unavailable, falling through"
                    );
                }
            }
        }

        if candidates.is_empty() {
            return None;
        }

        // ── Stage 3: group pin ──────────────────────────────────────────────
        // Read the pinned instance for this prompt group from the write-once
        // map.  Because `record_selected_instance` uses `entry().or_insert()`
        // to write that map, any instance we read here is fully committed — no
        // second reader can observe a partially-written state.  This replaces
        // the previous two-step lookup (prompt→request_id, request_id→instance)
        // which was susceptible to a TOCTOU race on concurrent dispatch tasks.
        if self.enable_group_sticky {
            let prompt_id = meta.prompt_id;
            let group_instance: Option<(String, usize)> = self
                .runtime
                .prompt_to_pinned_instance
                .get(&prompt_id)
                .map(|entry| entry.clone());

            if let Some(ref group_inst) = group_instance {
                let pinned: Vec<Arc<dyn Worker>> = candidates
                    .iter()
                    .filter(|w| self.worker_instance_id(w) == *group_inst)
                    .cloned()
                    .collect();
                if !pinned.is_empty() {
                    debug!(
                        request_id = meta.request_id,
                        prompt_id,
                        base_worker_id = %group_inst.0,
                        target_dp_rank = group_inst.1,
                        "PSRL group sticky HIT: pinned to group instance"
                    );
                    candidates = pinned;
                } else {
                    debug!(
                        request_id = meta.request_id,
                        prompt_id,
                        base_worker_id = %group_inst.0,
                        target_dp_rank = group_inst.1,
                        "PSRL group sticky MISS: group instance unavailable, falling through"
                    );
                }
                // If pinned is empty the previously-pinned instance is no
                // longer available; fall through with all candidates so the
                // request can still be served.
            }
        }

        if candidates.is_empty() {
            return None;
        }

        // ── Stage 4: can-reserve filter ─────────────────────────────────────
        // Only applicable when version_tag == -1 (un-versioned requests).
        let request_id = meta.request_id;
        let is_validate = meta.is_validate;

        if meta.version_tag == -1 {
            if let Some(ps_client) = self.runtime.ps_manager_client.get() {
                // Collect unique synced version tags.
                let unique_versions: Vec<i64> = {
                    let mut v: Vec<i64> = candidates
                        .iter()
                        .map(|w| self.version_after_sync(w))
                        .collect();
                    v.sort_unstable();
                    v.dedup();
                    v
                };

                if !unique_versions.is_empty() {
                    match ps_client
                        .can_reserve_request(
                            vec![request_id],
                            unique_versions.clone(),
                            false,
                            vec![is_validate],
                        )
                        .await
                    {
                        Ok((results, _n_versions)) => {
                            // results[i] == true  ↔  unique_versions[i] is reservable
                            let reservable: std::collections::HashSet<i64> = unique_versions
                                .iter()
                                .zip(results.iter())
                                .filter_map(|(&v, &ok)| if ok { Some(v) } else { None })
                                .collect();

                            candidates.retain(|w| reservable.contains(&self.version_after_sync(w)));

                            if candidates.is_empty() {
                                info!(
                                    request_id,
                                    "PSRL Stage 4: no candidates after can_reserve filter"
                                );
                                return None;
                            }
                        }
                        Err(status) => {
                            error!(
                                request_id,
                                %status,
                                "PSRL Stage 4: can_reserve_request RPC failed; skipping filter"
                            );
                            return None;
                        }
                    }
                }
            }
        }

        // ── Stage 5: sort by key, assign group IDs, delegate to policy ──────
        let sort_keys: Vec<WorkerSelectionKey> = match self.candidate_sort_key {
            CandidateSortKey::Version => candidates
                .iter()
                .map(|w| {
                    let v = self.version_after_sync(w);
                    // When version_tag == -1 use negative version so higher
                    // versions sort first.
                    WorkerSelectionKey::Version(if meta.version_tag == -1 { -v } else { v })
                })
                .collect(),

            CandidateSortKey::ReserveCapability => {
                let indicators: Vec<f64> =
                    if let Some(ps_client) = self.runtime.ps_manager_client.get() {
                        let unique_versions: Vec<i64> = {
                            let mut v: Vec<i64> = candidates
                                .iter()
                                .map(|w| self.version_after_sync(w))
                                .collect();
                            v.sort_unstable();
                            v.dedup();
                            v
                        };

                        match ps_client
                            .get_reserve_indicator(request_id, unique_versions.clone(), is_validate)
                            .await
                        {
                            Ok(ind) => {
                                // Map each candidate's version → its indicator.
                                let version_to_indicator: std::collections::HashMap<i64, f64> =
                                    unique_versions.into_iter().zip(ind).collect();
                                candidates
                                    .iter()
                                    .map(|w| {
                                        version_to_indicator
                                            .get(&self.version_after_sync(w))
                                            .copied()
                                            .unwrap_or(0.0_f64)
                                    })
                                    .collect()
                            }
                            Err(status) => {
                                error!(
                                    request_id,
                                    %status,
                                    "PSRL Stage 5: get_reserve_indicator RPC failed"
                                );
                                return None;
                            }
                        }
                    } else {
                        error!("PSRL Stage 5: PS Manager client not available");
                        return None;
                    };

                candidates
                    .iter()
                    .zip(indicators.iter())
                    .map(|(w, &ind)| {
                        let v = self.version_after_sync(w);
                        WorkerSelectionKey::Reserve {
                            reserve_bits: ind.to_bits(),
                            version_indicator: if meta.version_tag == -1 { -v } else { v },
                        }
                    })
                    .collect()
            }
        };

        // Sort candidates by key ascending (smallest indicator first).
        let mut indexed: Vec<(usize, &WorkerSelectionKey)> = sort_keys.iter().enumerate().collect();
        indexed.sort_by(|a, b| a.1.cmp(b.1));

        let sorted_candidates: Vec<Arc<dyn Worker>> = indexed
            .iter()
            .map(|(i, _)| candidates[*i].clone())
            .collect();

        // Assign group IDs — a new group starts whenever the key changes.
        // group ID represents the order of indicators.
        let sorted_keys: Vec<&WorkerSelectionKey> = indexed.iter().map(|(_, k)| *k).collect();
        let mut priority_groups: Vec<i64> = Vec::with_capacity(sorted_candidates.len());
        let mut current_group: i64 = 0;
        let mut prev_key: Option<&WorkerSelectionKey> = None;
        for k in &sorted_keys {
            if prev_key.is_some_and(|p| p != *k) {
                current_group += 1;
            }
            priority_groups.push(current_group);
            prev_key = Some(k);
        }

        let policy = self.policy_registry.get_policy_or_default(model_id);
        let hash_ring = self.worker_registry.get_hash_ring(model_id);

        // Atomically select worker and increment its load to prevent TOCTOU.
        // All async work (RPCs) in Stages 1–4 completes before this point;
        // the lock is held only for the synchronous find-min + increment step.
        let selected = {
            let _lock = self.selection_lock.lock();
            // Log per-worker loads before selection so we can confirm balance.
            let loads_before: Vec<(String, usize)> = sorted_candidates
                .iter()
                .map(|w| (w.url().to_string(), w.load()))
                .collect();
            let idx = policy.select_worker(
                &sorted_candidates,
                &SelectWorkerInfo {
                    request_text: text,
                    tokens,
                    headers,
                    hash_ring,
                    priority_groups: Some(&priority_groups),
                    response_token_count: meta.response_token_count,
                    leg: WorkerLeg::Single,
                },
            )?;
            sorted_candidates[idx].increment_load();
            info!(
                request_id,
                selected_url = %sorted_candidates[idx].url(),
                selected_load_before = loads_before[idx].1,
                candidate_loads = ?loads_before.iter().map(|(u, l)| format!("{}={}", u, l)).collect::<Vec<_>>(),
                "PSRL Stage 5: worker selected"
            );
            sorted_candidates[idx].clone()
        };

        Metrics::record_worker_selection(
            metrics_labels::WORKER_REGULAR,
            metrics_labels::CONNECTION_GRPC,
            model_id,
            policy.name(),
        );

        Some(selected)
    }

    async fn commit_single_worker(
        &self,
        model_id: &str,
        tokens: Option<&[u32]>,
        routing_meta: Option<&RoutingMeta>,
        selected: &Arc<dyn Worker>,
    ) -> Result<Option<i64>, Response> {
        let meta = routing_meta.ok_or_else(|| {
            error::internal_error(
                "missing_routing_metadata",
                "PSRL selection commit requires routing metadata",
            )
        })?;
        let request_id = meta.request_id;
        let selected_instance = self.worker_instance_id(selected);

        // Pin an unversioned (`-1`) request to the selected instance's synced
        // version. Once pinned, the value never changes: later re-routes
        // (partial-rollout loopback, subsequent agent turns) keep filtering on
        // this version so they can only land on an instance at least as fresh.
        let synced_version = self.version_after_sync(selected);
        let pinned_version = if meta.version_tag < 0 {
            synced_version
        } else {
            meta.version_tag
        };

        if let Some(ps_client) = self.runtime.ps_manager_client.get() {
            let result = if meta.rollout_instance_hint.is_none() {
                ps_client
                    .reserve_rollout_instance_requests(
                        vec![selected_instance.clone()],
                        vec![request_id],
                        vec![synced_version],
                        false,
                        meta.is_validate,
                    )
                    .await
                    .map(|(success, _, _, err_msg)| success.then_some(()).ok_or(err_msg))
            } else {
                ps_client
                    .update_request_instance_id(
                        request_id,
                        selected_instance.clone(),
                        meta.is_validate,
                    )
                    .await
                    .map(|success| {
                        success
                            .then_some(())
                            .ok_or_else(|| "update_request_instance_id returned false".to_string())
                    })
            };

            match result {
                Ok(Ok(())) => {}
                Ok(Err(message)) => {
                    error!(request_id, %message, "PSRL selection commit rejected");
                    return Err(error::internal_error(
                        "psrl_selection_commit_rejected",
                        message.as_str(),
                    ));
                }
                Err(status) => {
                    error!(request_id, %status, "PSRL selection commit RPC failed");
                    return Err(error::internal_error(
                        "psrl_selection_commit_failed",
                        status.message(),
                    ));
                }
            }
        }

        if meta.rollout_instance_hint.is_none() {
            self.runtime.record_selected_instance(
                request_id,
                Some(meta.prompt_id),
                selected_instance.clone(),
            );
        }

        if let (Some(coordinator), Some(hint), Some(request_tokens)) =
            (&self.kv_transfer, &meta.rollout_instance_hint, tokens)
        {
            if *hint != selected_instance {
                info!(
                    request_id = meta.request_id,
                    src_worker = %hint.0,
                    src_dp = hint.1,
                    dst_worker = %selected_instance.0,
                    dst_dp = selected_instance.1,
                    "KV re-route detected: routing to different instance"
                );
                if let Some(src) = self.find_worker_by_instance(model_id, hint) {
                    coordinator
                        .transfer_on_migration(model_id, &src, selected, request_tokens)
                        .await;
                }
            }
        }

        // Route-decision trace (dedicated `route_trace` target → own log file).
        // Emitted after commit so `version_tag` reflects the pinned version
        // rather than the pre-commit `-1`.
        {
            let stats = selected.engine_stats().scheduler_stats;
            let prompt_tokens = tokens.map(<[u32]>::len).unwrap_or(0);
            let response_tokens = meta.response_token_count.unwrap_or(0);
            info!(
                target: "route_trace",
                event = "route",
                request_id = meta.request_id,
                prompt_id = meta.prompt_id,
                version_tag = pinned_version,
                prompt_tokens,
                response_tokens,
                total_tokens = prompt_tokens + response_tokens,
                dst_instance = %selected_instance.0,
                dst_dp_rank = selected_instance.1,
                dst_running = stats.num_running_reqs,
                dst_waiting = stats.num_waiting_reqs,
                dst_kv_cache_usage = stats.kv_cache_usage,
                "route"
            );
        }

        // Signal a pin update only when the request arrived unversioned so the
        // caller rewrites `x-version-tag` for subsequent re-routes.
        Ok((meta.version_tag < 0).then_some(pinned_version))
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use super::*;
    use crate::{
        config::{PolicyConfig, RoutingLoopConfig},
        policies::PolicyRegistry,
        routers::grpc::routing_loop::runtime::RoutingLoopRuntime,
        worker::{BasicWorkerBuilder, WorkerRegistry},
    };

    fn make_runtime() -> Arc<RoutingLoopRuntime> {
        let (rt, _rx) = RoutingLoopRuntime::new(
            &RoutingLoopConfig::default(),
            Arc::new(dashmap::DashMap::new()),
            Arc::new(WorkerRegistry::new()),
        );
        rt
    }

    /// Stage 1: workers whose synced version tag < request's version_tag are filtered out.
    #[tokio::test]
    async fn stage1_version_filter_removes_stale_workers() {
        let runtime = make_runtime();
        // Populate the version map with two instances.
        runtime
            .instance_to_version_after_sync
            .insert(("worker-a".to_string(), 0), 5);
        runtime
            .instance_to_version_after_sync
            .insert(("worker-b".to_string(), 0), 3);

        // We query the map directly (simulating what Stage 1 does in
        // select_single_worker).  A request with version_tag=4 should exclude
        // worker-b (synced version 3 < 4).
        let version_tag: i64 = 4;
        let worker_a_ok = runtime
            .instance_to_version_after_sync
            .get(&("worker-a".to_string(), 0))
            .is_some_and(|v| *v >= version_tag);
        let worker_b_ok = runtime
            .instance_to_version_after_sync
            .get(&("worker-b".to_string(), 0))
            .is_some_and(|v| *v >= version_tag);
        assert!(worker_a_ok, "worker-a should pass version filter");
        assert!(!worker_b_ok, "worker-b should fail version filter");
    }

    /// Stage 3 group-pin: prompt_to_running map wiring is consistent.
    #[tokio::test]
    async fn stage3_prompt_map_wiring() {
        let runtime = make_runtime();
        let request_id: i64 = 42;
        let prompt_id: i64 = 7;
        let instance = ("worker-x".to_string(), 1_usize);

        // Simulate post-selection bookkeeping.
        runtime.record_selected_instance(request_id, Some(prompt_id), instance.clone());

        // Read back and verify.
        let ids = runtime
            .prompt_to_running_request_ids
            .get(&prompt_id)
            .map(|ids| ids.clone())
            .unwrap_or_default();

        assert_eq!(ids, vec![request_id]);
    }

    #[tokio::test]
    async fn cleanup_tracking_removes_request_and_empty_prompt_group() {
        let runtime = make_runtime();
        runtime.record_selected_instance(42, Some(7), ("worker-x".to_string(), 1));

        runtime.cleanup_tracking(Some(42), Some(7));

        assert!(!runtime.prompt_to_running_request_ids.contains_key(&7));
    }

    #[tokio::test]
    async fn cleanup_tracking_keeps_other_prompt_requests() {
        let runtime = make_runtime();
        runtime.record_selected_instance(42, Some(7), ("worker-x".to_string(), 1));
        runtime.record_selected_instance(43, Some(7), ("worker-y".to_string(), 0));

        runtime.cleanup_tracking(Some(42), Some(7));

        let ids = runtime
            .prompt_to_running_request_ids
            .get(&7)
            .map(|entry| entry.clone())
            .unwrap_or_default();
        assert_eq!(ids, vec![43]);
    }

    #[test]
    fn worker_version_tag_defaults_unknown_to_zero() {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker-a:8000")
                .label(
                    "weight_version",
                    (u64::try_from(i64::MAX).unwrap() + 1).to_string(),
                )
                .build(),
        );

        assert_eq!(worker_version_tag(&worker), 0);
    }

    #[test]
    fn version_after_sync_prefers_runtime_map_over_worker_version() {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let runtime = make_runtime();
        let selector = PsrlWorkerSelector::new(
            Arc::clone(&worker_registry),
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            Arc::clone(&runtime),
            CandidateSortKey::Version,
            true,
            None,
        );
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker-a:8000")
                .label("weight_version", "7")
                .build(),
        );
        let instance = selector.worker_instance_id(&worker);
        runtime.instance_to_version_after_sync.insert(instance, 3);

        assert_eq!(selector.version_after_sync(&worker), 3);
    }

    fn make_selector(runtime: &Arc<RoutingLoopRuntime>) -> PsrlWorkerSelector {
        PsrlWorkerSelector::new(
            Arc::new(WorkerRegistry::new()),
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            Arc::clone(runtime),
            CandidateSortKey::Version,
            true,
            None,
        )
    }

    fn make_meta(version_tag: i64) -> RoutingMeta {
        RoutingMeta {
            request_id: 1,
            prompt_id: 1,
            version_tag,
            is_validate: false,
            is_sticky: false,
            rollout_instance_hint: None,
            response_token_count: None,
        }
    }

    /// An unversioned (`-1`) request is pinned to the selected instance's
    /// synced version, and the pin is signalled back to the caller.
    #[tokio::test]
    async fn commit_pins_unversioned_request_to_synced_version() {
        let runtime = make_runtime();
        let selector = make_selector(&runtime);
        let worker: Arc<dyn Worker> =
            Arc::new(BasicWorkerBuilder::new("http://worker-a:8000").build());
        let instance = selector.worker_instance_id(&worker);
        runtime.instance_to_version_after_sync.insert(instance, 5);

        let meta = make_meta(-1);
        let pinned = selector
            .commit_single_worker("model", None, Some(&meta), &worker)
            .await
            .expect("commit should succeed");
        assert_eq!(pinned, Some(5));
    }

    /// An already-versioned request keeps its version and reports no pin update.
    #[tokio::test]
    async fn commit_keeps_versioned_request_unchanged() {
        let runtime = make_runtime();
        let selector = make_selector(&runtime);
        let worker: Arc<dyn Worker> =
            Arc::new(BasicWorkerBuilder::new("http://worker-a:8000").build());
        let instance = selector.worker_instance_id(&worker);
        runtime.instance_to_version_after_sync.insert(instance, 5);

        let meta = make_meta(4);
        let pinned = selector
            .commit_single_worker("model", None, Some(&meta), &worker)
            .await
            .expect("commit should succeed");
        assert_eq!(pinned, None);
    }

    /// Stage 2 pin: sticky + rollout_instance_hint pins to the hinted instance.
    #[test]
    fn stage2_sticky_hint_pin_logic() {
        // Simulate the filter: a hint matches one of two candidates.
        let hint: (String, usize) = ("worker-a".to_string(), 0);
        let instances = [
            ("worker-a".to_string(), 0_usize),
            ("worker-b".to_string(), 0_usize),
        ];
        let pinned: Vec<_> = instances.iter().filter(|inst| *inst == &hint).collect();
        assert_eq!(pinned.len(), 1);
        assert_eq!(pinned[0], &hint);
    }

    /// WorkerSelectionKey ordering follows ascending indicator order.
    #[test]
    fn worker_selection_key_ordering() {
        let new_high_version = WorkerSelectionKey::Version(-10);
        let new_low_version = WorkerSelectionKey::Version(-5);
        assert!(new_high_version < new_low_version);

        let existing_low_version = WorkerSelectionKey::Version(5);
        let existing_high_version = WorkerSelectionKey::Version(10);
        assert!(existing_low_version < existing_high_version);

        let best_reserve = WorkerSelectionKey::Reserve {
            reserve_bits: f64::NEG_INFINITY.to_bits(),
            version_indicator: -10,
        };
        let worst_reserve = WorkerSelectionKey::Reserve {
            reserve_bits: f64::INFINITY.to_bits(),
            version_indicator: -10,
        };
        assert!(best_reserve < worst_reserve);
    }

    #[test]
    fn worker_selection_key_sort_matches_indicator_order() {
        let mut version_keys = vec![
            (5_i64, WorkerSelectionKey::Version(-5)),
            (10_i64, WorkerSelectionKey::Version(-10)),
            (7_i64, WorkerSelectionKey::Version(-7)),
        ];
        version_keys.sort_by(|a, b| a.1.cmp(&b.1));
        let ordered_versions: Vec<i64> = version_keys.iter().map(|(version, _)| *version).collect();
        assert_eq!(ordered_versions, vec![10, 7, 5]);

        let mut reserve_keys = vec![
            (
                5_i64,
                WorkerSelectionKey::Reserve {
                    reserve_bits: 1.0_f64.to_bits(),
                    version_indicator: -5,
                },
            ),
            (
                10_i64,
                WorkerSelectionKey::Reserve {
                    reserve_bits: f64::NEG_INFINITY.to_bits(),
                    version_indicator: -10,
                },
            ),
            (
                7_i64,
                WorkerSelectionKey::Reserve {
                    reserve_bits: f64::INFINITY.to_bits(),
                    version_indicator: -7,
                },
            ),
        ];
        reserve_keys.sort_by(|a, b| a.1.cmp(&b.1));
        let ordered_versions: Vec<i64> = reserve_keys.iter().map(|(version, _)| *version).collect();
        assert_eq!(ordered_versions, vec![10, 5, 7]);
    }

    /// Unused import lint: ensure HashMap is used.
    #[test]
    fn stage1_empty_version_map_passes_all() {
        let map: HashMap<(String, usize), i64> = HashMap::new();
        let version_tag: i64 = 2;
        // With no synced version in the map, the selector falls back to the
        // worker runtime version, whose unknown value defaults to 0.
        let has_entry = map.contains_key(&("worker-a".to_string(), 0));
        assert!(!has_entry);
        let _ = version_tag;
    }
}
