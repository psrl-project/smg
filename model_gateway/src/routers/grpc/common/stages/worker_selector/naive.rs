//! Naive worker selector — wraps the existing policy-based selection logic.

use std::sync::Arc;

use async_trait::async_trait;

use super::WorkerSelectorStrategy;
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    policies::{PolicyRegistry, SelectWorkerInfo, WorkerLeg},
    routers::grpc::routing_loop::metadata::RoutingMeta,
    worker::{ConnectionMode, Worker, WorkerRegistry, WorkerType, UNKNOWN_MODEL_ID},
};

/// Wraps the pre-existing policy-based single-worker selection.
///
/// This is the default strategy and is always available regardless of feature flags.
pub(crate) struct NaiveWorkerSelector {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    /// Guards the "find minimum-load worker + increment its load" step so that
    /// concurrent dispatch tasks cannot all read equal loads and pick the same
    /// worker before any guard has been created (TOCTOU race).
    selection_lock: parking_lot::Mutex<()>,
}

impl NaiveWorkerSelector {
    pub(crate) fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
    ) -> Self {
        Self {
            worker_registry,
            policy_registry,
            selection_lock: parking_lot::Mutex::new(()),
        }
    }
}

#[async_trait]
impl WorkerSelectorStrategy for NaiveWorkerSelector {
    async fn select_single_worker(
        &self,
        model_id: &str,
        text: Option<&str>,
        tokens: Option<&[u32]>,
        headers: Option<&http::HeaderMap>,
        _routing_meta: Option<&RoutingMeta>,
    ) -> Option<Arc<dyn Worker>> {
        // Treat "unknown" model as wildcard (match any worker)
        let model_filter = if model_id == UNKNOWN_MODEL_ID {
            None
        } else {
            Some(model_id)
        };

        // Get workers for the specified model, filtered by connection mode
        let workers = self.worker_registry.get_workers_filtered(
            model_filter,
            Some(WorkerType::Regular),
            Some(ConnectionMode::Grpc),
            None,  // any runtime type
            false, // get all workers, we'll filter by is_available() next
        );

        // Use into_iter() to take ownership of Arcs without cloning (avoids atomic inc/dec)
        let available: Vec<Arc<dyn Worker>> =
            workers.into_iter().filter(|w| w.is_available()).collect();

        if available.is_empty() {
            return None;
        }

        // Get the appropriate policy for this model
        let policy = self.policy_registry.get_policy_or_default(model_id);

        // Get cached hash ring for consistent hashing (O(log n) lookup)
        let hash_ring = self.worker_registry.get_hash_ring(model_id);

        // Atomically select worker and increment its load to prevent TOCTOU.
        // All N concurrent dispatch tasks call select_worker simultaneously;
        // without the lock they all read load=0 and all pick the same worker
        // (stable tie-break on equal loads → always index 0).
        let selected = {
            let _lock = self.selection_lock.lock();
            let idx = policy.select_worker(
                &available,
                &SelectWorkerInfo {
                    request_text: text,
                    tokens,
                    headers,
                    hash_ring,
                    priority_groups: None,
                    response_token_count: None,
                    leg: WorkerLeg::Single,
                },
            )?;
            available[idx].increment_load();
            available[idx].clone()
        };

        // Record worker selection metric
        Metrics::record_worker_selection(
            metrics_labels::WORKER_REGULAR,
            metrics_labels::CONNECTION_GRPC,
            model_id,
            policy.name(),
        );

        Some(selected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::types::PolicyConfig, policies::PolicyRegistry, worker::WorkerRegistry};

    #[tokio::test]
    async fn returns_none_when_no_workers() {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        let selector = NaiveWorkerSelector::new(worker_registry, policy_registry);
        let result = selector
            .select_single_worker("some-model", None, None, None, None)
            .await;
        assert!(result.is_none());
    }
}
