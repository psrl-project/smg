//! KV-cache transfer coordinator for cross-instance re-routing.
//!
//! When a PSRL rollout request is routed from its previous instance `A`
//! (`rollout_instance_hint`) to a newly-selected instance `B`, the cached
//! prefix that `A` already holds would otherwise be re-prefilled on `B`. This
//! coordinator proactively moves that prefix `A → B` out of band (a unary
//! `TransferKv` RPC to `A`'s servicer, which pulls it into `B` via LMCache
//! `transfer_direct`), so `B` resumes from cache.
//!
//! The re-route is detected by the PSRL worker selector; this module owns the
//! *mechanism*:
//!
//! - **Overlap gate** — consult the event-driven [`KvEventMonitor`] indexer to
//!   confirm `A` actually holds the request prefix; skip the move otherwise.
//! - **Two modes**:
//!   - [`KvTransferMode::Async`] — fire-and-forget; dispatch to `B` immediately.
//!   - [`KvTransferMode::Sync`] — await `TransferKv` before dispatch.
//! - **Back-pressure** — a per-source [`Semaphore`] bounds concurrent ZMQ moves.
//! - **Observability** — Prometheus counters/histogram replace merge_base's
//!   in-memory stat dict.
//!
//! The prefix stays protected while `B` reads it: LMCache takes a read lock on
//! the source objects in the same call that resolves them, and releases it when
//! the read finishes. An explicit pin around the transfer would be redundant
//! with that lock, and a GPU pin would not help because the transfer reads L1,
//! so [`KvTransferMode::PinSync`] is accepted and behaves as `Sync`.

use std::{sync::Arc, time::Duration};

use dashmap::DashMap;
use kv_index::{compute_request_content_hashes, Tier};
use metrics::{counter, histogram};
use smg_grpc_client::vllm_proto::TransferKvRequest;
use tokio::sync::Semaphore;
use tracing::{info, warn};

use crate::{
    config::types::{KvTransferConfig, KvTransferMode},
    worker::{KvEventMonitor, Worker, UNKNOWN_MODEL_ID},
};

/// Worker label carrying the LMCache instance id used as the transfer destination.
const LABEL_LMCACHE_INSTANCE_ID: &str = "lmcache_instance_id";
/// Worker label carrying the LMCache P2P peer URL (NIXL endpoint) of the destination.
const LABEL_LMCACHE_PEER_URL: &str = "lmcache_peer_url";
/// Default LMCache backend location string (matches merge_base default).
const DEFAULT_BACKEND: &str = "LocalCPUBackend";

/// Warns once when a deployment still selects the retired `pin_sync` mode.
static PIN_SYNC_WARNED: std::sync::Once = std::sync::Once::new();

/// Coordinates KV-cache transfers triggered by instance migration.
pub struct KvTransferCoordinator {
    config: KvTransferConfig,
    /// Indexer source used to verify the source instance holds the prefix.
    kv_event_monitor: Option<Arc<KvEventMonitor>>,
    /// Per-source-instance concurrency limiter, keyed by source worker URL.
    /// Bounds in-flight ZMQ moves out of a single instance.
    source_semaphores: DashMap<String, Arc<Semaphore>>,
}

impl KvTransferCoordinator {
    /// Build a coordinator. `kv_event_monitor` supplies the overlap gate; when
    /// `None` the coordinator transfers unconditionally (the migration detector
    /// is then the only gate).
    pub fn new(config: KvTransferConfig, kv_event_monitor: Option<Arc<KvEventMonitor>>) -> Self {
        Self {
            config,
            kv_event_monitor,
            source_semaphores: DashMap::new(),
        }
    }

    /// Trigger a prefix transfer from `src` (old instance A) to `dst` (new
    /// instance B) for `tokens`.
    ///
    /// For [`KvTransferMode::Async`] this returns as soon as the move is
    /// scheduled (the caller dispatches to `B` immediately). For `Sync` /
    /// `PinSync` it awaits the RPC(s) so `B` is warm before dispatch. Always
    /// best-effort: any failure is recorded and swallowed (the request still
    /// succeeds by re-prefilling on `B`).
    pub async fn transfer_on_migration(
        self: &Arc<Self>,
        model_id: &str,
        src: &Arc<dyn Worker>,
        dst: &Arc<dyn Worker>,
        tokens: &[u32],
    ) {
        if !self.config.enable {
            record_skip("disabled");
            return;
        }
        if tokens.is_empty() {
            record_skip("no_overlap");
            return;
        }

        // Overlap gate: confirm A actually holds this prefix.
        if !self.source_has_prefix(model_id, src, tokens) {
            record_skip("no_overlap");
            return;
        }

        // Resolve destination addressing from B's registration labels.
        let Some((dst_instance_id, dst_peer_url)) = destination_endpoint(dst) else {
            warn!(
                dst = dst.url(),
                "KV transfer skipped: destination missing lmcache_instance_id label"
            );
            record_skip("no_dst_endpoint");
            return;
        };

        let request = TransferKvRequest {
            tokens: tokens.to_vec(),
            dst_instance_id,
            dst_peer_url,
            src_backend: DEFAULT_BACKEND.to_string(),
            dst_backend: DEFAULT_BACKEND.to_string(),
            // copy=true: keep the prefix on the source after the push and let
            // its LRU evict naturally. copy=false makes move() remove the source
            // chunks immediately; under group-sticky routing many concurrent
            // migrations share the same prefix (e.g. the SWE-agent system
            // prompt), so racing removals double-free shared MemoryObjs and trip
            // the forced-eviction KeyError path. The destination re-prefills at
            // worst; it never depends on the source dropping its copy.
            copy: true,
            // Pass the destination's current training-checkpoint version so the
            // source servicer looks up the matching version pool in LMCache
            // (multi_version_kv). Falls back to -1 (version-agnostic) when the
            // u64 version overflows i32, which never happens in practice.
            dst_model_version: i32::try_from(dst.dyn_weight_version()).unwrap_or(-1),
        };

        match self.config.transfer_mode {
            KvTransferMode::Async => {
                // Fire-and-forget: schedule the move and let dispatch proceed.
                let this = Arc::clone(self);
                let src = Arc::clone(src);
                #[expect(
                    clippy::disallowed_methods,
                    reason = "async transfer is intentionally detached from the dispatch path"
                )]
                tokio::spawn(async move {
                    this.run_transfer(&src, request).await;
                });
            }
            KvTransferMode::Sync => {
                self.run_transfer(src, request).await;
            }
            KvTransferMode::PinSync => {
                PIN_SYNC_WARNED.call_once(|| {
                    warn!(
                        "kv_transfer.transfer_mode=pin_sync is deprecated and now behaves as \
                         sync: LMCache read-locks the source objects for the duration of the \
                         read, so the surrounding pin added nothing while costing two RPCs \
                         per transfer."
                    );
                });
                self.run_transfer(src, request).await;
            }
        }
    }

    /// Execute a single transfer against source `src`. Holds the per-source
    /// semaphore permit so a single instance can't be flooded with concurrent
    /// ZMQ moves.
    async fn run_transfer(&self, src: &Arc<dyn Worker>, request: TransferKvRequest) {
        let permit_src = self.semaphore_for(src.url());
        let _permit = permit_src.acquire().await;

        let client = match src.get_grpc_client().await {
            Ok(Some(client)) => client,
            Ok(None) | Err(_) => {
                warn!(src = src.url(), "KV transfer: source has no gRPC client");
                record_result("error");
                return;
            }
        };

        let timeout = Duration::from_millis(self.config.transfer_timeout_ms);

        let start = std::time::Instant::now();
        let outcome = tokio::time::timeout(timeout, client.transfer_kv(request)).await;
        histogram!("smg_kv_transfer_latency_seconds").record(start.elapsed().as_secs_f64());

        match outcome {
            Ok(Ok(resp)) if resp.success => {
                info!(
                    src = src.url(),
                    num_tokens = resp.num_tokens,
                    "KV transfer succeeded"
                );
                record_result(if resp.num_tokens > 0 {
                    "ok"
                } else {
                    "src_miss"
                });
            }
            Ok(Ok(resp)) => {
                warn!(src = src.url(), error = %resp.error, "KV transfer failed");
                record_result("error");
            }
            Ok(Err(status)) => {
                warn!(src = src.url(), %status, "KV transfer RPC error");
                record_result("error");
            }
            Err(_) => {
                warn!(src = src.url(), "KV transfer timed out");
                record_result("timeout");
            }
        }

        // The source keeps its copy: its LMCache read lock is already released,
        // and copy=false would race concurrent migrations that share a prefix.
    }

    /// Check whether the source instance holds any of the request's prefix
    /// blocks in either tier, using the event-driven indexer (no extra RPC).
    fn source_has_prefix(&self, model_id: &str, src: &Arc<dyn Worker>, tokens: &[u32]) -> bool {
        let Some(ref monitor) = self.kv_event_monitor else {
            // No indexer to consult — let the migration detector be the gate.
            return true;
        };
        let model_key = if model_id.is_empty() {
            UNKNOWN_MODEL_ID
        } else {
            model_id
        };
        let Some(indexer) = monitor.get_indexer(model_key) else {
            return true;
        };
        // The indexer interns each (worker, dp_rank) stream by `worker.url()`.
        let src_url = src.url();
        for tier in Tier::ALL {
            let block_size = match indexer.block_size(tier) {
                Some(bs) => bs,
                None => continue,
            };
            let hashes = compute_request_content_hashes(tokens, block_size);
            if hashes.is_empty() {
                continue;
            }
            let pi = indexer.tier(tier);
            let Some(wid) = pi.worker_id(src_url) else {
                continue;
            };
            let overlap = pi.find_matches(&hashes, true);
            if overlap.scores.get(&wid).is_some_and(|&s| s > 0) {
                return true;
            }
        }
        false
    }

    /// Get (or lazily create) the concurrency limiter for a source URL.
    fn semaphore_for(&self, src_url: &str) -> Arc<Semaphore> {
        self.source_semaphores
            .entry(src_url.to_string())
            .or_insert_with(|| {
                Arc::new(Semaphore::new(self.config.max_concurrent_per_source.max(1)))
            })
            .clone()
    }
}

/// Read the LMCache destination addressing from a worker's registration labels.
///
/// Only `lmcache_instance_id` is required: the source instance's servicer
/// resolves the real per-rank peer URLs from its own broadcast registry, keyed
/// by this id. `lmcache_peer_url` is an optional single-rank fallback seed,
/// consumed only when the destination is absent from that registry (e.g. an
/// instance added after the broadcast); when missing we pass an empty string,
/// which the servicer treats as "no seed, use the registry".
fn destination_endpoint(dst: &Arc<dyn Worker>) -> Option<(String, String)> {
    let labels = &dst.metadata().spec.labels;
    let instance_id = labels.get(LABEL_LMCACHE_INSTANCE_ID)?.clone();
    if instance_id.is_empty() {
        return None;
    }
    let peer_url = labels
        .get(LABEL_LMCACHE_PEER_URL)
        .cloned()
        .unwrap_or_default();
    Some((instance_id, peer_url))
}

fn record_result(result: &'static str) {
    counter!("smg_kv_transfer_total", "result" => result).increment(1);
}

fn record_skip(reason: &'static str) {
    counter!("smg_kv_transfer_skipped_total", "reason" => reason).increment(1);
}
#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::worker::{BasicWorkerBuilder, WorkerType};

    fn worker_with_labels(url: &str, labels: HashMap<String, String>) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .labels(labels)
                .build(),
        )
    }

    #[test]
    fn destination_endpoint_reads_labels() {
        let mut labels = HashMap::new();
        labels.insert(
            LABEL_LMCACHE_INSTANCE_ID.to_string(),
            "psrl_instance_3".to_string(),
        );
        labels.insert(
            LABEL_LMCACHE_PEER_URL.to_string(),
            "10.0.0.1:18200".to_string(),
        );
        let dst = worker_with_labels("http://b:8000", labels);
        let endpoint = destination_endpoint(&dst);
        assert_eq!(
            endpoint,
            Some(("psrl_instance_3".to_string(), "10.0.0.1:18200".to_string()))
        );
    }

    #[test]
    fn destination_endpoint_requires_only_instance_id() {
        // Missing instance_id (the only required label) → None.
        let dst = worker_with_labels("http://b:8000", HashMap::new());
        assert_eq!(destination_endpoint(&dst), None);

        // Empty instance_id is treated as missing.
        let mut labels = HashMap::new();
        labels.insert(LABEL_LMCACHE_INSTANCE_ID.to_string(), String::new());
        labels.insert(
            LABEL_LMCACHE_PEER_URL.to_string(),
            "10.0.0.1:18200".to_string(),
        );
        let dst = worker_with_labels("http://b:8000", labels);
        assert_eq!(destination_endpoint(&dst), None);

        // instance_id present but peer_url absent → Some with empty seed. The
        // servicer resolves the real per-rank URLs from its broadcast registry.
        let mut labels = HashMap::new();
        labels.insert(
            LABEL_LMCACHE_INSTANCE_ID.to_string(),
            "psrl_instance_7".to_string(),
        );
        let dst = worker_with_labels("http://b:8000", labels);
        assert_eq!(
            destination_endpoint(&dst),
            Some(("psrl_instance_7".to_string(), String::new()))
        );
    }

    #[test]
    fn disabled_coordinator_skips_transfer() {
        let config = KvTransferConfig {
            enable: false,
            ..KvTransferConfig::default()
        };
        let coordinator = Arc::new(KvTransferCoordinator::new(config, None));
        // A disabled coordinator must never attempt a transfer regardless of input.
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("build runtime");
        let src = worker_with_labels("http://a:8000", HashMap::new());
        let dst = worker_with_labels("http://b:8000", HashMap::new());
        rt.block_on(async {
            coordinator
                .transfer_on_migration("m", &src, &dst, &[1, 2, 3, 4])
                .await;
        });
        // No panic / no client call (workers have no gRPC client). The assertion
        // is implicit: the disabled early-return path runs without touching IO.
    }

    #[test]
    fn semaphore_is_per_source_and_bounded() {
        let config = KvTransferConfig {
            max_concurrent_per_source: 2,
            ..KvTransferConfig::default()
        };
        let coordinator = KvTransferCoordinator::new(config, None);
        let s1 = coordinator.semaphore_for("http://a:8000");
        let s1_again = coordinator.semaphore_for("http://a:8000");
        let s2 = coordinator.semaphore_for("http://b:8000");
        // Same source → same semaphore instance; different source → different.
        assert!(Arc::ptr_eq(&s1, &s1_again));
        assert!(!Arc::ptr_eq(&s1, &s2));
        assert_eq!(s1.available_permits(), 2);
    }
}
