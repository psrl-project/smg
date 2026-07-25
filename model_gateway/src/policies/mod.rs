//! Load balancing policies for SGLang router
//!
//! This module provides a unified abstraction for routing policies that work
//! across both regular and prefill-decode (PD) routing modes.

use std::{fmt::Debug, sync::Arc};

use openai_protocol::worker::WorkerLoadResponse;

use crate::worker::{HashRing, Worker};

mod bucket;
mod cache_aware;
mod cache_aware_v1;
mod consistent_hashing;
pub(crate) mod cost_model_utils;
mod dp_min_token;
mod factory;
mod least_load;
mod manual;
mod passthrough;
mod power_of_two;
mod prefix_hash;
mod random;
mod registry;
mod request_num_balance;
mod round_robin;
mod throughput_optimal;
pub(crate) mod utils;

pub use bucket::BucketPolicy;
pub use cache_aware::{CacheAwarePolicy, TreeHandle, TreeKind};
pub use cache_aware_v1::CacheAwareV1Policy;
pub use consistent_hashing::ConsistentHashingPolicy;
pub use dp_min_token::MinimumTokensPolicy;
pub use factory::PolicyFactory;
// Re-export PrefixMatchResult from kv_index for production use
pub use kv_index::PrefixMatchResult;
pub use least_load::LeastLoadPolicy;
pub use manual::{ManualConfig, ManualPolicy};
pub use passthrough::PassthroughPolicy;
pub use power_of_two::PowerOfTwoPolicy;
pub use prefix_hash::{PrefixHashConfig, PrefixHashPolicy};
pub use random::RandomPolicy;
pub use registry::PolicyRegistry;
pub use request_num_balance::RequestNumBalancePolicy;
pub use round_robin::RoundRobinPolicy;
pub use throughput_optimal::{
    ThroughputOptimalConfig, ThroughputOptimalPolicy, ThroughputOptimalWithBudgetPolicy,
};

/// Returns true for SMG cache-aware routing policy names.
pub(crate) fn is_cache_aware_policy_name(name: &str) -> bool {
    matches!(name, "cache_aware" | "cache_aware_v1")
}

/// Core trait for load balancing policies
///
/// This trait provides a unified interface for implementing routing algorithms
/// that can work with both regular single-worker selection and PD dual-worker selection.
pub trait LoadBalancingPolicy: Send + Sync + Debug {
    /// Select a single worker from the available workers
    ///
    /// This is used for regular routing mode where requests go to a single worker.
    /// Now uses Arc<dyn Worker> for better performance and to avoid unnecessary cloning.
    ///
    /// # Arguments
    /// * `workers` - Available workers to select from
    /// * `info` - Additional information for routing decisions
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize>;

    /// Update policy state after request completion
    ///
    /// This is called when a request completes (successfully or not) to allow
    /// policies to update their internal state.
    fn on_request_complete(&self, _worker_url: &str, _success: bool) {
        // Default: no-op for stateless policies
    }

    /// Get policy name for metrics and debugging
    fn name(&self) -> &'static str;

    /// Check if this policy needs request text for routing decisions
    fn needs_request_text(&self) -> bool {
        false // Default: most policies don't need request text
    }

    /// Whether this policy routes on the per-worker load counter
    /// ([`Worker::load`]), and therefore requires a [`WorkerLoadGuard`] to be
    /// minted on the request path so the counter is incremented on admit and
    /// decremented on departure.
    ///
    /// The gRPC routing loop always mints load guards (in the execution stage),
    /// so this flag only governs the HTTP router, which mints them
    /// conditionally to avoid the (tiny) overhead for policies that ignore the
    /// counter. Load-aware policies (`request_num_balance`, `throughput_optimal`,
    /// `cache_aware`, `manual`) must return `true`; stateless ones
    /// (`random`, `round_robin`, `consistent_hashing`, …) leave the default.
    ///
    /// [`Worker::load`]: crate::worker::Worker::load
    /// [`WorkerLoadGuard`]: crate::worker::WorkerLoadGuard
    fn needs_load_guard(&self) -> bool {
        false // Default: stateless policies don't read the load counter
    }

    /// Update worker load information
    ///
    /// This is called periodically with current load information for load-aware policies.
    fn update_loads(&self, _loads: &std::collections::HashMap<String, WorkerLoadResponse>) {
        // Default: no-op for policies that don't use load information
    }

    /// Drop any cached per-worker state for a removed worker.
    ///
    /// Called when a worker leaves the registry so load-aware policies don't
    /// accumulate stale load reports under worker churn (autoscaling, rolling
    /// updates). Default is a no-op for stateless policies.
    fn remove_worker(&self, _url: &str) {
        // Default: no-op for policies that don't cache per-worker state
    }

    /// Reset any internal state
    ///
    /// This is useful for policies that maintain state (e.g., round-robin counters).
    fn reset(&self) {
        // Default: no-op for stateless policies
    }

    /// Admission gate: whether `worker` can accept a new request without pushing
    /// the instance into a queued / overloaded state.
    ///
    /// Applied by the PSRL worker selector to every candidate *before* delegating
    /// the final pick, so it governs all routing methods uniformly (the old
    /// per-policy `enable_kv_admission_control` only affected cache-aware). When
    /// every candidate is rejected the selector returns `None` and the routing
    /// loop re-enqueues the request (wait) instead of piling onto a full instance.
    ///
    /// The default gates on three signals: the in-flight request-count cap
    /// (`worker.load()`, exact and race-free under the selector lock), the engine
    /// waiting queue, and a coarse KV-usage ceiling. `cache_aware_v1` overrides
    /// this with a prefix-tree-based marginal-KV estimate. Returns `true` to admit.
    fn admits(
        &self,
        worker: &Arc<dyn Worker>,
        info: &SelectWorkerInfo,
        cfg: &AdmissionGateConfig,
    ) -> bool {
        self.admission_reject_reason(worker, info, cfg).is_none()
    }

    /// Same gate as [`LoadBalancingPolicy::admits`] but returns *why* a candidate
    /// was rejected (`None` = admit), so the selector can emit fine-grained
    /// diagnostics (which limb fired, on what speculative-vs-real values).
    /// Overriding this is the single point a policy customizes admission.
    fn admission_reject_reason(
        &self,
        worker: &Arc<dyn Worker>,
        _info: &SelectWorkerInfo,
        cfg: &AdmissionGateConfig,
    ) -> Option<&'static str> {
        if admission_count_full(worker, cfg) {
            return Some("count_cap");
        }
        if cfg.reject_on_waiting
            && worker.engine_stats_timestamp_ms() != 0
            && worker.engine_stats_arc().scheduler_stats.num_waiting_reqs > 0
        {
            return Some("waiting_queue");
        }
        None
    }

    /// Drop any speculative per-instance state for a worker whose model version
    /// just bumped (a Pull / weight sync).
    ///
    /// On a version bump the engine interrupts and clears its run/wait queues and
    /// its KV cache for the old weights, so any router-side speculative state for
    /// that instance (cache-affinity / hypothetical KV trees, per-instance token
    /// deltas, …) is stale and must be dropped. This is a uniform concern across
    /// strategies — every stateful policy should override it — so it lives on the
    /// trait with a no-op default rather than being special-cased per policy.
    /// Called from the worker weight-version update path with the instance URL.
    fn on_version_bump(&self, _worker_url: &str) {
        // Default: no-op for policies without per-instance speculative state
    }

    /// Get as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;
}

pub trait DPRankLoadPolicy: Send + Sync + Debug {
    fn select_dp_rank(&self, worker: &dyn Worker, estimated_cost: isize) -> Option<isize>;
}

/// Configuration for cache-aware policy
#[derive(Debug, Clone)]
pub struct CacheAwareConfig {
    pub cache_threshold: f32,
    pub balance_abs_threshold: usize,
    pub balance_rel_threshold: f32,
    pub eviction_interval_secs: u64,
    pub max_tree_size: usize,
    /// Backend KV cache block size (tokens per block) for event-driven routing.
    /// Used by `compute_request_content_hashes` to chunk request tokens into blocks.
    /// Must match the backend's block size. Default: 16 (SGLang page size).
    pub block_size: usize,
    /// Weight applied to GPU-tier overlap in event-driven scoring. A GPU hit
    /// implies zero reload cost, so it is weighted highest. Default: 1.0.
    pub gpu_overlap_weight: f64,
    /// Weight applied to LMCache-tier (off-GPU) overlap in event-driven scoring.
    /// A hit still requires loading the prefix back onto the GPU, so it scores
    /// below a GPU hit. Set 0.0 to ignore the off-GPU tier. Default: 0.5.
    pub lmcache_overlap_weight: f64,
    /// KV-usage **spread** (hottest minus coldest backend, 0.0–1.0) above which
    /// the pool is treated as imbalanced and cache affinity is abandoned for
    /// shortest-queue. Requires the backend to report `token_usage`
    /// (gRPC/`GetLoads`); falls back to the count spread when unavailable.
    /// `>= 1.0` disables it (default).
    pub balance_token_usage_threshold: f32,
    /// Backend KV-cache utilization **ceiling** (0.0–1.0): when the hottest
    /// engine exceeds it the pool is treated as imbalanced regardless of spread.
    /// A safety valve, best set high (e.g. 0.9). `>= 1.0` disables it (default).
    pub overload_token_usage_threshold: f32,
    /// KV-capacity admission control (port of the old Python router's
    /// `_can_run_directly`). When enabled, `select_worker` rejects any candidate
    /// that would either queue behind existing waiting requests or exceed its KV
    /// token capacity (`num_used_tokens + new_request_tokens >
    /// max_total_num_tokens`); if every healthy candidate is rejected the call
    /// returns `None` and the routing loop re-enqueues the request for the next
    /// tick. Off by default to preserve current always-route behavior.
    pub enable_kv_admission_control: bool,
    /// Fraction of KV capacity (0.0–1.0] at which admission is refused. A value
    /// below 1.0 reserves headroom for response-token growth during decode
    /// (e.g. 0.9 rejects when effective_used + new > 0.9 × capacity). Default
    /// 1.0 means reject only at full capacity (original behavior).
    pub kv_capacity_threshold: f64,
}

impl Default for CacheAwareConfig {
    fn default() -> Self {
        Self {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 30,
            max_tree_size: 10000,
            block_size: 16,
            gpu_overlap_weight: 1.0,
            lmcache_overlap_weight: 0.5,
            balance_token_usage_threshold: 1.0,
            overload_token_usage_threshold: 1.0,
            enable_kv_admission_control: false,
            kv_capacity_threshold: 1.0,
        }
    }
}

/// KV-capacity admission predicate (port of the old Python router's
/// `_can_run_directly`).
///
/// An engine **admits** a request directly iff some DP rank has no waiting
/// queue *and* room for `new_request_tokens`. The scheduler places a request on
/// exactly one rank, so one uncongested rank with capacity means the request
/// runs directly — hence the per-rank predicate is OR-reduced across ranks.
/// Summing ranks would be wrong (a request can't be split), and a worst-case
/// (max-used rank) check would be needlessly pessimistic. Reduces to the
/// Python single-instance check when `loads.len() == 1`. An empty snapshot is
/// treated as "admit" (no data ⇒ no gate, avoid deadlock).
pub(crate) fn kv_admits(load: &WorkerLoadResponse, new_request_tokens: i64) -> bool {
    if load.loads.is_empty() {
        return true;
    }
    load.loads.iter().any(|r| {
        r.num_waiting_reqs == 0
            && (r.num_used_tokens as i64 + new_request_tokens) <= r.max_total_num_tokens as i64
    })
}

/// Worker KV-cache capacity in tokens, derived from the `max_model_len` or
/// `max_total_tokens` worker label. Returns 0 when neither label is set,
/// which disables KV-capacity checks for that worker.
pub(crate) fn worker_kv_capacity(worker: &Arc<dyn Worker>) -> i64 {
    let labels = &worker.metadata().spec.labels;
    labels
        .get("max_model_len")
        .and_then(|v| v.parse::<i64>().ok())
        .or_else(|| {
            labels
                .get("max_total_tokens")
                .and_then(|v| v.parse::<i64>().ok())
        })
        .unwrap_or(0)
}

/// Effective KV-token occupancy: engine-snapshot tokens + speculative inflight.
///
/// Mirrors `ThroughputRuntime::current_token_num`: uses exact per-request token
/// maps when populated; falls back to `kv_cache_usage × capacity` when maps are
/// empty but the queue is non-zero; otherwise returns 0 (no stats or idle).
/// The `inflight_tokens_sum()` delta covers requests admitted since the last
/// snapshot and is cleared once the snapshot's request count agrees with the
/// router's in-flight counter (engine_stats agreement-gated rebase).
pub(crate) fn effective_kv_used_tokens(worker: &Arc<dyn Worker>) -> i64 {
    let stats = worker.engine_stats_arc();
    let has_queue =
        stats.scheduler_stats.num_running_reqs > 0 || stats.scheduler_stats.num_waiting_reqs > 0;

    let engine_tokens: i64 = if stats.total_token_num() > 0 {
        stats.total_token_num() as i64
    } else if has_queue {
        // Per-request token maps not populated — derive from kv_cache_usage ratio.
        let capacity = worker_kv_capacity(worker);
        if capacity > 0 {
            (stats.scheduler_stats.kv_cache_usage * capacity as f64).ceil() as i64
        } else {
            0
        }
    } else {
        0
    };

    engine_tokens + worker.inflight_tokens_sum() as i64
}

/// Admission gate params for the PSRL worker selector (Stage 5). Always active;
/// default (count cap 0, `reject_on_waiting` false) is a no-op. No coarse KV-usage
/// limb: a reactive hard KV cutoff oscillates (bang-bang); use the count cap for
/// flow control and the reservation-based path for KV.
#[derive(Debug, Clone, Copy)]
pub struct AdmissionGateConfig {
    /// Reject when in-flight request count reaches this cap. `0` disables.
    pub max_concurrent_seqs_per_instance: usize,
    /// When true, reject any worker with `num_waiting_reqs > 0` (strict). Separate
    /// from `max_num_waiting_reqs_after_preemption` (vLLM preemption notification).
    pub reject_on_waiting: bool,
}

impl Default for AdmissionGateConfig {
    fn default() -> Self {
        Self {
            max_concurrent_seqs_per_instance: 0,
            reject_on_waiting: false,
        }
    }
}

/// Whether `worker` is at or above the in-flight request-count cap. Reads the
/// exact `load()` counter (race-free under the selector lock). `cap == 0`
/// disables the check.
pub(crate) fn admission_count_full(worker: &Arc<dyn Worker>, cfg: &AdmissionGateConfig) -> bool {
    cfg.max_concurrent_seqs_per_instance > 0
        && worker.load() >= cfg.max_concurrent_seqs_per_instance
}

#[derive(Debug, Clone)]
pub struct BucketConfig {
    pub balance_abs_threshold: usize,
    pub balance_rel_threshold: f32,
    pub bucket_adjust_interval_secs: usize,
}

impl Default for BucketConfig {
    fn default() -> Self {
        Self {
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.0001,
            bucket_adjust_interval_secs: 5,
        }
    }
}

/// Helper function to filter healthy workers and return their indices
pub(crate) fn get_healthy_worker_indices(workers: &[Arc<dyn Worker>]) -> Vec<usize> {
    workers
        .iter()
        .enumerate()
        .filter(|(_, w)| w.is_available())
        .map(|(idx, _)| idx)
        .collect()
}

/// Helper function to normalize model_id to a key for policy lookups.
///
/// Returns UNKNOWN_MODEL_ID for empty model_ids to ensure consistent behavior
/// across single-model and multi-model deployments.
#[inline]
pub(crate) fn normalize_model_key(model_id: &str) -> &str {
    if model_id.is_empty() {
        crate::worker::UNKNOWN_MODEL_ID
    } else {
        model_id
    }
}

/// Display/correlation metadata for `score_trace.log` (KV-cache-aware-v1
/// per-decision score trace). Populated by the PSRL worker selector only when
/// the `score_trace` target is enabled; a `None` `SelectWorkerInfo::score_trace`
/// makes score tracing a no-op (zero extra work on the hot path).
#[derive(Debug, Clone, Copy)]
pub struct ScoreTraceCtx<'a> {
    /// Request id, for correlating with `route_trace.log`.
    pub request_id: i64,
    /// Prompt id, for correlating with `route_trace.log`.
    pub prompt_id: i64,
    /// Instance id per candidate, indexed identically to the `workers` slice
    /// passed to `select_worker` (same convention as `priority_groups`).
    pub instance_ids: &'a [String],
    /// Instance id the request ran on last round (rollout hint), if any.
    pub prev_instance_id: Option<&'a str>,
}

/// Information passed to policy for worker selection
#[derive(Debug, Clone, Default)]
pub struct SelectWorkerInfo<'a> {
    /// Request text for cache-aware routing
    pub request_text: Option<&'a str>,
    /// Tokenized request for prefix-hash routing
    /// Used by PrefixHashPolicy for token-based prefix hashing
    pub tokens: Option<&'a [u32]>,
    /// HTTP headers for header-based routing policies
    /// Policies can extract routing information from headers like:
    /// - X-SMG-Target-Worker: Direct routing to a specific worker by index
    /// - X-SMG-Routing-Key: Consistent hash routing for session affinity
    pub headers: Option<&'a http::HeaderMap>,
    /// Pre-computed hash ring for O(log n) consistent hashing
    /// Built and cached by WorkerRegistry, passed through to avoid per-request rebuilds
    pub hash_ring: Option<Arc<HashRing>>,
    /// Number of response tokens already generated (for continuation / multi-turn requests).
    ///
    /// Used by throughput-optimal policies to correctly split prompt vs. response tokens
    /// when computing KV-cache budget-aligned token counts. When `None`, the policy
    /// conservatively treats all tokens as prompt tokens.
    ///
    /// TODO(psrl-refactor): populate from parsed request body (commit 25ad721b)
    pub response_token_count: Option<usize>,
    /// Per-worker priority group values for version-aware routing.
    ///
    /// When set, this slice has one entry per worker (indexed identically to the
    /// `workers` slice passed to `select_worker`).  Workers are grouped by their
    /// priority value; the group with the **largest** value is tried first, and the
    /// policy falls back to lower-priority groups only if no worker in the
    /// higher-priority group can accept the request.
    ///
    /// `None` means all workers are treated as equal priority (default behaviour).
    pub priority_groups: Option<&'a [i64]>,
    /// Display/correlation metadata for the KV-cache-aware-v1 `score_trace.log`.
    /// `None` (default) disables score tracing entirely; set by the PSRL worker
    /// selector only when the `score_trace` target is enabled.
    pub score_trace: Option<ScoreTraceCtx<'a>>,
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::{HealthCheckConfig, WorkerStatus};

    use super::*;
    use crate::worker::{BasicWorkerBuilder, WorkerType};

    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_get_healthy_worker_indices() {
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key2")
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .health_config(no_health_check())
                    .build(),
            ),
        ];

        // All healthy initially
        let indices = get_healthy_worker_indices(&workers);
        assert_eq!(indices, vec![0, 1, 2]);

        // Mark one unhealthy
        workers[1].set_status(WorkerStatus::NotReady);
        let indices = get_healthy_worker_indices(&workers);
        assert_eq!(indices, vec![0, 2]);
    }

    /// Only `Ready` workers may be selected. Pending, NotReady, Failed, and
    /// Draining are all excluded. Draining specifically guards against
    /// routing new traffic to a worker that is being torn down.
    #[test]
    fn test_get_healthy_worker_indices_excludes_each_non_ready_status() {
        let cases = [
            (WorkerStatus::Pending, false),
            (WorkerStatus::Ready, true),
            (WorkerStatus::NotReady, false),
            (WorkerStatus::Failed, false),
            (WorkerStatus::Draining, false),
        ];

        for (status, expected_included) in cases {
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new("http://w:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("k")
                    .health_config(no_health_check())
                    .build(),
            );
            worker.set_status(status);
            let workers = vec![worker];
            let indices = get_healthy_worker_indices(&workers);
            assert_eq!(
                indices == vec![0],
                expected_included,
                "status {status:?} should be {}",
                if expected_included {
                    "included"
                } else {
                    "excluded"
                }
            );
        }
    }
}
