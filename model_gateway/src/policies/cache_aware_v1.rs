/*
    Cache-Aware Load Balancing Router

    When load is balanced, uses cache-aware routing. When imbalanced, uses
    shortest-queue. A system is imbalanced when both:
        (max - min) > abs_threshold  AND  max > rel_threshold * min

    Three types of cache-aware routing (mutually exclusive, selected by
    worker connection mode and KV event availability):

    1. Event-Driven (gRPC + KV events)
    -------------------------------------------
    Uses PositionalIndexer overlap scoring from KvEventMonitor. Routes based
    on actual backend KV cache state. Selects the worker with the highest
    overlap count; tie-breaks by load (lower) then tree size (smaller).
    Falls back to min-load when no cache overlap exists.

    2. Approximate Token Tree (gRPC, no KV events)
    -------------------------------------------
    Maintains a TokenTree per model tracking which token prefixes were routed
    where. If match_rate > cache_threshold, routes to the best-matching worker.
    Otherwise routes to the worker with the smallest tree (most cache capacity).

    3. Approximate String Tree (HTTP)
    -------------------------------------------
    Same algorithm as (2) but operates on raw text characters instead of
    token IDs, avoiding tokenization overhead.

    Load Balancing (Shortest Queue)
    -------------------------------------------
    When the system is imbalanced, routes to the least busy worker regardless
    of cache affinity.

    Configuration Parameters:
    ------------------------
    cache_threshold:         Min prefix match ratio for highest-match routing (0.0-1.0)
    balance_abs_threshold:   Absolute load diff threshold for imbalance detection
    balance_rel_threshold:   Relative load ratio threshold for imbalance detection
    eviction_interval_secs:  Interval between LRU eviction cycles
    max_tree_size:           Max nodes per approximate tree before eviction
    block_size:              Backend KV cache block size for event-driven routing
*/

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use dashmap::DashMap;
use kv_index::{compute_request_content_hashes, Tier, TieredIndexer, TokenTree, Tree};
use openai_protocol::worker::WorkerLoadResponse;
use parking_lot::RwLock;
use rand::RngExt;
use tokio::sync::watch;
use tracing::{debug, info, warn};

use super::{
    effective_kv_used_tokens, normalize_model_key, utils::PeriodicTask, worker_kv_capacity,
    CacheAwareConfig, LoadBalancingPolicy, ScoreTraceCtx, SelectWorkerInfo, TreeHandle, TreeKind,
};
use crate::{
    mesh::adapters::tree_sync::{RepairEntry, TreeRepairPage},
    observability::logging::SCORE_TRACE_TARGET,
    worker::{KvEventMonitor, Worker},
};

/// Latest per-worker backend load snapshot stream, keyed by worker URL.
pub(crate) type LoadReceiver = watch::Receiver<HashMap<String, WorkerLoadResponse>>;

/// Format `inst<id>/dp<rank>` for candidate `idx` using the score-trace
/// instance-id slice (indexed identically to `workers`). Used only when
/// building `score_trace.log` lines.
fn score_inst_label(st: &ScoreTraceCtx, workers: &[Arc<dyn Worker>], idx: usize) -> String {
    let inst = st.instance_ids.get(idx).map(String::as_str).unwrap_or("?");
    let dp = workers[idx].dp_rank().unwrap_or(0);
    format!("inst{inst}/dp{dp}")
}

/// Candidate index whose instance id equals the request's previous-round
/// instance (`prev_instance_id`), if that instance is among `workers`.
fn score_prev_idx(st: &ScoreTraceCtx, workers: &[Arc<dyn Worker>]) -> Option<usize> {
    let prev = st.prev_instance_id?;
    (0..workers.len()).find(|&idx| st.instance_ids.get(idx).map(String::as_str) == Some(prev))
}

/// Render the previous-round instance for a load-based (non-scoring) channel:
/// `inst../dp..(load=N)`, or a reason string when there is no prev candidate.
fn score_prev_load_label(st: &ScoreTraceCtx, workers: &[Arc<dyn Worker>]) -> String {
    match st.prev_instance_id {
        None => "none(first_turn)".to_string(),
        Some(prev) => match score_prev_idx(st, workers) {
            Some(idx) => format!("{}(load={})", score_inst_label(st, workers, idx), workers[idx].load()),
            None => format!("inst{prev}(not_a_candidate)"),
        },
    }
}

/// Cache-aware routing policy
///
/// Routes requests based on cache affinity when load is balanced,
/// switches to shortest-queue routing when load is imbalanced.
/// Maintains separate trees per model for multi-model support.
/// Supports mesh synchronization of tree operations across cluster nodes.
/// When mesh is not enabled, the policy works independently without synchronization.
///
/// Supports both HTTP (string-based) and gRPC (token-based) connections:
/// - HTTP requests use StringTree (character-based prefix matching)
/// - gRPC requests use TokenTree (token-based prefix matching, page-aligned)
#[derive(Debug)]
pub struct CacheAwareV1Policy {
    config: CacheAwareConfig,
    /// String-based trees for HTTP connections (text input)
    string_trees: Arc<DashMap<String, Arc<Tree>>>,
    /// Token-based trees for gRPC connections (pre-tokenized input)
    token_trees: Arc<DashMap<String, Arc<TokenTree>>>,
    _eviction_task: Option<PeriodicTask>,
    /// Event-driven KV cache monitor for overlap scoring (gRPC workers only).
    kv_monitor: RwLock<Option<Arc<KvEventMonitor>>>,
    /// Latest per-worker backend load snapshot (keyed by worker URL) from the
    /// `WorkerMonitor` load poll. Read on the hot path for the KV-usage imbalance
    /// trigger. `None` until wired by the registry (then the policy stays
    /// count-only, preserving current behavior).
    load_rx: RwLock<Option<LoadReceiver>>,
    /// Model-scoped hash indexes for resolving tenant delta hashes.
    /// Outer key is the normalized model_id; inner maps hold
    /// `hash → reconstructable prefix/tokens` per tree kind.
    /// Spec §7.1 mandates model scoping: the same hash can refer
    /// to different prefixes in different models, so a global
    /// index mis-routes multi-model deployments. Bounded by
    /// eviction at `max_tree_size` total entries.
    ///
    /// Per-entry value semantics differ by populate site:
    /// - `select_worker_*` (request hot paths) store the prior
    ///   shared prefix from a pre-insert match. Bytes/entry is
    ///   bounded by tree depth, not input size — a 32K-token
    ///   request costs O(matched-prefix), not O(input).
    /// - `apply_repair_page` (cold-start replay) stores the full
    ///   inserted path because the canonical path is required to
    ///   attach remote tenants at the correct node. This path
    ///   runs at replay frequency, not request rate.
    hash_index: Arc<DashMap<String, PerModelHashIndex>>,
    /// Gate request-hot-path `hash_index` writes. The index's only
    /// consumers are mesh paths (`apply_known_remote_insert` reads,
    /// `apply_repair_page` writes). When mesh is disabled the
    /// hot-path writes accumulate with no reader and OOM the
    /// gateway. Off by default; the mesh wiring code flips it on
    /// when it attaches.
    populate_hash_index: AtomicBool,
}

/// Per-model inner container for [`CacheAwarePolicy::hash_index`].
/// Keeping both kinds in one struct per model makes the
/// "separate model-scoped hash indexes for string and token
/// trees" invariant from spec §7.1 explicit in the type.
#[derive(Debug, Default)]
struct PerModelHashIndex {
    /// path hash → matched prefix (reconstructs the string-tree node).
    string_tree: DashMap<u64, String>,
    /// token-path hash → tokens (reconstructs the token-tree node).
    token_tree: DashMap<u64, Vec<u32>>,
}

impl CacheAwareV1Policy {
    pub fn new() -> Self {
        Self::with_config(CacheAwareConfig::default())
    }

    pub fn with_config(config: CacheAwareConfig) -> Self {
        let string_trees = Arc::new(DashMap::<String, Arc<Tree>>::new());
        let token_trees = Arc::new(DashMap::<String, Arc<TokenTree>>::new());
        let hash_index = Arc::new(DashMap::<String, PerModelHashIndex>::new());

        // Start background eviction thread if configured
        let eviction_task = if config.eviction_interval_secs > 0 {
            let string_trees_clone = Arc::clone(&string_trees);
            let token_trees_clone = Arc::clone(&token_trees);
            let hash_index_clone = Arc::clone(&hash_index);
            let max_tree_size = config.max_tree_size;

            Some(PeriodicTask::spawn(
                config.eviction_interval_secs,
                "Eviction",
                move || {
                    // Evict string trees (HTTP)
                    for tree_ref in string_trees_clone.iter() {
                        let model_id = tree_ref.key();
                        let tree = tree_ref.value();
                        tree.evict_tenant_by_size(max_tree_size);

                        debug!(
                            "String tree eviction completed for model {}, max_size: {}",
                            model_id, max_tree_size
                        );
                    }
                    // Evict token trees (gRPC)
                    for tree_ref in token_trees_clone.iter() {
                        let model_id = tree_ref.key();
                        let tree = tree_ref.value();
                        tree.evict_tenant_by_size(max_tree_size);

                        debug!(
                            "Token tree eviction completed for model {}, max_size: {}",
                            model_id, max_tree_size
                        );
                    }
                    // Evict hash index per model: `max_tree_size` is a
                    // per-tree bound, so clearing one model's overflow
                    // must not wipe other models' still-valid metadata.
                    // Each tree kind is checked independently.
                    let mut hash_total: usize = 0;
                    for entry in hash_index_clone.iter() {
                        let per_model = entry.value();
                        if per_model.string_tree.len() > max_tree_size {
                            per_model.string_tree.clear();
                            debug!(
                                model_id = entry.key(),
                                "String hash index cleared (exceeded max_tree_size: {})",
                                max_tree_size
                            );
                        }
                        if per_model.token_tree.len() > max_tree_size {
                            per_model.token_tree.clear();
                            debug!(
                                model_id = entry.key(),
                                "Token hash index cleared (exceeded max_tree_size: {})",
                                max_tree_size
                            );
                        }
                        hash_total += per_model.string_tree.len() + per_model.token_tree.len();
                    }

                    // Log tree sizes — model counts + hash-index total.
                    // DO NOT call tree.snapshot() here — it clones all
                    // edge text (~170 MB) every cycle.
                    tracing::info!(
                        "Tree memory: string_trees={} models, token_trees={} models, \
                         hash_index={} models / {} entries",
                        string_trees_clone.len(),
                        token_trees_clone.len(),
                        hash_index_clone.len(),
                        hash_total,
                    );
                },
            ))
        } else {
            None
        };

        Self {
            config,
            string_trees,
            token_trees,
            _eviction_task: eviction_task,
            kv_monitor: RwLock::new(None),
            load_rx: RwLock::new(None),
            hash_index,
            populate_hash_index: AtomicBool::new(false),
        }
    }

    /// Enable request-hot-path `hash_index` population. Called by mesh
    /// wiring when the policy is attached to a mesh adapter; otherwise
    /// the index stays empty (its only readers are mesh-only paths).
    pub fn set_populate_hash_index(&self, enabled: bool) {
        self.populate_hash_index.store(enabled, Ordering::Relaxed);
    }

    fn should_populate_hash_index(&self) -> bool {
        self.populate_hash_index.load(Ordering::Relaxed)
    }

    /// Set event-driven KV cache monitor (thread-safe, can be called after construction).
    /// Uses interior mutability so this works on policies behind `Arc<dyn LoadBalancingPolicy>`.
    pub fn set_kv_event_monitor(&self, monitor: Option<Arc<KvEventMonitor>>) {
        *self.kv_monitor.write() = monitor;
    }

    /// Set the backend load-snapshot receiver (thread-safe, after construction).
    /// Wired from the `WorkerMonitor` via the `PolicyRegistry` so the KV-usage
    /// imbalance trigger can read fresh per-worker `token_usage`.
    pub fn set_load_receiver(&self, rx: Option<LoadReceiver>) {
        *self.load_rx.write() = rx;
    }

    /// True when the pool is imbalanced enough to abandon cache affinity.
    ///
    /// Three independent triggers, OR'd together. The two KV-based triggers
    /// require a backend `token_usage` snapshot and are disabled at their `1.0`
    /// default (utilization and spread are both `<= 1.0`, so `> 1.0` never
    /// fires):
    ///
    /// - **overload** (`overload_token_usage_threshold`): the hottest engine's
    ///   KV utilization exceeds the ceiling — a critically-saturated engine,
    ///   shed regardless of balance. Set high (e.g. 0.9) as a safety valve.
    /// - **KV spread** (`balance_token_usage_threshold`): the hottest engine is
    ///   materially more KV-saturated than the coldest, i.e. a cooler engine
    ///   exists to spill toward. This is the true balance signal for long-context
    ///   workloads, and — unlike request counts, which each gateway sees only
    ///   locally — it is invariant to the number of gateway replicas.
    /// - **count spread**: request-count dispersion (abs AND rel) over healthy
    ///   workers. Always evaluated, so high-count / low-KV imbalance is still
    ///   caught when KV looks even.
    /// Whether to abandon cache affinity for shortest-queue because the pool is
    /// imbalanced — by backend KV usage (overload ceiling or hot-vs-cool spread)
    /// or by request-count spread. `min_load`/`max_load` are the request-count
    /// bounds over the healthy workers, which `select_worker` gathers in its
    /// single worker pass (tests use the `imbalanced` helper to fold them).
    fn is_imbalanced(
        &self,
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
        min_load: usize,
        max_load: usize,
    ) -> bool {
        // KV-based triggers — need a load snapshot; both default 1.0 = disabled.
        if let Some((min_usage, max_usage)) =
            self.backend_token_usage_bounds(workers, healthy_indices)
        {
            // Overload: a single engine is critically saturated.
            if max_usage > f64::from(self.config.overload_token_usage_threshold) {
                return true;
            }
            // KV imbalance: a hot engine with a materially cooler home.
            if max_usage - min_usage > f64::from(self.config.balance_token_usage_threshold) {
                return true;
            }
        }

        // Count spread (abs AND rel) over healthy workers.
        max_load.saturating_sub(min_load) > self.config.balance_abs_threshold
            && (max_load as f32) > (min_load as f32 * self.config.balance_rel_threshold)
    }

    /// Min and max effective KV-cache utilization (0.0–1.0) across healthy workers,
    /// as `(min, max)`. Reads engine_stats + inflight speculation (see
    /// `effective_kv_used_tokens`) so the value reflects just-admitted requests
    /// without waiting for the next load poll. `None` when no healthy worker has a
    /// `max_model_len`/`max_total_tokens` label (→ caller relies on request-count spread).
    fn backend_token_usage_bounds(
        &self,
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
    ) -> Option<(f64, f64)> {
        let mut bounds: Option<(f64, f64)> = None;
        for &idx in healthy_indices {
            let worker = &workers[idx];
            let capacity = worker_kv_capacity(worker);
            if capacity <= 0 {
                continue;
            }
            let used = effective_kv_used_tokens(worker);
            let usage = (used as f64 / capacity as f64).clamp(0.0, 1.0);
            bounds = Some(match bounds {
                Some((min, max)) => (min.min(usage), max.max(usage)),
                None => (usage, usage),
            });
        }
        bounds
    }

    /// KV-capacity admission gate using live engine_stats + speculative inflight.
    ///
    /// Rejects when the candidate worker would queue (`num_waiting_reqs > 0`) or
    /// when the effective KV occupancy (snapshot + in-flight estimate) plus
    /// `new_request_tokens` would exceed the worker's `max_model_len` capacity.
    ///
    /// Returns `false` (admit) when the gate is disabled, when the worker has not
    /// yet reported any engine snapshot, or when no capacity label is set —
    /// degrading gracefully so the routing loop never deadlocks on missing data.
    fn admission_rejects(&self, worker: &Arc<dyn Worker>, new_request_tokens: i64) -> bool {
        if !self.config.enable_kv_admission_control {
            return false;
        }
        // No snapshot yet → admit (graceful degrade).
        if worker.engine_stats_timestamp_ms() == 0 {
            return false;
        }
        let stats = worker.engine_stats_arc();
        if stats.scheduler_stats.num_waiting_reqs > 0 {
            return true;
        }
        let capacity = worker_kv_capacity(worker);
        if capacity <= 0 {
            return false; // no capacity label → can't check, admit.
        }
        let threshold_tokens = (capacity as f64 * f64::from(self.config.kv_capacity_threshold).clamp(0.0, 1.0)) as i64;
        effective_kv_used_tokens(worker) + new_request_tokens > threshold_tokens
    }

    /// Least-loaded healthy worker that passes the admission gate for
    /// `new_request_tokens` of new KV (matched prefix = 0, conservative — used
    /// by fallback paths that have abandoned cache affinity). Replaces the
    /// unconditional first-healthy / random / min-load fallbacks when the gate
    /// is enabled, so a fully-gated-out request returns `None` (re-enqueued by
    /// the routing loop) instead of being force-routed to a saturated engine.
    /// Reuses the `(load, processed, idx)` ordering from `select_worker`.
    fn first_admitting_by_load(
        &self,
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
        new_request_tokens: i64,
    ) -> Option<usize> {
        let mut best: Option<(usize, usize, usize)> = None;
        for &idx in healthy_indices {
            if self.admission_rejects(&workers[idx], new_request_tokens) {
                continue;
            }
            let state = workers[idx].routing_state();
            let key = (state.load, state.processed, idx);
            match best {
                Some(b) if key >= b => {}
                _ => best = Some(key),
            }
        }
        best.map(|(_, _, idx)| idx)
    }

    /// Initialize the trees with worker URLs (used only during initial setup)
    /// Initializes both string trees (HTTP) and token trees (gRPC) for each model.
    pub fn init_workers(&self, workers: &[Arc<dyn Worker>]) {
        // Group workers by model
        let mut model_workers: HashMap<String, Vec<&Arc<dyn Worker>>> = HashMap::new();
        for worker in workers {
            let tree_key = normalize_model_key(worker.model_id());
            model_workers
                .entry(tree_key.to_string())
                .or_default()
                .push(worker);
        }

        // Initialize trees for each model (both string and token trees)
        for (tree_key, model_workers) in model_workers {
            // Initialize string tree (HTTP)
            let string_tree = self
                .string_trees
                .entry(tree_key.clone())
                .or_insert_with(|| Arc::new(Tree::new()));
            // Initialize token tree (gRPC)
            let token_tree = self
                .token_trees
                .entry(tree_key)
                .or_insert_with(|| Arc::new(TokenTree::new()));

            for worker in model_workers {
                string_tree.insert_text("", worker.url());
                token_tree.insert_tokens(&[], worker.url());
            }
        }
    }

    /// Add a single worker to the trees (incremental update)
    pub fn add_worker(&self, worker: &dyn Worker) {
        let tree_key = normalize_model_key(worker.model_id()).to_string();
        // Add to string tree (HTTP)
        let string_tree = self
            .string_trees
            .entry(tree_key.clone())
            .or_insert_with(|| Arc::new(Tree::new()));
        string_tree.insert_text("", worker.url());
        // Add to token tree (gRPC)
        let token_tree = self
            .token_trees
            .entry(tree_key)
            .or_insert_with(|| Arc::new(TokenTree::new()));
        token_tree.insert_tokens(&[], worker.url());
    }

    /// Add a worker by URL and model (for backward compatibility)
    pub fn add_worker_by_url(&self, url: &str, model_id: &str) {
        let model_id_string = model_id.to_string();
        // Add to string tree (HTTP)
        let string_tree = self
            .string_trees
            .entry(model_id_string.clone())
            .or_insert_with(|| Arc::new(Tree::new()));
        string_tree.insert_text("", url);
        // Add to token tree (gRPC)
        let token_tree = self
            .token_trees
            .entry(model_id_string)
            .or_insert_with(|| Arc::new(TokenTree::new()));
        token_tree.insert_tokens(&[], url);
    }

    /// Remove a worker from the trees
    ///
    /// Note: Currently a no-op. Stale entries are cleaned up by LRU eviction.
    /// Worker registry removes workers first, so routing will skip them anyway.
    /// TODO: Implement efficient remove_tenant in kv_index with reverse index.
    #[expect(
        clippy::unused_self,
        reason = "no-op stub; will use self once remove_tenant is implemented"
    )]
    pub fn remove_worker(&self, _worker: &dyn Worker) {
        // No-op: rely on LRU eviction to clean up stale entries
    }

    /// Remove a worker by URL (removes from all model trees for backward compatibility)
    ///
    /// Note: Currently a no-op. Stale entries are cleaned up by LRU eviction.
    /// TODO: Implement efficient remove_tenant in kv_index with reverse index.
    #[expect(
        clippy::unused_self,
        reason = "no-op stub; will use self once remove_tenant is implemented"
    )]
    pub fn remove_worker_by_url(&self, _url: &str) {
        // No-op: rely on LRU eviction to clean up stale entries
    }

    /// Run cache eviction to prevent unbounded growth
    pub fn evict_cache(&self, max_size: usize) {
        // Evict string trees (HTTP)
        for tree_ref in self.string_trees.iter() {
            let model_id = tree_ref.key();
            let tree = tree_ref.value();
            tree.evict_tenant_by_size(max_size);
            debug!(
                "String tree eviction for model {}, max_size: {}",
                model_id, max_size
            );
        }
        // Evict token trees (gRPC)
        for tree_ref in self.token_trees.iter() {
            let model_id = tree_ref.key();
            let tree = tree_ref.value();
            tree.evict_tenant_by_size(max_size);
            debug!(
                "Token tree eviction for model {}, max_size: {}",
                model_id, max_size
            );
        }
        // Evict hash index per model per tree kind. `max_size` is a
        // per-tree bound; clearing one model's overflow must not wipe
        // other models' still-valid metadata.
        for entry in self.hash_index.iter() {
            let per_model = entry.value();
            if per_model.string_tree.len() > max_size {
                per_model.string_tree.clear();
                debug!(
                    model_id = entry.key(),
                    "String hash index cleared (exceeded max_size: {})", max_size
                );
            }
            if per_model.token_tree.len() > max_size {
                per_model.token_tree.clear();
                debug!(
                    model_id = entry.key(),
                    "Token hash index cleared (exceeded max_size: {})", max_size
                );
            }
        }
    }

    /// Select worker with minimum load (used when load is imbalanced)
    /// Handles both HTTP (text-based) and gRPC (token-based) requests.
    fn select_worker_min_load(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
        healthy_indices: &[usize],
        min_load_idx: Option<usize>,
        model_id: &str,
        base_tokens: i64,
    ) -> Option<usize> {
        // Log load balancing trigger (only compute worker loads if debug enabled)
        if tracing::enabled!(tracing::Level::DEBUG) {
            let worker_loads: Vec<(&str, usize)> =
                workers.iter().map(|w| (w.url(), w.load())).collect();
            debug!("Load balancing triggered | workers: {:?}", worker_loads);
        }

        // Shortest queue when imbalanced. The min-load index is gathered upstream
        // in select_worker with the (load, processed_requests, idx) tie-break
        // from #1714 (spreads load when decode outpaces prefill).
        //
        // KV-capacity admission gate: the imbalanced path abandons cache
        // affinity, so the matched prefix is treated as 0 (conservative). If the
        // min-load worker would be saturated, fall back to the least-loaded
        // worker that still admits; if none admits, return None so the routing
        // loop re-enqueues the request rather than force-routing to a full
        // engine. The gate only applies to gRPC (token) requests — HTTP/text
        // requests have no token capacity signal.
        let gate_active = self.config.enable_kv_admission_control && info.tokens.is_some();
        let chosen_idx = match min_load_idx {
            Some(idx) if gate_active && self.admission_rejects(&workers[idx], base_tokens) => {
                self.first_admitting_by_load(workers, healthy_indices, base_tokens)?
            }
            Some(idx) => idx,
            None => return None,
        };

        // Score trace: imbalance channel abandoned cache affinity for
        // shortest-queue — label the channel + best/prev instance loads.
        if let Some(st) = info.score_trace.as_ref() {
            info!(
                target: SCORE_TRACE_TARGET,
                event = "route_decision",
                channel = "imbalanced_min_load",
                request_id = st.request_id,
                prompt_id = st.prompt_id,
                trigger = "count/kv spread imbalance",
                best = %format!(
                    "{}(load={})",
                    score_inst_label(st, workers, chosen_idx),
                    workers[chosen_idx].load()
                ),
                prev = %score_prev_load_label(st, workers),
                "min-load (imbalanced)"
            );
        }

        let worker_url = workers[chosen_idx].url();

        // Even in imbalanced mode, update the appropriate tree to maintain cache state
        // Prefer token tree for gRPC requests, fall back to string tree for HTTP
        if let Some(tokens) = info.tokens {
            // gRPC request: update token tree
            let tree = self
                .token_trees
                .get(model_id)
                .map(|entry| entry.value().clone());
            if let Some(tree) = tree {
                // We need the match result (the prior shared prefix) BEFORE the
                // insert so the hash_index stores only that bounded prefix, not
                // the full path that exists post-insert (32K tokens × 4 bytes ×
                // max_tree_size = multi-GB/model). `match_and_insert` resolves
                // the match against the pre-insert tree and inserts in the SAME
                // descent, so `result.matched_token_count` is the same prior
                // prefix length the standalone match returned. When we don't
                // populate the index, a plain insert (no match) suffices.
                if self.should_populate_hash_index() {
                    let result = tree.match_and_insert(tokens, worker_url);
                    let matched_prefix: Vec<u32> = tokens[..result.matched_token_count].to_vec();
                    self.hash_index
                        .entry(model_id.to_string())
                        .or_default()
                        .token_tree
                        .insert(kv_index::hash_token_path(tokens), matched_prefix);
                } else {
                    tree.insert_tokens(tokens, worker_url);
                }
            }
        } else if let Some(text) = info.request_text {
            // HTTP request: update string tree
            let tree = self
                .string_trees
                .get(model_id)
                .map(|entry| entry.value().clone());

            if let Some(tree) = tree {
                // Match BEFORE insert so the hash_index stores only the prior
                // shared prefix (~50-200 chars), not the full prompt (20KB+)
                // that exists post-insert. `match_and_insert` does both in a
                // single descent; `result.matched_char_count` is the same prior
                // prefix length the standalone match returned. When we don't
                // populate the index, a plain insert (no match) suffices.
                if self.should_populate_hash_index() {
                    let result = tree.match_and_insert(text, worker_url);
                    let matched_prefix: String =
                        text.chars().take(result.matched_char_count).collect();
                    let path_hash = kv_index::hash_node_path(text);
                    self.hash_index
                        .entry(model_id.to_string())
                        .or_default()
                        .string_tree
                        .insert(path_hash, matched_prefix);
                } else {
                    tree.insert_text(text, worker_url);
                }
            } else {
                debug!(
                    "Warning: No string tree found for model '{}', skipping cache update",
                    model_id
                );
            }
        }

        // Increment processed counter
        workers[chosen_idx].increment_processed();

        Some(chosen_idx)
    }
}

impl TreeHandle for CacheAwareV1Policy {
    fn apply_known_remote_insert(
        &self,
        model_id: &str,
        tree_kind: TreeKind,
        node_hash: u64,
        worker_url: &str,
    ) -> bool {
        // Normalize empty → UNKNOWN_MODEL_ID so lookups match the
        // key shape every populate site already uses.
        let model_id = normalize_model_key(model_id);
        let Some(model_entry) = self.hash_index.get(model_id) else {
            return false;
        };
        match tree_kind {
            TreeKind::String => {
                let Some(path) = model_entry.string_tree.get(&node_hash) else {
                    return false;
                };
                let Some(tree) = self.string_trees.get(model_id) else {
                    // Hash index entry without a corresponding
                    // tree means a populate site mutated
                    // `hash_index` without creating the tree
                    // (or eviction dropped the tree but left the
                    // index). Returning false here masks the
                    // invariant violation as a spurious repair
                    // request, so log loudly.
                    warn!(
                        model_id,
                        node_hash,
                        "string hash_index entry without matching string_trees entry; populate-site invariant violated",
                    );
                    return false;
                };
                tree.insert_text(path.value(), worker_url);
                true
            }
            TreeKind::Token => {
                let Some(tokens) = model_entry.token_tree.get(&node_hash) else {
                    return false;
                };
                let Some(tree) = self.token_trees.get(model_id) else {
                    warn!(
                        model_id,
                        node_hash,
                        "token hash_index entry without matching token_trees entry; populate-site invariant violated",
                    );
                    return false;
                };
                tree.insert_tokens(tokens.value(), worker_url);
                true
            }
        }
    }

    fn open_repair_stream(
        &self,
        model_id: &str,
        tree_kind: TreeKind,
    ) -> Option<Box<dyn Iterator<Item = RepairEntry> + Send>> {
        let model_id = normalize_model_key(model_id);
        match tree_kind {
            TreeKind::String => {
                let tree = self.string_trees.get(model_id)?.value().clone();
                Some(Box::new(tree.iter_entries().map(|(path, tenants)| {
                    RepairEntry::String { path, tenants }
                })))
            }
            TreeKind::Token => {
                let tree = self.token_trees.get(model_id)?.value().clone();
                Some(Box::new(tree.iter_entries().map(|(tokens, tenants)| {
                    RepairEntry::Token { tokens, tenants }
                })))
            }
        }
    }

    fn apply_repair_page(&self, page: &TreeRepairPage) -> usize {
        let model_id = normalize_model_key(&page.model_id);
        let mut applied: usize = 0;
        match page.tree_kind {
            TreeKind::String => {
                // Create the tree on first repair page if it
                // doesn't exist yet locally — repair is the
                // primary cold-start path for a fresh peer.
                let tree = self
                    .string_trees
                    .entry(model_id.to_string())
                    .or_insert_with(|| Arc::new(Tree::new()))
                    .clone();
                for entry in &page.entries {
                    match entry {
                        RepairEntry::String { path, tenants } => {
                            for (tenant, _epoch) in tenants {
                                tree.insert_text(path, tenant);
                            }
                            self.hash_index
                                .entry(model_id.to_string())
                                .or_default()
                                .string_tree
                                .insert(kv_index::hash_node_path(path), path.clone());
                            applied += 1;
                        }
                        RepairEntry::Token { .. } => {
                            warn!(
                                model_id,
                                session_id = %page.session_id,
                                page_index = page.page_index,
                                "RepairEntry variant mismatch: page kind=String but entry kind=Token; skipping",
                            );
                        }
                    }
                }
            }
            TreeKind::Token => {
                let tree = self
                    .token_trees
                    .entry(model_id.to_string())
                    .or_insert_with(|| Arc::new(TokenTree::new()))
                    .clone();
                for entry in &page.entries {
                    match entry {
                        RepairEntry::Token { tokens, tenants } => {
                            for (tenant, _epoch) in tenants {
                                tree.insert_tokens(tokens, tenant);
                            }
                            self.hash_index
                                .entry(model_id.to_string())
                                .or_default()
                                .token_tree
                                .insert(kv_index::hash_token_path(tokens), tokens.clone());
                            applied += 1;
                        }
                        RepairEntry::String { .. } => {
                            warn!(
                                model_id,
                                session_id = %page.session_id,
                                page_index = page.page_index,
                                "RepairEntry variant mismatch: page kind=Token but entry kind=String; skipping",
                            );
                        }
                    }
                }
            }
        }
        applied
    }
}

impl LoadBalancingPolicy for CacheAwareV1Policy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        let request_text = info.request_text;
        let request_tokens = info.tokens;

        // Single O(workers) gather: read each worker once via routing_state()
        // (status + load + processed under one ArcSwap guard), replacing the
        // former separate passes whose per-worker guard traffic dominated routing
        // CPU at scale. Collects healthy indices, load min/max, and the min-load
        // index; cache-hit tenant lookup is a hash-free scan over healthy_indices.
        let mut healthy_indices: Vec<usize> = Vec::with_capacity(workers.len());
        let mut min_load = usize::MAX;
        let mut max_load = 0usize;
        // Min-load worker, (load, processed_requests, idx) tie-break (#1714);
        // `processed` rides the same guard as `load`, so it is free here.
        let mut min_key: Option<(usize, usize, usize)> = None;
        let mut min_load_idx: Option<usize> = None;
        for (idx, worker) in workers.iter().enumerate() {
            let state = worker.routing_state();
            if state.healthy && state.can_execute {
                healthy_indices.push(idx);
                min_load = min_load.min(state.load);
                max_load = max_load.max(state.load);
                let key = (state.load, state.processed, idx);
                match min_key {
                    Some(best) if key >= best => {}
                    _ => {
                        min_key = Some(key);
                        min_load_idx = Some(idx);
                    }
                }
            }
        }

        if healthy_indices.is_empty() {
            return None;
        }
        let min_load = if min_load == usize::MAX { 0 } else { min_load };

        // Determine the model for this set of workers (router pre-filters by model)
        // All workers should be from the same model
        let model_id = normalize_model_key(workers[healthy_indices[0]].model_id());

        // Full request token footprint (prompt + response-so-far) used by the
        // KV-capacity admission gate. The cache-hit prefix is subtracted at each
        // candidate, so only genuinely-new KV is counted against capacity.
        let base_tokens: i64 = info
            .tokens
            .map_or(0, |t| t.len() as i64)
            + info.response_token_count.unwrap_or(0) as i64;

        // Abandon cache affinity for shortest-queue when the pool is imbalanced —
        // by request count (using the loads already gathered above), or (for
        // long-context workloads) by backend KV usage.
        if self.is_imbalanced(workers, &healthy_indices, min_load, max_load) {
            return self.select_worker_min_load(
                workers,
                info,
                &healthy_indices,
                min_load_idx,
                model_id,
                base_tokens,
            );
        }

        // Cache-aware routing when balanced — three types (mutually exclusive):
        //   1. Event-driven: PositionalIndexer overlap scoring (gRPC + KV events)
        //   2. Approximate token tree: TokenTree prefix matching (gRPC, no events)
        //   3. Approximate string tree: Tree prefix matching (HTTP)
        let st = info.score_trace.as_ref();
        if let Some(tokens) = request_tokens {
            if self.has_event_indexer(model_id) {
                self.select_worker_event_driven(
                    workers,
                    tokens,
                    &healthy_indices,
                    min_load_idx,
                    model_id,
                    base_tokens,
                    st,
                )
            } else {
                self.select_worker_with_tokens(
                    workers,
                    tokens,
                    &healthy_indices,
                    min_load_idx,
                    model_id,
                    base_tokens,
                    st,
                )
            }
        } else {
            let text = request_text.unwrap_or("");
            self.select_worker_with_text(
                workers,
                text,
                &healthy_indices,
                min_load_idx,
                model_id,
                st,
            )
        }
    }

    fn on_request_complete(&self, worker_url: &str, success: bool) {
        // Could track success rates per worker for more intelligent routing
        if !success {
            // Optionally reduce affinity for failed requests
            tracing::debug!(
                "Request to {} completed with success={}",
                worker_url,
                success
            );
        }
    }

    fn name(&self) -> &'static str {
        "cache_aware_v1"
    }

    fn on_version_bump(&self, worker_url: &str) {
        // Model weights changed: the engine cleared its KV cache for the old
        // version, so the cache-affinity prefixes we recorded for this instance
        // are stale (they would mislead cache-hit routing and any KV-marginal
        // estimate). Drop the instance's tenant from every per-model tree.
        let tenant: Arc<str> = Arc::from(worker_url);
        for tree in self.token_trees.iter() {
            tree.value().evict_tenant(&tenant, 0);
        }
        for tree in self.string_trees.iter() {
            tree.value().remove_tenant_all(&tenant);
        }
    }

    fn needs_request_text(&self) -> bool {
        true // Cache-aware policy needs request text for cache affinity
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// Private helper methods for select_worker
impl CacheAwareV1Policy {
    /// Check if an event-driven indexer exists with data for this model.
    /// Returns false when the indexer is empty (startup, reconnect) so
    /// routing falls through to the approximate token tree instead of
    /// taking the event-driven path with no data and landing on min-load.
    fn has_event_indexer(&self, model_id: &str) -> bool {
        let guard = self.kv_monitor.read();
        guard
            .as_ref()
            .and_then(|m| m.get_indexer(model_id))
            .is_some_and(|indexer| indexer.current_size() > 0)
    }

    /// Event-driven routing: multi-tier overlap scoring (Type 1).
    ///
    /// Queries GPU and LMCache tiers independently, computing a weighted score
    /// per worker. A GPU hit (zero reload cost) outranks an LMCache hit (needs
    /// reload) of the same depth via configurable weights.
    fn select_worker_event_driven(
        &self,
        workers: &[Arc<dyn Worker>],
        tokens: &[u32],
        healthy_indices: &[usize],
        min_load_idx: Option<usize>,
        model_id: &str,
        base_tokens: i64,
        st: Option<&ScoreTraceCtx>,
    ) -> Option<usize> {
        let guard = self.kv_monitor.read();
        let monitor = guard.as_ref()?;
        let tiered_indexer = monitor.get_indexer(model_id)?;

        if let Some(idx) = self.score_overlap_tiered(
            workers,
            tokens,
            healthy_indices,
            &tiered_indexer,
            base_tokens,
            st,
        ) {
            return Some(idx);
        }

        // No cache overlap — min-load fallback. Apply the admission gate
        // (matched=0; no overlap means no resident prefix to subtract). If the
        // min-load worker is saturated, fall back to the least-loaded admitting
        // worker; if none admits, return None (re-enqueue).
        let min_idx = match min_load_idx {
            Some(idx)
                if self.config.enable_kv_admission_control
                    && self.admission_rejects(&workers[idx], base_tokens) =>
            {
                self.first_admitting_by_load(workers, healthy_indices, base_tokens)?
            }
            Some(idx) => idx,
            None => return None,
        };
        debug!(
            worker = workers[min_idx].url(),
            model_id, "Event-driven routing: no overlap, min-load fallback"
        );
        // Score trace: event-driven path found no cache overlap for any
        // candidate, so it fell through to shortest-queue — label the channel.
        if let Some(st) = st {
            info!(
                target: SCORE_TRACE_TARGET,
                event = "route_decision",
                channel = "ev_min_load_fallback",
                request_id = st.request_id,
                prompt_id = st.prompt_id,
                best = %format!(
                    "{}(load={})",
                    score_inst_label(st, workers, min_idx),
                    workers[min_idx].load()
                ),
                prev = %score_prev_load_label(st, workers),
                "event-driven: no overlap, min-load fallback"
            );
        }
        workers[min_idx].increment_processed();
        Some(min_idx)
    }

    /// Score healthy workers by weighted multi-tier overlap and select the best.
    ///
    /// Each tier (GPU, LMCache) is queried independently with its own learned
    /// block size. A worker's score is:
    ///   `gpu_overlap_weight * gpu_score + lmcache_overlap_weight * lmcache_score`
    ///
    /// Returns `Some(idx)` if at least one worker has a positive weighted score.
    fn score_overlap_tiered(
        &self,
        workers: &[Arc<dyn Worker>],
        tokens: &[u32],
        healthy_indices: &[usize],
        indexer: &TieredIndexer,
        base_tokens: i64,
        st: Option<&ScoreTraceCtx>,
    ) -> Option<usize> {
        let mut weighted: Vec<f64> = vec![0.0; workers.len()];
        let mut tree_sizes: Vec<usize> = vec![0; workers.len()];
        // GPU-resident matched prefix per worker (blocks * gpu block_size). Only
        // the GPU tier reduces the new-KV the engine must allocate — an LMCache
        // hit still has to be loaded onto the GPU, consuming KV capacity — so
        // the admission gate subtracts only GPU-tier matched tokens.
        let mut gpu_matched: Vec<usize> = vec![0; workers.len()];
        // Raw per-tier overlap block counts per worker, retained for the
        // `score_trace.log` breakdown (matched tokens = raw * block_size,
        // weighted contribution = weight * raw).
        let mut gpu_raw: Vec<u32> = vec![0; workers.len()];
        let mut lmcache_raw: Vec<u32> = vec![0; workers.len()];
        let mut gpu_block: usize = self.config.block_size;
        let mut lmcache_block: usize = self.config.block_size;

        for (tier, weight) in [
            (Tier::GPU, self.config.gpu_overlap_weight),
            (Tier::Lmcache, self.config.lmcache_overlap_weight),
        ] {
            if weight <= 0.0 {
                continue;
            }
            let block_size = indexer.block_size(tier).unwrap_or(self.config.block_size);
            if matches!(tier, Tier::GPU) {
                gpu_block = block_size;
            } else {
                lmcache_block = block_size;
            }
            let content_hashes = compute_request_content_hashes(tokens, block_size);
            if content_hashes.is_empty() {
                continue;
            }
            let pi = indexer.tier(tier);
            let overlap = pi.find_matches(&content_hashes, false);
            if overlap.scores.is_empty() {
                continue;
            }
            for &idx in healthy_indices {
                if let Some(wid) = pi.worker_id(workers[idx].url()) {
                    if let Some(&score) = overlap.scores.get(&wid) {
                        weighted[idx] += weight * f64::from(score);
                        if matches!(tier, Tier::GPU) {
                            gpu_matched[idx] = (score as usize) * block_size;
                            gpu_raw[idx] = score;
                        } else {
                            lmcache_raw[idx] = score;
                        }
                    }
                    tree_sizes[idx] += overlap.tree_sizes.get(&wid).copied().unwrap_or(0);
                }
            }
        }

        let best_idx = healthy_indices
            .iter()
            .copied()
            .filter(|&idx| {
                weighted[idx] > 0.0
                    && !self.admission_rejects(
                        &workers[idx],
                        base_tokens.saturating_sub(gpu_matched[idx] as i64),
                    )
            })
            .max_by(|&a, &b| {
                let load_a = workers[a].load();
                let load_b = workers[b].load();
                weighted[a]
                    .total_cmp(&weighted[b])
                    .then(load_b.cmp(&load_a))
                    .then(tree_sizes[b].cmp(&tree_sizes[a]))
            });

        // Score trace: emit the per-tier GPU/LMCache breakdown for the chosen
        // (best) instance and the request's previous-round instance. Only when a
        // positive-overlap winner exists; the no-overlap case is logged by the
        // caller's min-load fallback instead (one line per decision).
        if let (Some(st), Some(best)) = (st, best_idx) {
            self.emit_tiered_score_trace(
                st,
                workers,
                &weighted,
                &gpu_raw,
                &lmcache_raw,
                gpu_block,
                lmcache_block,
                best,
            );
        }

        let best_idx = best_idx?;

        debug!(
            worker = workers[best_idx].url(),
            score = weighted[best_idx],
            "Event-driven routing: weighted multi-tier overlap match"
        );
        workers[best_idx].increment_processed();
        Some(best_idx)
    }

    /// Emit one `score_trace.log` line for the event-driven tiered channel,
    /// showing the GPU + LMCache score breakdown (raw block count, matched
    /// tokens, weight, weighted contribution, total) for the chosen (`best`)
    /// instance and the request's previous-round (`prev`) instance. `best` and
    /// `prev` may be the same instance (identical numbers).
    #[allow(clippy::too_many_arguments)]
    fn emit_tiered_score_trace(
        &self,
        st: &ScoreTraceCtx,
        workers: &[Arc<dyn Worker>],
        weighted: &[f64],
        gpu_raw: &[u32],
        lmcache_raw: &[u32],
        gpu_block: usize,
        lmcache_block: usize,
        best_idx: usize,
    ) {
        let gpu_w = self.config.gpu_overlap_weight;
        let lmc_w = self.config.lmcache_overlap_weight;
        let group = |idx: usize| -> String {
            let gr = gpu_raw[idx];
            let lr = lmcache_raw[idx];
            format!(
                "{}{{gpu_raw={gr} gpu_tok={gt} gpu_wtd={gwt:.3} lmc_raw={lr} lmc_tok={lt} lmc_wtd={lwt:.3} total={tot:.3}}}",
                score_inst_label(st, workers, idx),
                gt = gr as usize * gpu_block,
                gwt = gpu_w * f64::from(gr),
                lt = lr as usize * lmcache_block,
                lwt = lmc_w * f64::from(lr),
                tot = weighted[idx],
            )
        };
        let prev = match st.prev_instance_id {
            None => "none(first_turn)".to_string(),
            Some(prev) => match score_prev_idx(st, workers) {
                Some(idx) => group(idx),
                None => format!("inst{prev}(not_a_candidate)"),
            },
        };
        info!(
            target: SCORE_TRACE_TARGET,
            event = "score",
            channel = "event_driven_tiered",
            request_id = st.request_id,
            prompt_id = st.prompt_id,
            gpu_weight = gpu_w,
            lmcache_weight = lmc_w,
            gpu_block_size = gpu_block,
            lmcache_block_size = lmcache_block,
            best = %group(best_idx),
            prev = %prev,
            "kv-aware tiered score"
        );
    }

    /// Emit one `score_trace.log` line for an approximate-tree channel (token or
    /// string). These channels model no cache tiers, so there is no GPU/LMCache
    /// breakdown — the `channel` name + `match_rate`/`hit` explain the decision.
    fn emit_approx_score_trace(
        &self,
        st: &ScoreTraceCtx,
        workers: &[Arc<dyn Worker>],
        channel: &'static str,
        selected: usize,
        hit: bool,
        match_rate: f32,
    ) {
        info!(
            target: SCORE_TRACE_TARGET,
            event = "route_decision",
            channel,
            request_id = st.request_id,
            prompt_id = st.prompt_id,
            match_rate,
            cache_threshold = self.config.cache_threshold,
            hit,
            best = %format!(
                "{}(load={})",
                score_inst_label(st, workers, selected),
                workers[selected].load()
            ),
            prev = %score_prev_load_label(st, workers),
            "approx-tree decision"
        );
    }

    /// Select worker using token-based tree (gRPC path)
    fn select_worker_with_tokens(
        &self,
        workers: &[Arc<dyn Worker>],
        tokens: &[u32],
        healthy_indices: &[usize],
        min_load_idx: Option<usize>,
        model_id: &str,
        base_tokens: i64,
        st: Option<&ScoreTraceCtx>,
    ) -> Option<usize> {
        let tree = self
            .token_trees
            .get(model_id)
            .map(|entry| entry.value().clone());

        if let Some(tree) = tree {
            // Single tree descent: match, pick the worker from the match
            // result, then insert for it — replacing the former
            // match_prefix_with_counts + insert_tokens pair (two full descents
            // over the same prefix). The selection closure runs once, after the
            // match, mirroring the previous branch exactly:
            //   * cache hit  (match_rate > threshold): route to the matched
            //     worker if it is still healthy — insert for it;
            //   * cache miss (match_rate <= threshold): route to the least-loaded
            //     worker — insert for it;
            //   * matched worker gone/unhealthy: select nothing and DON'T insert
            //     (closure returns None), falling back to first-healthy below.
            //
            // KV-capacity admission gate: a candidate is dropped (closure returns
            // None, skipping the insert) when it would queue or overflow. The
            // cache-hit prefix is already resident, so only `base - matched` new
            // KV is counted. Dropped candidates fall through to the gated
            // fallback below.
            let mut selected_idx: Option<usize> = None;
            // Captured for score_trace.log (set inside the match closure).
            let mut match_rate: f32 = 0.0;
            let mut cache_hit = false;
            let result = tree.match_and_insert_with(tokens, |result| {
                match_rate = if result.input_token_count == 0 {
                    0.0
                } else {
                    result.matched_token_count as f32 / result.input_token_count as f32
                };
                cache_hit = match_rate > self.config.cache_threshold;

                let candidate = if cache_hit {
                    // Cache hit: scan healthy_indices for the tenant (hash-free;
                    // url() is cheap). "Healthy" excludes circuit-broken workers,
                    // so a CB-tripped tenant falls through to min-load (intended).
                    let tenant_url: &str = &result.tenant;
                    healthy_indices
                        .iter()
                        .copied()
                        .find(|&idx| workers[idx].url() == tenant_url)
                } else {
                    min_load_idx
                };

                let matched = if cache_hit { result.matched_token_count } else { 0 };
                let new_tokens = base_tokens.saturating_sub(matched as i64);
                selected_idx = candidate
                    .filter(|&idx| !self.admission_rejects(&workers[idx], new_tokens));

                // Insert for the selected worker (None => no insert, exactly
                // like the old `if let Some(idx)` guard around insert_tokens).
                selected_idx.map(|idx| workers[idx].url())
            });

            if let Some(idx) = selected_idx {
                // Record hash(full_tokens)→matched_prefix tokens.
                // The hash key matches what sync_tree_operation
                // sends on the wire (hash of full sequence). The
                // VALUE is only the matched prefix — not the full
                // sequence (32K tokens × 4 bytes = 128 KB worst
                // case). The `TreeHandle` impl consults this map
                // per incoming token delta, so maintain it
                // alongside the tree. Mirrors the string side at
                // the analogous block; reuses the match `result`
                // returned by match_and_insert_with.
                if self.should_populate_hash_index() {
                    let matched_prefix: Vec<u32> = tokens[..result.matched_token_count].to_vec();
                    self.hash_index
                        .entry(model_id.to_string())
                        .or_default()
                        .token_tree
                        .insert(kv_index::hash_token_path(tokens), matched_prefix);
                }
                if let Some(st) = st {
                    self.emit_approx_score_trace(
                        st, workers, "approx_token_tree", idx, cache_hit, match_rate,
                    );
                }
                workers[idx].increment_processed();
                return Some(idx);
            }

            // Selected worker no longer exists / unhealthy, OR was rejected by
            // the admission gate. When the gate is enabled, pick the
            // least-loaded admitting worker (inserting for it so cache state
            // tracks the routing) and return None if none admits — the routing
            // loop re-enqueues. When the gate is disabled, preserve the original
            // first-healthy fallback (no insert; stale entries age out via LRU).
            if self.config.enable_kv_admission_control {
                let idx = self.first_admitting_by_load(workers, healthy_indices, base_tokens)?;
                tree.insert_tokens(tokens, workers[idx].url());
                if let Some(st) = st {
                    self.emit_approx_score_trace(
                        st, workers, "approx_token_tree_gate_fallback", idx, cache_hit, match_rate,
                    );
                }
                workers[idx].increment_processed();
                Some(idx)
            } else {
                let idx = healthy_indices.first().copied();
                if let (Some(st), Some(idx)) = (st, idx) {
                    self.emit_approx_score_trace(
                        st, workers, "approx_token_tree_stale_fallback", idx, cache_hit, match_rate,
                    );
                }
                idx
            }
        } else {
            debug!(
                "Warning: No token tree found for model '{}', using random worker selection",
                model_id
            );
            if self.config.enable_kv_admission_control {
                let idx = self.first_admitting_by_load(workers, healthy_indices, base_tokens)?;
                if let Some(st) = st {
                    self.emit_approx_score_trace(
                        st, workers, "approx_token_tree_no_tree", idx, false, 0.0,
                    );
                }
                workers[idx].increment_processed();
                Some(idx)
            } else {
                let mut rng = rand::rng();
                let random_idx = rng.random_range(0..healthy_indices.len());
                let idx = healthy_indices[random_idx];
                if let Some(st) = st {
                    self.emit_approx_score_trace(
                        st, workers, "approx_token_tree_no_tree", idx, false, 0.0,
                    );
                }
                Some(idx)
            }
        }
    }

    /// Select worker using string-based tree (HTTP path)
    fn select_worker_with_text(
        &self,
        workers: &[Arc<dyn Worker>],
        text: &str,
        healthy_indices: &[usize],
        min_load_idx: Option<usize>,
        model_id: &str,
        st: Option<&ScoreTraceCtx>,
    ) -> Option<usize> {
        let tree = self
            .string_trees
            .get(model_id)
            .map(|entry| entry.value().clone());

        if let Some(tree) = tree {
            // Single tree descent: match, pick the worker from the match result,
            // then insert for it — replacing the former match_prefix_with_counts
            // + insert_text pair. Selection logic is unchanged (see the token
            // path for the per-branch rationale).
            let mut selected_idx: Option<usize> = None;
            // Captured for score_trace.log (set inside the match closure).
            let mut match_rate: f32 = 0.0;
            let mut cache_hit = false;
            let result = tree.match_and_insert_with(text, |result| {
                match_rate = if result.input_char_count == 0 {
                    0.0
                } else {
                    result.matched_char_count as f32 / result.input_char_count as f32
                };
                cache_hit = match_rate > self.config.cache_threshold;

                selected_idx = if cache_hit {
                    // Cache hit: scan healthy_indices for the tenant (hash-free;
                    // url() is cheap). "Healthy" excludes circuit-broken workers, so
                    // a CB-tripped tenant falls through to min-load (intended).
                    let tenant_url: &str = &result.tenant;
                    healthy_indices
                        .iter()
                        .copied()
                        .find(|&idx| workers[idx].url() == tenant_url)
                } else {
                    min_load_idx
                };

                // Insert for the selected worker (None => no insert, exactly
                // like the old `if let Some(idx)` guard around insert_text).
                selected_idx.map(|idx| workers[idx].url())
            });

            if let Some(idx) = selected_idx {
                // Record hash(full_text)→matched_prefix for mesh tenant delta
                // resolution. The hash key matches what sync_tree_operation sends
                // on the wire (hash of full text). The VALUE is only the matched
                // prefix (~50-200 chars), not the full prompt (20KB+). When a
                // remote delta arrives, we look up the hash and call
                // insert_text(matched_prefix, worker) which routes to the same
                // tree node. This keeps the index memory-bounded.
                if self.should_populate_hash_index() {
                    let matched_prefix: String =
                        text.chars().take(result.matched_char_count).collect();
                    let path_hash = kv_index::hash_node_path(text);
                    self.hash_index
                        .entry(model_id.to_string())
                        .or_default()
                        .string_tree
                        .insert(path_hash, matched_prefix);
                }

                if let Some(st) = st {
                    self.emit_approx_score_trace(
                        st, workers, "approx_string_tree", idx, cache_hit, match_rate,
                    );
                }
                workers[idx].increment_processed();
                return Some(idx);
            }

            // Selected worker no longer exists or unhealthy - fall back to first healthy
            // Stale entries will be cleaned up by LRU eviction
            let idx = healthy_indices.first().copied();
            if let (Some(st), Some(idx)) = (st, idx) {
                self.emit_approx_score_trace(
                    st, workers, "approx_string_tree_stale_fallback", idx, cache_hit, match_rate,
                );
            }
            idx
        } else {
            debug!(
                "Warning: No string tree found for model '{}', using random worker selection",
                model_id
            );
            let mut rng = rand::rng();
            let random_idx = rng.random_range(0..healthy_indices.len());
            let idx = healthy_indices[random_idx];
            if let Some(st) = st {
                self.emit_approx_score_trace(
                    st, workers, "approx_string_tree_no_tree", idx, false, 0.0,
                );
            }
            Some(idx)
        }
    }
}

impl Default for CacheAwareV1Policy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::{HealthCheckConfig, SchedulerLoadSnapshot, WorkerLoadResponse};
    use std::sync::Arc;
    use tokio::sync::watch;

    use super::*;
    use crate::policies::kv_admits;
    use crate::worker::{BasicWorkerBuilder, WorkerType};

    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    /// Healthy workers (health checks disabled) for the given URLs.
    fn make_workers(urls: &[&str]) -> Vec<Arc<dyn Worker>> {
        urls.iter()
            .map(|u| {
                Arc::new(
                    BasicWorkerBuilder::new(*u)
                        .worker_type(WorkerType::Regular)
                        .health_config(no_health_check())
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect()
    }

    /// One DP rank reporting the given KV state. `token_usage` is derived so
    /// `is_imbalanced`'s KV triggers stay quiet unless we want them loud.
    fn kv_snap(num_waiting: i32, used: i64, max: i64) -> WorkerLoadResponse {
        let usage = if max > 0 {
            (used as f64) / (max as f64)
        } else {
            0.0
        };
        WorkerLoadResponse {
            loads: vec![SchedulerLoadSnapshot {
                num_waiting_reqs: num_waiting,
                num_used_tokens: used as i32,
                max_total_num_tokens: max as i32,
                token_usage: usage,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    /// Inject per-worker (by index) KV snapshots; bind the sender to keep the
    /// watch channel open.
    fn inject_snaps(
        policy: &CacheAwareV1Policy,
        workers: &[Arc<dyn Worker>],
        snaps: &[WorkerLoadResponse],
    ) -> watch::Sender<HashMap<String, WorkerLoadResponse>> {
        let map: HashMap<String, WorkerLoadResponse> = workers
            .iter()
            .zip(snaps)
            .map(|(w, s)| (w.url().to_string(), s.clone()))
            .collect();
        let (tx, rx) = watch::channel(map);
        policy.set_load_receiver(Some(rx));
        tx
    }

    fn gated_policy() -> CacheAwareV1Policy {
        CacheAwareV1Policy::with_config(CacheAwareConfig {
            enable_kv_admission_control: true,
            // Disable imbalance KV triggers so tests control which path runs;
            // count-spread still works when we set worker.load() directly.
            balance_token_usage_threshold: 1.0,
            overload_token_usage_threshold: 1.0,
            ..Default::default()
        })
    }

    // ---- pure predicate ----

    #[test]
    fn kv_admits_reduces_to_single_rank() {
        // waiting>0 => reject.
        assert!(!kv_admits(&kv_snap(1, 0, 100), 10));
        // fits => admit.
        assert!(kv_admits(&kv_snap(0, 0, 100), 10));
        // overflow => reject.
        assert!(!kv_admits(&kv_snap(0, 95, 100), 10));
        // exact fit (==) => admit.
        assert!(kv_admits(&kv_snap(0, 90, 100), 10));
        // empty snapshot => admit (no data, avoid deadlock).
        assert!(kv_admits(&WorkerLoadResponse::default(), 10));
    }

    // ---- shared admission gate (LoadBalancingPolicy::admits default) ----

    #[test]
    fn admission_gate_count_cap_and_waiting_switch() {
        use crate::policies::{AdmissionGateConfig, RequestNumBalancePolicy};
        use crate::worker::EngineStats;

        let policy = RequestNumBalancePolicy::new();
        let cfg = AdmissionGateConfig {
            max_concurrent_seqs_per_instance: 3,
            reject_on_waiting: true,
        };
        let info = SelectWorkerInfo {
            request_text: None,
            tokens: None,
            headers: None,
            hash_ring: None,
            priority_groups: None,
            response_token_count: None,
            score_trace: None,
            leg: crate::policies::WorkerLeg::Single,
        };

        // No snapshot yet (timestamp 0) and load 0 < cap => admit (count cap only).
        let w: Arc<dyn Worker> = make_workers(&["http://gate-a:8000"]).remove(0);
        assert!(policy.admits(&w, &info, &cfg));

        // Count cap: load reaches the cap => reject, even without a snapshot.
        for _ in 0..3 {
            w.increment_load();
        }
        assert!(!policy.admits(&w, &info, &cfg));

        // Neutered gate (count cap 0, no waiting gate) => always admit.
        let off = AdmissionGateConfig {
            max_concurrent_seqs_per_instance: 0,
            reject_on_waiting: false,
        };
        assert!(policy.admits(&w, &info, &off));

        // Waiting switch ON: a fresh worker whose snapshot reports any waiting => reject.
        let w2: Arc<dyn Worker> = make_workers(&["http://gate-b:8000"]).remove(0);
        let mut stats = EngineStats::default();
        stats.scheduler_stats.num_waiting_reqs = 1;
        w2.update_engine_stats(stats, 0);
        assert!(!policy.admits(&w2, &info, &cfg));

        // Waiting switch OFF: same worker with waiting=1 => admit (ignore queue).
        let no_wait_gate = AdmissionGateConfig {
            reject_on_waiting: false,
            ..cfg
        };
        assert!(policy.admits(&w2, &info, &no_wait_gate));
    }

    #[test]
    fn kv_admits_any_rank_dp() {
        // One full rank, one free rank => admit (request lands on the free one).
        let mut load = WorkerLoadResponse::default();
        load.loads = vec![kv_snap(1, 99, 100).loads[0].clone(), kv_snap(0, 0, 100).loads[0].clone()];
        assert!(kv_admits(&load, 10));
        // Both ranks waiting => reject.
        let mut load = WorkerLoadResponse::default();
        load.loads = vec![kv_snap(1, 0, 100).loads[0].clone(), kv_snap(1, 0, 100).loads[0].clone()];
        assert!(!kv_admits(&load, 10));
    }

    // ---- select_worker integration ----

    #[test]
    fn gate_disabled_preserves_always_route() {
        // Gate off: even with every engine saturated, a worker is still chosen.
        let policy = CacheAwareV1Policy::with_config(CacheAwareConfig {
            enable_kv_admission_control: false,
            ..Default::default()
        });
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);
        let _tx = inject_snaps(
            &policy,
            &workers,
            &[kv_snap(1, 99, 100), kv_snap(1, 99, 100)],
        );
        let tokens: Vec<u32> = (1..=32).collect();
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&tokens),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(idx < 2);
    }

    #[test]
    fn gate_rejects_all_returns_none() {
        // Gate on, every engine has a waiting queue => no candidate admits =>
        // None (routing loop re-enqueues). KV usage kept low so count/KV
        // imbalance triggers stay quiet and the token-tree path is exercised.
        let policy = gated_policy();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);
        workers[0].update_engine_stats(engine_stats_with_waiting(1), 0);
        workers[1].update_engine_stats(engine_stats_with_waiting(1), 0);
        let tokens: Vec<u32> = (1..=32).collect();
        let idx = policy.select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&tokens),
                ..Default::default()
            },
        );
        assert!(idx.is_none(), "all-saturated pool must return None, got {idx:?}");
    }

    #[test]
    fn gate_prefix_subtraction_admits_cache_hit() {
        // base = 32 tokens. Worker A already holds the full prefix (cache hit =>
        // matched=32), so new KV = 0. Without subtraction A would overflow
        // (95+32>100); with subtraction it admits (95+0<=100). Assert A wins.
        let policy = gated_policy();
        // Workers need a capacity label for the KV overflow check.
        let workers = vec![
            make_worker_with_capacity("http://w1:8000", 100),
            make_worker_with_capacity("http://w2:8000", 100),
        ];
        policy.init_workers(&workers);
        // First route with no engine stats (timestamp=0 → always admit): seeds tree on A.
        let tokens: Vec<u32> = (1..=32).collect();
        let first = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&tokens),
                    ..Default::default()
                },
            )
            .unwrap();
        // Now set A to 95 tokens used, no waiting. B stays unconstrained.
        // 95 + 32 (full request) > 100, but 95 + 0 (matched prefix subtracted) <= 100.
        workers[first].update_engine_stats(engine_stats_with_tokens(95, 5), 0);
        let second = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&tokens),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            first, second,
            "cache-hit worker must be re-selected thanks to prefix subtraction"
        );
    }

    #[test]
    fn gate_min_load_fallback_picks_admitting_worker() {
        // Force count-spread imbalance (A load 20, B load 0) so the min-load
        // path runs. B (the min-load pick) is saturated; A has room. The gate
        // must reject B and fall back to A.
        let policy = gated_policy();
        let w = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&w);
        for _ in 0..20 {
            w[0].increment_load();
        }
        // B saturated (waiting); A has no snapshot → always admits.
        w[1].update_engine_stats(engine_stats_with_waiting(1), 0);
        let tokens: Vec<u32> = (1..=32).collect();
        let idx = policy
            .select_worker(
                &w,
                &SelectWorkerInfo {
                    tokens: Some(&tokens),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx, 0, "saturated min-load worker rejected; A (room) selected");
    }

    #[test]
    fn gate_all_min_load_saturated_returns_none() {
        // Imbalanced (count spread) and BOTH workers saturated => min-load path
        // rejects both => None.
        let policy = gated_policy();
        let w = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&w);
        for _ in 0..20 {
            w[0].increment_load();
        }
        w[0].update_engine_stats(engine_stats_with_waiting(1), 0);
        w[1].update_engine_stats(engine_stats_with_waiting(1), 0);
        let tokens: Vec<u32> = (1..=32).collect();
        let idx = policy.select_worker(
            &w,
            &SelectWorkerInfo {
                tokens: Some(&tokens),
                ..Default::default()
            },
        );
        assert!(idx.is_none(), "all-saturated imbalanced pool must return None");
    }

    #[test]
    fn gate_missing_snapshot_admits() {
        // Only A has a snapshot (saturated); B has none => B admits (no data).
        let policy = gated_policy();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);
        // A: waiting=1 → rejected. B: no engine_stats (timestamp=0) → always admits.
        workers[0].update_engine_stats(engine_stats_with_waiting(1), 0);
        let tokens: Vec<u32> = (1..=32).collect();
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&tokens),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx, 1, "worker without a snapshot admits (graceful no-data)");
    }

    #[test]
    fn gate_string_path_skipped() {
        // HTTP/text requests carry no tokens, so the gate is skipped and a
        // saturated pool still routes (string-tree path unchanged).
        let policy = gated_policy();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);
        let _tx = inject_snaps(
            &policy,
            &workers,
            &[kv_snap(1, 99, 100), kv_snap(1, 99, 100)],
        );
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("hello world"),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(idx < 2);
    }

    // ---- engine_stats + inflight speculation (new KV read path) ----

    use crate::worker::stats::{EngineSchedulerStats, EngineStats};
    use std::collections::HashMap;

    fn make_worker_with_capacity(url: &str, capacity: i64) -> Arc<dyn Worker> {
        let mut labels = HashMap::new();
        labels.insert("max_model_len".to_string(), capacity.to_string());
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .labels(labels)
                .build(),
        ) as Arc<dyn Worker>
    }

    fn engine_stats_with_tokens(prompt_tokens: usize, running: usize) -> EngineStats {
        let mut map = HashMap::new();
        for i in 0..running {
            map.insert(format!("req-{i}"), prompt_tokens / running.max(1));
        }
        let total = map.values().sum();
        EngineStats {
            timestamp: chrono::Utc::now(),
            scheduler_stats: EngineSchedulerStats {
                req_id_to_prompt_token_num: map,
                num_running_reqs: running,
                total_prompt_tokens: total,
                ..Default::default()
            },
        }
    }

    fn engine_stats_with_waiting(waiting: usize) -> EngineStats {
        EngineStats {
            timestamp: chrono::Utc::now(),
            scheduler_stats: EngineSchedulerStats {
                num_waiting_reqs: waiting,
                ..Default::default()
            },
        }
    }

    /// inflight_tokens tightens the gate: worker admits at snapshot-only usage
    /// but is rejected once inflight pushes effective used + new > capacity.
    #[test]
    fn spec_inflight_tightens_gate() {
        let policy = gated_policy();
        let w = make_worker_with_capacity("http://w1:8000", 100);
        let workers = vec![w.clone()];
        policy.init_workers(&workers);

        // Snapshot: 60 tokens used (3 running), capacity 100 → 40 free.
        w.update_engine_stats(engine_stats_with_tokens(60, 3), 0);

        // First route: 30 new tokens, no inflight → 60+0+30=90 ≤ 100 → admits.
        let tokens1: Vec<u32> = (1..=30).collect();
        let idx = policy.select_worker(
            &workers,
            &SelectWorkerInfo { tokens: Some(&tokens1), ..Default::default() },
        );
        assert!(idx.is_some(), "should admit when 60+30 <= 100");

        // Register 20 inflight tokens (a request admitted but not yet in snapshot).
        let _id = w.register_inflight_tokens(20);

        // Second route with DIFFERENT tokens (no cache hit → new_tokens = 32).
        // effective_used = 60+20 = 80. 80+32 = 112 > 100 → reject → None.
        let tokens2: Vec<u32> = (100..=131).collect();
        let idx2 = policy.select_worker(
            &workers,
            &SelectWorkerInfo { tokens: Some(&tokens2), ..Default::default() },
        );
        assert!(idx2.is_none(), "inflight pushed over capacity → None (re-enqueue)");
    }

    /// No engine snapshot → always admit (graceful degrade).
    #[test]
    fn spec_no_snapshot_admits() {
        let policy = gated_policy();
        let w = make_worker_with_capacity("http://w1:8000", 100);
        let workers = vec![w.clone()];
        policy.init_workers(&workers);
        // No update_engine_stats call → timestamp = 0.
        let tokens: Vec<u32> = (1..=90).collect();
        let idx = policy.select_worker(
            &workers,
            &SelectWorkerInfo { tokens: Some(&tokens), ..Default::default() },
        );
        assert!(idx.is_some(), "no snapshot → admit");
    }

    /// Waiting queue rejects even when KV has space.
    #[test]
    fn spec_waiting_queue_rejects() {
        let policy = gated_policy();
        let w = make_worker_with_capacity("http://w1:8000", 100);
        let workers = vec![w.clone()];
        policy.init_workers(&workers);
        // Only 10 tokens used, but waiting > 0 → reject.
        w.update_engine_stats(engine_stats_with_waiting(1), 0);
        let tokens: Vec<u32> = (1..=10).collect();
        let idx = policy.select_worker(
            &workers,
            &SelectWorkerInfo { tokens: Some(&tokens), ..Default::default() },
        );
        assert!(idx.is_none(), "waiting queue → reject");
    }

    /// No capacity label → admit even if gate is on (can't check, don't block).
    #[test]
    fn spec_no_capacity_label_admits() {
        let policy = gated_policy();
        let workers = make_workers(&["http://w1:8000"]);
        policy.init_workers(&workers);
        workers[0].update_engine_stats(engine_stats_with_tokens(90, 5), 0);
        let tokens: Vec<u32> = (1..=50).collect();
        let idx = policy.select_worker(
            &workers,
            &SelectWorkerInfo { tokens: Some(&tokens), ..Default::default() },
        );
        assert!(idx.is_some(), "no max_model_len label → admit");
    }

    /// KV imbalance trigger uses effective usage (snapshot + inflight).
    #[test]
    fn spec_imbalance_trigger_sees_effective_usage() {
        let policy = CacheAwareV1Policy::with_config(CacheAwareConfig {
            enable_kv_admission_control: true,
            overload_token_usage_threshold: 0.9,
            balance_token_usage_threshold: 0.2, // tight spread threshold
            ..Default::default()
        });
        let w1 = make_worker_with_capacity("http://w1:8000", 100);
        let w2 = make_worker_with_capacity("http://w2:8000", 100);
        let workers = vec![w1.clone(), w2.clone()];
        policy.init_workers(&workers);

        // Both workers low at 10 tokens. Spread = 0 → NOT imbalanced yet.
        w1.update_engine_stats(engine_stats_with_tokens(10, 1), 0);
        w2.update_engine_stats(engine_stats_with_tokens(10, 1), 0);
        let healthy = vec![0, 1];
        assert!(!policy.is_imbalanced(&workers, &healthy, 1, 1));

        // Add 40 inflight to w1 → effective w1 = 50, w2 = 10, spread = 0.4 > 0.2 → imbalanced.
        let _id = w1.register_inflight_tokens(40);
        assert!(
            policy.is_imbalanced(&workers, &healthy, 1, 1),
            "inflight on w1 should flip imbalanced"
        );
    }
}


