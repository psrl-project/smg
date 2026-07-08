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
use tracing::{debug, warn};

use super::{
    normalize_model_key, utils::PeriodicTask, CacheAwareConfig, LoadBalancingPolicy,
    SelectWorkerInfo, TreeHandle, TreeKind,
};
use crate::{
    mesh::adapters::tree_sync::{RepairEntry, TreeRepairPage},
    worker::{KvEventMonitor, Worker},
};

/// Latest per-worker backend load snapshot stream, keyed by worker URL.
pub(crate) type LoadReceiver = watch::Receiver<HashMap<String, WorkerLoadResponse>>;

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

    /// Min and max backend KV-cache utilization (0.0–1.0) across healthy workers
    /// that have a `WorkerMonitor` snapshot entry, as `(min, max)`. `None` when
    /// no receiver is wired or no healthy worker has a load entry (→ caller
    /// relies on the request-count spread).
    fn backend_token_usage_bounds(
        &self,
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
    ) -> Option<(f64, f64)> {
        let guard = self.load_rx.read();
        let rx = guard.as_ref()?;
        let loads = rx.borrow();
        let mut bounds: Option<(f64, f64)> = None;
        for &idx in healthy_indices {
            if let Some(load) = loads.get(workers[idx].url()) {
                let usage = load.effective_token_usage();
                bounds = Some(match bounds {
                    Some((min, max)) => (min.min(usage), max.max(usage)),
                    None => (usage, usage),
                });
            }
        }
        bounds
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
        min_load_idx: Option<usize>,
        model_id: &str,
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
        let min_load_idx = min_load_idx?;

        let worker_url = workers[min_load_idx].url();

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
        workers[min_load_idx].increment_processed();

        Some(min_load_idx)
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

        // Abandon cache affinity for shortest-queue when the pool is imbalanced —
        // by request count (using the loads already gathered above), or (for
        // long-context workloads) by backend KV usage.
        if self.is_imbalanced(workers, &healthy_indices, min_load, max_load) {
            return self.select_worker_min_load(workers, info, min_load_idx, model_id);
        }

        // Cache-aware routing when balanced — three types (mutually exclusive):
        //   1. Event-driven: PositionalIndexer overlap scoring (gRPC + KV events)
        //   2. Approximate token tree: TokenTree prefix matching (gRPC, no events)
        //   3. Approximate string tree: Tree prefix matching (HTTP)
        if let Some(tokens) = request_tokens {
            if self.has_event_indexer(model_id) {
                self.select_worker_event_driven(
                    workers,
                    tokens,
                    &healthy_indices,
                    min_load_idx,
                    model_id,
                )
            } else {
                self.select_worker_with_tokens(
                    workers,
                    tokens,
                    &healthy_indices,
                    min_load_idx,
                    model_id,
                )
            }
        } else {
            let text = request_text.unwrap_or("");
            self.select_worker_with_text(workers, text, &healthy_indices, min_load_idx, model_id)
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
    ) -> Option<usize> {
        let guard = self.kv_monitor.read();
        let monitor = guard.as_ref()?;
        let tiered_indexer = monitor.get_indexer(model_id)?;

        if let Some(idx) =
            self.score_overlap_tiered(workers, tokens, healthy_indices, &tiered_indexer)
        {
            return Some(idx);
        }

        // No cache overlap — min-load fallback
        let min_idx = min_load_idx?;
        debug!(
            worker = workers[min_idx].url(),
            model_id, "Event-driven routing: no overlap, min-load fallback"
        );
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
    ) -> Option<usize> {
        let mut weighted: Vec<f64> = vec![0.0; workers.len()];
        let mut tree_sizes: Vec<usize> = vec![0; workers.len()];

        for (tier, weight) in [
            (Tier::GPU, self.config.gpu_overlap_weight),
            (Tier::Lmcache, self.config.lmcache_overlap_weight),
        ] {
            if weight <= 0.0 {
                continue;
            }
            let block_size = indexer.block_size(tier).unwrap_or(self.config.block_size);
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
                    }
                    tree_sizes[idx] += overlap.tree_sizes.get(&wid).copied().unwrap_or(0);
                }
            }
        }

        let best_idx = healthy_indices
            .iter()
            .copied()
            .filter(|&idx| weighted[idx] > 0.0)
            .max_by(|&a, &b| {
                let load_a = workers[a].load();
                let load_b = workers[b].load();
                weighted[a]
                    .total_cmp(&weighted[b])
                    .then(load_b.cmp(&load_a))
                    .then(tree_sizes[b].cmp(&tree_sizes[a]))
            })?;

        debug!(
            worker = workers[best_idx].url(),
            score = weighted[best_idx],
            "Event-driven routing: weighted multi-tier overlap match"
        );
        workers[best_idx].increment_processed();
        Some(best_idx)
    }

    /// Select worker using token-based tree (gRPC path)
    fn select_worker_with_tokens(
        &self,
        workers: &[Arc<dyn Worker>],
        tokens: &[u32],
        healthy_indices: &[usize],
        min_load_idx: Option<usize>,
        model_id: &str,
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
            let mut selected_idx: Option<usize> = None;
            let result = tree.match_and_insert_with(tokens, |result| {
                let match_rate = if result.input_token_count == 0 {
                    0.0
                } else {
                    result.matched_token_count as f32 / result.input_token_count as f32
                };

                selected_idx = if match_rate > self.config.cache_threshold {
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
                workers[idx].increment_processed();
                return Some(idx);
            }

            // Selected worker no longer exists or unhealthy - fall back to first healthy
            // Stale entries will be cleaned up by LRU eviction
            healthy_indices.first().copied()
        } else {
            debug!(
                "Warning: No token tree found for model '{}', using random worker selection",
                model_id
            );
            let mut rng = rand::rng();
            let random_idx = rng.random_range(0..healthy_indices.len());
            Some(healthy_indices[random_idx])
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
            let result = tree.match_and_insert_with(text, |result| {
                let match_rate = if result.input_char_count == 0 {
                    0.0
                } else {
                    result.matched_char_count as f32 / result.input_char_count as f32
                };

                selected_idx = if match_rate > self.config.cache_threshold {
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

                workers[idx].increment_processed();
                return Some(idx);
            }

            // Selected worker no longer exists or unhealthy - fall back to first healthy
            // Stale entries will be cleaned up by LRU eviction
            healthy_indices.first().copied()
        } else {
            debug!(
                "Warning: No string tree found for model '{}', using random worker selection",
                model_id
            );
            let mut rng = rand::rng();
            let random_idx = rng.random_range(0..healthy_indices.len());
            Some(healthy_indices[random_idx])
        }
    }
}

impl Default for CacheAwareV1Policy {
    fn default() -> Self {
        Self::new()
    }
}
