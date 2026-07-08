//! Per-worker KV cache event subscription manager.
//!
//! `KvEventMonitor` spawns a background tokio task per (worker, dp_rank) that
//! subscribes to KV cache events and feeds them into a shared [`TieredIndexer`]
//! (one per model). This enables event-driven, multi-tier cache-aware routing as
//! an alternative to the approximate radix tree approach.
//!
//! **Multi-tier**: backends with a layered KV cache (vLLM GPU prefix cache +
//! LMCache CPU offload) tag each block with `cache_level` (0 = GPU, 1 = LMCache).
//! Events are routed into the matching [`Tier`] so the router can score GPU and
//! LMCache hits with different weights. An `AllBlocksCleared` event clears *all*
//! tiers for the worker (PSRL weight-sync invalidates both levels at once).
//!
//! **Data parallelism**: when one gRPC server fronts several DP ranks (PSRL),
//! each rank publishes on an independent stream with its own sequence counter.
//! Subscriptions are therefore keyed by `(url, dp_rank)` and each carries its
//! `dp_rank` in the subscribe request; interning/indexing is per-rank so caches
//! from different ranks never alias.
//!
//! Lifecycle:
//! - `on_worker_added` — spawns one streaming task per dp_rank, creates indexer
//! - `on_worker_removed` — signals graceful shutdown, task cleans up indexer
//! - `stop` — signals shutdown to all tasks, clears state

use std::{collections::HashMap, fmt, sync::Arc, time::Duration};

use dashmap::DashMap;
use kv_index::{
    compute_content_hash, ApplyError, SequenceHash, StoredBlock, Tier, TieredIndexer,
    WorkerBlockMap,
};
use smg_grpc_client::common_proto::{
    kv_cache_event, KvBlock, KvBlocksRemoved, KvBlocksStored, KvCacheEvent, KvEventBatch,
};
use tokio::{
    sync::{oneshot, Mutex},
    task::JoinHandle,
};
use tracing::{debug, info, warn};

use crate::worker::{ConnectionMode, Worker, UNKNOWN_MODEL_ID};

/// Default jump size for new `PositionalIndexer` instances.
const DEFAULT_JUMP_SIZE: usize = 64;

/// Initial reconnection delay after stream failure.
const INITIAL_RECONNECT_DELAY_MS: u64 = 100;

/// Maximum reconnection delay (caps exponential backoff).
const MAX_RECONNECT_DELAY_MS: u64 = 30_000;

/// Manages per-worker KV cache event subscriptions.
///
/// Each (gRPC worker, dp_rank) gets a dedicated tokio task that subscribes to
/// the backend's KV cache event stream and feeds events into a shared
/// [`TieredIndexer`] (one per `model_id`). Workers serving the same model share
/// the same indexer.
pub struct KvEventMonitor {
    /// Per-model tiered indexers: model_id → shared indexer.
    pub(crate) indexers: DashMap<String, Arc<TieredIndexer>>,
    /// Per-worker subscription handles, keyed by `(url, dp_rank)`.
    /// Mutex matches LoadMonitor pattern for atomic abort + remove.
    worker_handles: Mutex<HashMap<SubscriptionKey, WorkerSubscription>>,
    /// Jump size for new `TieredIndexer` instances.
    jump_size: usize,
}

/// Identifies a single subscription stream. A backend may front multiple DP
/// ranks behind one URL, each with an independent event stream, so the URL
/// alone is not unique.
type SubscriptionKey = (String, Option<u32>);

/// Tracks a single worker's subscription state.
struct WorkerSubscription {
    handle: JoinHandle<()>,
    model_id: String,
    /// Signals the subscription task to shut down gracefully.
    /// The task owns its `WorkerBlockMap` and cleans up the indexer on exit.
    shutdown_tx: oneshot::Sender<()>,
}

/// Result of processing a stream connection to completion.
enum StreamResult {
    /// Stream closed normally (server-side).
    Ended,
    /// Stream produced an error.
    Error(String),
}

impl KvEventMonitor {
    /// Create a new `KvEventMonitor`.
    ///
    /// `jump_size` controls the `PositionalIndexer` jump search stride.
    /// Pass `None` for the default (64).
    pub fn new(jump_size: Option<usize>) -> Self {
        let jump_size = jump_size.unwrap_or(DEFAULT_JUMP_SIZE).max(1);
        Self {
            indexers: DashMap::new(),
            worker_handles: Mutex::new(HashMap::new()),
            jump_size,
        }
    }

    /// Start a KV event subscription for a worker.
    ///
    /// Spawns a background tokio task that subscribes to KV cache events via
    /// server-streaming gRPC and applies them to the model's [`TieredIndexer`].
    /// Subscriptions are keyed by `(url, dp_rank)`; duplicate calls for the same
    /// key are no-ops. When a backend fronts several DP ranks behind one URL,
    /// this method is invoked once per rank (one Worker per rank), so each rank
    /// gets its own stream.
    pub async fn on_worker_added(&self, worker: &Arc<dyn Worker>) {
        let url = worker.url().to_string();
        let dp_rank = worker.dp_rank().map(|r| r as u32);
        // Normalize model_id to match routing's normalize_model_key — empty → "unknown".
        let model_id = Self::normalize_model_id(worker.model_id());

        if *worker.connection_mode() == ConnectionMode::Http {
            debug!(worker_url = %url, "HTTP worker, skipping KV event subscription");
            return;
        }

        let key: SubscriptionKey = (url.clone(), dp_rank);
        let mut handles = self.worker_handles.lock().await;
        if handles.contains_key(&key) {
            debug!(worker_url = %url, ?dp_rank, "KV event subscription already active, skipping");
            return;
        }

        let indexer = self
            .indexers
            .entry(model_id.clone())
            .or_insert_with(|| Arc::new(TieredIndexer::new(self.jump_size)))
            .clone();

        // Seed GPU-tier block_size provisionally from WorkerSpec. The event
        // stream overwrites this with the backend's actual page size once a
        // stored event arrives (the GPU prefix-cache block size is the one the
        // router uses to chunk request tokens for overlap scoring).
        if let Some(bs) = worker.metadata().spec.kv_block_size {
            if bs > 0 {
                indexer.set_block_size(Tier::GPU, bs);
            } else {
                warn!(worker_url = %url, "Worker reports kv_block_size=0, ignoring");
            }
        }

        let worker = Arc::clone(worker);
        let worker_url = url.clone();

        info!(
            worker_url = %url,
            ?dp_rank,
            model_id = %model_id,
            "Starting KV event subscription"
        );

        let (shutdown_tx, shutdown_rx) = oneshot::channel();

        #[expect(
            clippy::disallowed_methods,
            reason = "KV event monitor: runs for the lifetime of the worker, \
                      handle is stored and graceful shutdown is sent on removal"
        )]
        let handle = tokio::spawn(async move {
            Self::subscription_loop(worker, worker_url, dp_rank, indexer, shutdown_rx).await;
        });

        handles.insert(
            key,
            WorkerSubscription {
                handle,
                model_id,
                shutdown_tx,
            },
        );
    }

    /// Stop the KV event subscriptions for a worker (all dp_ranks behind `url`).
    ///
    /// Sends a graceful shutdown signal — each task cleans up its own
    /// `WorkerBlockMap` in the indexer before exiting.
    pub async fn on_worker_removed(&self, worker_url: &str) {
        // A single URL may carry multiple dp_rank subscriptions; remove them all.
        let subscriptions: Vec<WorkerSubscription> = {
            let mut handles = self.worker_handles.lock().await;
            let keys: Vec<SubscriptionKey> = handles
                .keys()
                .filter(|(u, _)| u == worker_url)
                .cloned()
                .collect();
            keys.into_iter()
                .filter_map(|k| handles.remove(&k))
                .collect()
        };

        if subscriptions.is_empty() {
            return;
        }

        info!(worker_url = %worker_url, count = subscriptions.len(), "Stopping KV event subscriptions");
        let mut model_ids: Vec<String> = Vec::new();
        for sub in subscriptions {
            // Signal graceful shutdown — task cleans up its worker_blocks in the indexer.
            let _ = sub.shutdown_tx.send(());
            let _ = sub.handle.await;
            if !model_ids.contains(&sub.model_id) {
                model_ids.push(sub.model_id);
            }
        }

        // Re-check under lock whether each model still has any subscription.
        // Must re-acquire lock after shutdown to avoid TOCTOU with concurrent
        // on_worker_added that may have added a new worker for the same model
        // between our first lock release and this point.
        let handles = self.worker_handles.lock().await;
        for model_id in model_ids {
            if !handles.values().any(|other| other.model_id == model_id) {
                self.indexers.remove(&model_id);
            }
        }
    }

    /// Stop all subscriptions and clean up.
    pub async fn stop(&self) {
        let subscriptions: HashMap<SubscriptionKey, WorkerSubscription> = {
            let mut handles = self.worker_handles.lock().await;
            std::mem::take(&mut *handles)
        };

        if !subscriptions.is_empty() {
            info!(
                count = subscriptions.len(),
                "Stopping all KV event subscriptions"
            );
            for ((url, dp_rank), sub) in subscriptions {
                debug!(worker_url = %url, ?dp_rank, "Stopping KV event subscription");
                let _ = sub.shutdown_tx.send(());
                let _ = sub.handle.await;
            }
        }

        self.indexers.clear();
    }

    /// Get the indexer for a model (used by `CacheAwarePolicy` for queries).
    pub fn get_indexer(&self, model_id: &str) -> Option<Arc<TieredIndexer>> {
        self.indexers.get(model_id).map(|r| Arc::clone(&r))
    }

    /// Get the GPU-tier block size for a model (learned from events or seeded
    /// from WorkerSpec). This is the page size the router uses to chunk request
    /// tokens for overlap scoring.
    pub fn block_size(&self, model_id: &str) -> Option<usize> {
        self.indexers
            .get(model_id)
            .and_then(|idx| idx.block_size(Tier::GPU))
    }

    /// Set the GPU-tier block size for a model (e.g. from WorkerSpec during
    /// registration). Does not overwrite a value already learned from events.
    pub fn set_block_size(&self, model_id: &str, block_size: usize) {
        self.indexers
            .entry(model_id.to_string())
            .or_insert_with(|| Arc::new(TieredIndexer::new(self.jump_size)))
            .set_block_size(Tier::GPU, block_size);
    }

    /// Check if any subscription is running.
    pub async fn is_running(&self) -> bool {
        !self.worker_handles.lock().await.is_empty()
    }

    /// Normalize model_id to match routing's `normalize_model_key`.
    /// Empty model IDs map to UNKNOWN_MODEL_ID for consistent keying.
    fn normalize_model_id(model_id: &str) -> String {
        if model_id.is_empty() {
            UNKNOWN_MODEL_ID.to_string()
        } else {
            model_id.to_string()
        }
    }

    // -----------------------------------------------------------------------
    // Subscription loop
    // -----------------------------------------------------------------------

    /// Learn `block_size` from the first `KvBlock` in a stored event.
    ///
    /// Called once per model when the first stored event arrives, providing
    /// ground truth from the backend. `CacheAwarePolicy` uses this to chunk
    /// Learn the per-tier `block_size` from the first stored event seen for
    /// each tier. GPU and LMCache use different page sizes, so each tier learns
    /// independently. Overwrites any provisional value seeded from `WorkerSpec`.
    fn learn_block_sizes(indexer: &TieredIndexer, batch: &KvEventBatch) {
        for event in &batch.events {
            if let Some(kv_cache_event::Data::Stored(stored)) = &event.data {
                if let Some(block) = stored.blocks.first() {
                    if block.block_size > 0 {
                        let tier = Tier::from_cache_level(block.cache_level);
                        if indexer.block_size(tier).is_none() {
                            indexer.set_block_size(tier, block.block_size as usize);
                            info!(
                                ?tier,
                                block_size = block.block_size,
                                "Learned block_size from KV event"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Main subscription loop for a single (worker, dp_rank).
    ///
    /// Owns the per-tier [`WorkerTierState`] and cleans it up on exit.
    /// Exits when `shutdown_rx` fires or the backend returns `Unimplemented`.
    async fn subscription_loop(
        worker: Arc<dyn Worker>,
        worker_url: String,
        dp_rank: Option<u32>,
        indexer: Arc<TieredIndexer>,
        mut shutdown_rx: oneshot::Receiver<()>,
    ) {
        let mut state = WorkerTierState::new(&indexer, &worker_url);
        let mut last_seq: u64 = 0;
        let mut reconnect_delay_ms = INITIAL_RECONNECT_DELAY_MS;

        /// Sleep with shutdown check. Returns `true` if shutdown was signaled.
        macro_rules! sleep_or_shutdown {
            ($delay:expr, $rx:expr) => {
                tokio::select! {
                    _ = tokio::time::sleep($delay) => false,
                    _ = &mut *$rx => true,
                }
            };
        }

        loop {
            let grpc_client = match worker.get_grpc_client().await {
                Ok(Some(client)) => client,
                Ok(None) => {
                    // HTTP workers are filtered in on_worker_added, so this should
                    // be unreachable. Retry defensively rather than exiting and
                    // leaving a stale entry in worker_handles.
                    warn!(
                        worker_url = %worker_url,
                        delay_ms = reconnect_delay_ms,
                        "Worker has no gRPC client yet, retrying"
                    );
                    if sleep_or_shutdown!(
                        Duration::from_millis(reconnect_delay_ms),
                        &mut shutdown_rx
                    ) {
                        state.cleanup(&indexer);
                        return;
                    }
                    reconnect_delay_ms = (reconnect_delay_ms * 2).min(MAX_RECONNECT_DELAY_MS);
                    continue;
                }
                Err(e) => {
                    warn!(
                        worker_url = %worker_url,
                        error = %e,
                        delay_ms = reconnect_delay_ms,
                        "Failed to get gRPC client, retrying"
                    );
                    if sleep_or_shutdown!(
                        Duration::from_millis(reconnect_delay_ms),
                        &mut shutdown_rx
                    ) {
                        state.cleanup(&indexer);
                        return;
                    }
                    reconnect_delay_ms = (reconnect_delay_ms * 2).min(MAX_RECONNECT_DELAY_MS);
                    continue;
                }
            };

            let stream = match grpc_client.subscribe_kv_events(last_seq, dp_rank).await {
                Ok(stream) => {
                    info!(
                        worker_url = %worker_url,
                        ?dp_rank,
                        start_seq = last_seq,
                        "KV event stream connected"
                    );
                    reconnect_delay_ms = INITIAL_RECONNECT_DELAY_MS;
                    stream
                }
                Err(e) => {
                    // If the backend doesn't implement SubscribeKvEvents (e.g. vLLM),
                    // stop retrying — this RPC will never succeed.
                    if e.code() == tonic::Code::Unimplemented {
                        warn!(
                            worker_url = %worker_url,
                            "Backend does not implement SubscribeKvEvents, \
                             disabling KV event subscription for this worker"
                        );
                        state.cleanup(&indexer);
                        return;
                    }
                    warn!(
                        worker_url = %worker_url,
                        error = %e,
                        delay_ms = reconnect_delay_ms,
                        "Failed to subscribe to KV events, retrying"
                    );
                    if sleep_or_shutdown!(
                        Duration::from_millis(reconnect_delay_ms),
                        &mut shutdown_rx
                    ) {
                        state.cleanup(&indexer);
                        return;
                    }
                    reconnect_delay_ms = (reconnect_delay_ms * 2).min(MAX_RECONNECT_DELAY_MS);
                    continue;
                }
            };

            let stream_result = tokio::select! {
                result = Self::process_stream(
                    stream, &worker_url, &indexer, &mut state, &mut last_seq,
                ) => result,
                _ = &mut shutdown_rx => {
                    state.cleanup(&indexer);
                    return;
                }
            };

            match stream_result {
                StreamResult::Ended => {
                    info!(
                        worker_url = %worker_url,
                        last_seq = last_seq,
                        delay_ms = reconnect_delay_ms,
                        "KV event stream ended, reconnecting"
                    );
                    // Backoff to avoid tight reconnect loop if server keeps
                    // closing the stream cleanly (e.g., rolling connections).
                    if sleep_or_shutdown!(
                        Duration::from_millis(reconnect_delay_ms),
                        &mut shutdown_rx
                    ) {
                        state.cleanup(&indexer);
                        return;
                    }
                    reconnect_delay_ms = (reconnect_delay_ms * 2).min(MAX_RECONNECT_DELAY_MS);
                }
                StreamResult::Error(e) => {
                    warn!(
                        worker_url = %worker_url,
                        error = %e,
                        last_seq = last_seq,
                        delay_ms = reconnect_delay_ms,
                        "KV event stream error, reconnecting"
                    );
                    if sleep_or_shutdown!(
                        Duration::from_millis(reconnect_delay_ms),
                        &mut shutdown_rx
                    ) {
                        state.cleanup(&indexer);
                        return;
                    }
                    reconnect_delay_ms = (reconnect_delay_ms * 2).min(MAX_RECONNECT_DELAY_MS);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Stream processing + proto conversion
    // -----------------------------------------------------------------------

    /// Process batches from a single stream connection.
    async fn process_stream(
        mut stream: tonic::Streaming<KvEventBatch>,
        worker_url: &str,
        indexer: &TieredIndexer,
        state: &mut WorkerTierState,
        last_seq: &mut u64,
    ) -> StreamResult {
        use tokio_stream::StreamExt;

        while let Some(result) = stream.next().await {
            let batch = match result {
                Ok(batch) => batch,
                Err(e) => return StreamResult::Error(e.to_string()),
            };

            // Skip stale/duplicate batches (can occur after reconnect).
            if *last_seq > 0 && batch.sequence_number <= *last_seq {
                debug!(
                    worker_url = %worker_url,
                    last_seq = *last_seq,
                    received = batch.sequence_number,
                    "Skipping stale KV event batch"
                );
                continue;
            }

            // Monotonic acceptance: a seq strictly greater than last_seq is
            // accepted as-is. We do NOT require seq == last_seq+1.
            //
            // Rationale (debugged 2026-07): the publisher's seq can have benign
            // holes (occasional dropped ZMQ frame; the inline servicer path does
            // not honor start_sequence_number, so reconnect-for-replay can never
            // refill a hole). Treating a hole as a fault triggered a reconnect/
            // resync storm (thousands/sec) that churned the stream without ever
            // recovering the missing seq. Empirically, dropping a few Stored
            // events does not hurt routing (overlap stays ~100%), whereas the
            // storm did. So we tolerate gaps: just advance last_seq and keep
            // applying live events.
            if *last_seq > 0 && batch.sequence_number > *last_seq + 1 {
                debug!(
                    worker_url = %worker_url,
                    last_seq = *last_seq,
                    received = batch.sequence_number,
                    "KV event seq gap tolerated (advancing without replay)"
                );
            }

            Self::learn_block_sizes(indexer, &batch);

            for event in &batch.events {
                Self::apply_event(event, indexer, state);
            }

            *last_seq = batch.sequence_number;
        }

        StreamResult::Ended
    }

    /// Apply a single KV cache event to the matching tier of the indexer.
    fn apply_event(event: &KvCacheEvent, indexer: &TieredIndexer, state: &mut WorkerTierState) {
        let Some(ref data) = event.data else {
            return;
        };

        match data {
            kv_cache_event::Data::Stored(stored) => {
                Self::apply_stored(stored, indexer, state);
            }
            kv_cache_event::Data::Removed(removed) => {
                Self::apply_removed(removed, indexer, state);
            }
            kv_cache_event::Data::Cleared(_) => {
                // Dual-clear: a clear event (PSRL weight-sync) invalidates the
                // whole multi-tier cache for this worker. LMCache emits no
                // explicit clear event of its own, so clearing every tier on the
                // GPU-side `AllBlocksCleared` is what keeps the off-GPU tier from
                // serving stale hits after a weight update.
                for tier in Tier::ALL {
                    indexer
                        .tier(tier)
                        .apply_cleared(state.worker_id(tier), state.blocks_mut(tier));
                }
            }
        }
    }

    /// Convert proto `KvBlocksStored` and apply it to the block's tier.
    ///
    /// vLLM emits each store event from a single source (GPU `kv_cache_manager`
    /// or the LMCache connector), so all blocks in one event share a tier; we
    /// take the tier from the first block.
    fn apply_stored(stored: &KvBlocksStored, indexer: &TieredIndexer, state: &mut WorkerTierState) {
        let Some(first) = stored.blocks.first() else {
            return;
        };
        let tier = Tier::from_cache_level(first.cache_level);
        let blocks: Vec<StoredBlock> = stored.blocks.iter().map(convert_kv_block).collect();
        let parent_seq_hash = stored.parent_block_hash.map(SequenceHash::from);

        let pi = indexer.tier(tier);
        let worker_id = state.worker_id(tier);
        let worker_blocks = state.blocks_mut(tier);

        match pi.apply_stored(worker_id, &blocks, parent_seq_hash, worker_blocks) {
            Ok(()) => {}
            Err(ApplyError::WorkerNotTracked | ApplyError::ParentBlockNotFound) => {
                // Cold start or parent evicted — retry without parent to start a new chain.
                if let Err(e) = pi.apply_stored(worker_id, &blocks, None, worker_blocks) {
                    warn!(
                        worker_id = worker_id,
                        ?tier,
                        error = %e,
                        "Failed to apply stored event after fallback"
                    );
                }
            }
        }
    }

    /// Convert proto `KvBlocksRemoved` and apply it to the event's tier.
    fn apply_removed(
        removed: &KvBlocksRemoved,
        indexer: &TieredIndexer,
        state: &mut WorkerTierState,
    ) {
        let tier = Tier::from_cache_level(removed.cache_level);
        let seq_hashes: Vec<SequenceHash> = removed
            .block_hashes
            .iter()
            .map(|&h| SequenceHash::from(h))
            .collect();

        indexer.tier(tier).apply_removed(
            state.worker_id(tier),
            &seq_hashes,
            state.blocks_mut(tier),
        );
    }
}

/// Per-tier subscription state for one (worker, dp_rank) stream.
///
/// Each tier interns the worker independently (its own internal `u32` id) and
/// keeps its own reverse `WorkerBlockMap`, because the two tiers index content
/// hashes in separate spaces. Owned by the subscription task; cleaned up on exit.
struct WorkerTierState {
    worker_ids: [u32; Tier::COUNT],
    blocks: [WorkerBlockMap; Tier::COUNT],
}

impl WorkerTierState {
    fn new(indexer: &TieredIndexer, worker_url: &str) -> Self {
        Self {
            worker_ids: Tier::ALL.map(|tier| {
                indexer
                    .intern_worker(tier, worker_url)
                    .expect("worker id space exhausted")
            }),
            blocks: std::array::from_fn(|_| WorkerBlockMap::default()),
        }
    }

    #[inline]
    fn worker_id(&self, tier: Tier) -> u32 {
        self.worker_ids[tier.index()]
    }

    #[inline]
    fn blocks_mut(&mut self, tier: Tier) -> &mut WorkerBlockMap {
        &mut self.blocks[tier.index()]
    }

    /// Drop this worker's blocks from every tier of the indexer.
    fn cleanup(&mut self, indexer: &TieredIndexer) {
        for tier in Tier::ALL {
            let worker_blocks = std::mem::take(&mut self.blocks[tier.index()]);
            indexer
                .tier(tier)
                .remove_worker(self.worker_ids[tier.index()], worker_blocks);
        }
    }
}

/// Convert a proto `KvBlock` to a kv-index `StoredBlock` (tier handled by caller).
fn convert_kv_block(block: &KvBlock) -> StoredBlock {
    StoredBlock {
        seq_hash: SequenceHash::from(block.block_hash),
        content_hash: compute_content_hash(&block.token_ids),
    }
}

impl Drop for KvEventMonitor {
    fn drop(&mut self) {
        if let Ok(mut handles) = self.worker_handles.try_lock() {
            for (_, sub) in handles.drain() {
                let _ = sub.shutdown_tx.send(());
                sub.handle.abort(); // Can't await in Drop, abort as fallback
            }
        }
    }
}

impl fmt::Debug for KvEventMonitor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KvEventMonitor")
            .field("models", &self.indexers.len())
            .field("jump_size", &self.jump_size)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a stored-blocks event for a given tier (via `cache_level`).
    fn stored_event(
        blocks: &[(i64, &[u32])],
        block_size: i32,
        parent: Option<i64>,
        cache_level: Option<i32>,
    ) -> KvBlocksStored {
        KvBlocksStored {
            blocks: blocks
                .iter()
                .map(|(hash, tokens)| KvBlock {
                    block_hash: *hash,
                    token_ids: tokens.to_vec(),
                    block_size,
                    lora_id: None,
                    cache_level,
                })
                .collect(),
            parent_block_hash: parent,
        }
    }

    // -----------------------------------------------------------------------
    // Proto → kv-index conversion
    // -----------------------------------------------------------------------

    #[test]
    fn test_convert_kv_block() {
        let block = KvBlock {
            block_hash: 42,
            token_ids: vec![1, 2, 3, 4],
            block_size: 4,
            lora_id: None,
            cache_level: None,
        };
        let stored = convert_kv_block(&block);
        assert_eq!(stored.seq_hash, SequenceHash::from(42i64));
        assert_eq!(stored.content_hash, compute_content_hash(&[1, 2, 3, 4]));
    }

    #[test]
    fn test_convert_kv_block_negative_hash() {
        let block = KvBlock {
            block_hash: -1,
            token_ids: vec![10, 20],
            block_size: 2,
            lora_id: None,
            cache_level: None,
        };
        let stored = convert_kv_block(&block);
        assert_eq!(stored.seq_hash, SequenceHash(u64::MAX));
    }

    // -----------------------------------------------------------------------
    // apply_event integration with TieredIndexer
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_stored_no_parent() {
        let indexer = TieredIndexer::new(64);
        let mut state = WorkerTierState::new(&indexer, "http://w1:8000");
        let stored = stored_event(
            &[(1, &[10, 20, 30, 40]), (2, &[50, 60, 70, 80])],
            4,
            None,
            None,
        );
        KvEventMonitor::apply_stored(&stored, &indexer, &mut state);
        assert_eq!(indexer.tier(Tier::GPU).current_size(), 2);
    }

    #[test]
    fn test_apply_stored_with_parent() {
        let indexer = TieredIndexer::new(64);
        let mut state = WorkerTierState::new(&indexer, "http://w1:8000");
        KvEventMonitor::apply_stored(
            &stored_event(&[(1, &[10, 20, 30, 40])], 4, None, None),
            &indexer,
            &mut state,
        );
        KvEventMonitor::apply_stored(
            &stored_event(&[(2, &[50, 60, 70, 80])], 4, Some(1), None),
            &indexer,
            &mut state,
        );
        assert_eq!(indexer.tier(Tier::GPU).current_size(), 2);
    }

    #[test]
    fn test_apply_stored_fallback_on_parent_not_found() {
        let indexer = TieredIndexer::new(64);
        let mut state = WorkerTierState::new(&indexer, "http://new-worker:8000");
        // parent for a cold worker — should fall back to no-parent insert.
        KvEventMonitor::apply_stored(
            &stored_event(&[(1, &[10, 20, 30, 40])], 4, Some(999), None),
            &indexer,
            &mut state,
        );
        assert_eq!(indexer.tier(Tier::GPU).current_size(), 1);
    }

    #[test]
    fn test_apply_removed() {
        let indexer = TieredIndexer::new(64);
        let mut state = WorkerTierState::new(&indexer, "http://w1:8000");
        KvEventMonitor::apply_stored(
            &stored_event(
                &[(1, &[10, 20, 30, 40]), (2, &[50, 60, 70, 80])],
                4,
                None,
                None,
            ),
            &indexer,
            &mut state,
        );
        KvEventMonitor::apply_removed(
            &KvBlocksRemoved {
                block_hashes: vec![2],
                cache_level: None,
            },
            &indexer,
            &mut state,
        );
        assert_eq!(indexer.tier(Tier::GPU).current_size(), 1);
    }

    /// Stored events route to the tier indicated by `cache_level`, and the two
    /// tiers track size independently.
    #[test]
    fn test_apply_stored_routes_by_tier() {
        let indexer = TieredIndexer::new(64);
        let mut state = WorkerTierState::new(&indexer, "http://w1:8000");

        KvEventMonitor::apply_stored(
            &stored_event(&[(1, &[1, 2, 3, 4])], 4, None, Some(0)), // GPU
            &indexer,
            &mut state,
        );
        KvEventMonitor::apply_stored(
            &stored_event(
                &[(2, &[5, 6, 7, 8]), (3, &[9, 10, 11, 12])],
                4,
                None,
                Some(1),
            ), // LMCache
            &indexer,
            &mut state,
        );

        assert_eq!(indexer.tier(Tier::GPU).current_size(), 1);
        assert_eq!(indexer.tier(Tier::Lmcache).current_size(), 2);
    }

    /// A clear event must wipe BOTH tiers: LMCache emits no clear
    /// of its own, so the GPU-side `AllBlocksCleared` is the invalidation signal
    /// for the whole multi-tier cache.
    #[test]
    fn test_apply_cleared_clears_all_tiers() {
        let indexer = TieredIndexer::new(64);
        let mut state = WorkerTierState::new(&indexer, "http://w1:8000");
        KvEventMonitor::apply_stored(
            &stored_event(&[(1, &[1, 2, 3, 4])], 4, None, Some(0)),
            &indexer,
            &mut state,
        );
        KvEventMonitor::apply_stored(
            &stored_event(&[(2, &[5, 6, 7, 8])], 4, None, Some(1)),
            &indexer,
            &mut state,
        );
        assert_eq!(indexer.current_size(), 2);

        KvEventMonitor::apply_event(
            &KvCacheEvent {
                event_id: 9,
                data: Some(kv_cache_event::Data::Cleared(
                    smg_grpc_client::common_proto::KvCacheCleared {},
                )),
            },
            &indexer,
            &mut state,
        );
        assert_eq!(indexer.tier(Tier::GPU).current_size(), 0);
        assert_eq!(indexer.tier(Tier::Lmcache).current_size(), 0);
    }

    #[test]
    fn test_apply_event_no_data() {
        let indexer = TieredIndexer::new(64);
        let mut state = WorkerTierState::new(&indexer, "http://w1:8000");
        KvEventMonitor::apply_event(
            &KvCacheEvent {
                event_id: 1,
                data: None,
            },
            &indexer,
            &mut state,
        );
        assert_eq!(indexer.current_size(), 0);
    }

    #[test]
    fn test_learn_block_sizes_per_tier() {
        let indexer = TieredIndexer::new(64);
        let batch = KvEventBatch {
            sequence_number: 1,
            timestamp: 0.0,
            dp_rank: None,
            events: vec![
                KvCacheEvent {
                    event_id: 1,
                    data: Some(kv_cache_event::Data::Stored(stored_event(
                        &[(1, &[1, 2, 3, 4])],
                        16,
                        None,
                        Some(0),
                    ))),
                },
                KvCacheEvent {
                    event_id: 2,
                    data: Some(kv_cache_event::Data::Stored(stored_event(
                        &[(2, &[5, 6])],
                        256,
                        None,
                        Some(1),
                    ))),
                },
            ],
        };
        KvEventMonitor::learn_block_sizes(&indexer, &batch);
        assert_eq!(indexer.block_size(Tier::GPU), Some(16));
        assert_eq!(indexer.block_size(Tier::Lmcache), Some(256));
    }

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_monitor_new() {
        let monitor = KvEventMonitor::new(None);
        assert!(!monitor.is_running().await);
    }

    #[tokio::test]
    async fn test_monitor_new_clamps_zero_jump_size() {
        let monitor = KvEventMonitor::new(Some(0));
        assert_eq!(monitor.jump_size, 1);
    }

    #[tokio::test]
    async fn test_get_indexer_nonexistent() {
        let monitor = KvEventMonitor::new(None);
        assert!(monitor.get_indexer("nonexistent").is_none());
    }

    #[tokio::test]
    async fn test_stop_empty_monitor() {
        let monitor = KvEventMonitor::new(None);
        monitor.stop().await;
    }

    #[tokio::test]
    async fn test_on_worker_removed_nonexistent() {
        let monitor = KvEventMonitor::new(None);
        monitor.on_worker_removed("http://nonexistent:8000").await;
    }

    // -----------------------------------------------------------------------
    // block_size learning (GPU tier is the router-facing default)
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_block_size() {
        let monitor = KvEventMonitor::new(None);
        assert!(monitor.block_size("llama").is_none());

        monitor.set_block_size("llama", 32);
        assert_eq!(monitor.block_size("llama"), Some(32));

        // set_block_size doesn't overwrite an existing value (learn-once).
        monitor.set_block_size("llama", 64);
        assert_eq!(monitor.block_size("llama"), Some(32));
    }

    #[tokio::test]
    async fn test_stop_clears_block_sizes() {
        let monitor = KvEventMonitor::new(None);
        monitor.set_block_size("llama", 16);
        assert_eq!(monitor.block_size("llama"), Some(16));

        monitor.stop().await;
        assert!(monitor.block_size("llama").is_none());
    }
}
