//! Per-instance preemption subscriber task.
//!
//! Spawned once per vLLM instance at gateway startup. Maintains a persistent
//! gRPC stream and calls the existing `Abort` RPC for each preempted request
//! ID received. On stream end or error, reconnects with backoff.
//!
//! Re-queuing is handled by SMG's existing `dispatch_task` loopback on
//! `finish_reason="abort"` — no routing-loop changes required.
//!
//! [`PreemptionMonitor`] tracks per-worker subscriber handles and supports
//! dynamic worker registration and removal, following the `LoadMonitor` pattern.

use std::{collections::HashMap, sync::Arc, time::Duration};

use smg_grpc_client::VllmEngineClient;
use tokio::{sync::Mutex, task::JoinHandle};
use tracing::{info, warn};

use crate::worker::registry::WorkerId;

/// Tracks per-instance preemption subscriber tasks, handling dynamic
/// worker registration and removal.
///
/// Analogous to `LoadMonitor` but operates at the per-worker level rather
/// than per-group level, since each vLLM worker needs its own gRPC stream.
pub struct PreemptionMonitor {
    handles: Arc<Mutex<HashMap<WorkerId, JoinHandle<()>>>>,
}

impl PreemptionMonitor {
    pub fn new() -> Self {
        Self {
            handles: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Spawn a preemption subscriber for a newly registered vLLM worker.
    /// No-op if the worker already has a running subscriber.
    pub async fn on_worker_added(&self, worker_id: WorkerId, client: VllmEngineClient) {
        let mut handles = self.handles.lock().await;
        if handles.contains_key(&worker_id) {
            return; // already has a subscriber
        }
        info!("PreemptionMonitor: starting subscriber for {:?}", worker_id);
        #[expect(
            clippy::disallowed_methods,
            reason = "preemption subscriber runs for the lifetime of the worker"
        )]
        let handle = tokio::spawn(subscribe_preemptions(worker_id.clone(), client));
        handles.insert(worker_id, handle);
    }

    /// Abort and clean up the subscriber for a removed worker.
    pub async fn on_worker_removed(&self, worker_id: &WorkerId) {
        let mut handles = self.handles.lock().await;
        if let Some(handle) = handles.remove(worker_id) {
            handle.abort();
            info!("PreemptionMonitor: stopped subscriber for {:?}", worker_id);
        }
    }
}

impl Default for PreemptionMonitor {
    fn default() -> Self {
        Self::new()
    }
}

const RECONNECT_DELAY: Duration = Duration::from_secs(5);

/// Events older than this are skipped after reconnection.
///
/// During a reconnect window the Python side keeps queuing events into the
/// bounded asyncio queue.  On reconnection the Rust side drains that backlog
/// immediately; events whose `timestamp_ns` is older than `EVENT_TTL` are
/// almost certainly already resolved (re-queued or finished) and aborting them
/// would be harmful.  30 s comfortably covers the 5 s `RECONNECT_DELAY` plus
/// any transient Python-side queue depth.
const EVENT_TTL: Duration = Duration::from_secs(30);

/// Subscribe to preemption events from a single vLLM instance and abort
/// all reported requests in a single batch `Abort` RPC.
///
/// This function runs indefinitely; cancel the spawned task to stop it.
pub async fn subscribe_preemptions(instance_id: WorkerId, client: VllmEngineClient) {
    info!(
        "Starting preemption subscriber for instance {:?}",
        instance_id
    );
    loop {
        match client.subscribe_preemption_events().await {
            Ok(mut stream) => {
                info!("Preemption stream connected for instance {:?}", instance_id);
                loop {
                    match stream.message().await {
                        Ok(Some(event)) => {
                            // TTL filter: skip events that accumulated in the Python
                            // queue during a reconnect window — they are likely stale.
                            let age_ns =
                                current_time_ns().saturating_sub(event.timestamp_ns as u128);
                            if age_ns > EVENT_TTL.as_nanos() {
                                warn!(
                                    "Dropping stale preemption event ({} req(s), age {:.1}s) from {:?}",
                                    event.request_ids.len(),
                                    age_ns as f64 / 1e9,
                                    instance_id,
                                );
                                continue;
                            }

                            let req_ids = event.request_ids;
                            let n = req_ids.len();
                            info!(
                                target: "route_trace",
                                event = "preempt_recv",
                                instance = ?instance_id,
                                num_reqs = n,
                                age_ms = (age_ns / 1_000_000) as u64,
                                "preemption event received"
                            );

                            // Batch abort: one RPC for all IDs in the event,
                            // reducing latency from O(N × RTT) to O(1 RTT).
                            match client.abort_request(req_ids).await {
                                Ok(_) => info!(
                                    target: "route_trace",
                                    event = "preempt_abort_ok",
                                    instance = ?instance_id,
                                    num_reqs = n,
                                    "aborted preempted requests"
                                ),
                                Err(e) => warn!(
                                    "Failed to batch-abort preempted requests on {:?}: {}",
                                    instance_id, e
                                ),
                            }
                        }
                        Ok(None) => {
                            info!(
                                "Preemption stream ended for instance {:?}, reconnecting",
                                instance_id
                            );
                            break;
                        }
                        Err(e) => {
                            warn!(
                                "Preemption stream error for instance {:?}: {}, reconnecting",
                                instance_id, e
                            );
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                warn!(
                    "Failed to connect preemption stream for instance {:?}: {}",
                    instance_id, e
                );
            }
        }
        tokio::time::sleep(RECONNECT_DELAY).await;
    }
}

/// Return the current wall-clock time as nanoseconds since the Unix epoch.
#[inline]
fn current_time_ns() -> u128 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}
