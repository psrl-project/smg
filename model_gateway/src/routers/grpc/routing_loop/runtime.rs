//! Runtime for request routing-loop dispatch.

use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, OnceLock,
    },
    time::Duration,
};

use axum::{
    http::HeaderValue,
    response::{IntoResponse, Response},
};
use dashmap::DashMap;
use metrics::{counter, gauge, histogram};
use openai_protocol::chat::ChatCompletionResponse;
use serde::Serialize;
use tokio::{
    sync::{mpsc, oneshot, Mutex, Notify},
    task::{yield_now, JoinSet},
    time::sleep,
};
use tracing::{debug, error, info};

use super::{
    metadata::{parse_routing_request_meta_from_context, RoutingMeta},
    partial_rollout::{
        drain_stream_for_partial_rollout, merge_into_partial_state, reset_ctx_for_loopback,
    },
    queue::{MultiPriorityRequestQueue, RequestPriority},
};
use crate::{
    config::RoutingLoopConfig,
    routers::{
        error as router_error,
        grpc::{
            context::{
                ExecutionResult, FinalResponse, LoadGuards, RequestContext, WorkerSelection,
            },
            harmony::ResponsesIterationResult,
            pipeline::RequestPipeline,
        },
    },
    worker::{Worker, WorkerRegistry},
};

pub(crate) type InstanceVersionMap = Arc<DashMap<(String, usize), i64>>;

pub(crate) struct RoutingQueueEntry {
    pub(crate) ctx: RequestContext,
    pub(crate) pipeline: RequestPipeline,
    pub(crate) completion: RoutingLoopCompletion,
    pub(crate) routing_meta: Option<RoutingMeta>,
}

pub(crate) enum RoutingLoopCompletion {
    Http(oneshot::Sender<Response>),
    ChatForResponses(oneshot::Sender<Result<ChatCompletionResponse, Response>>),
    HarmonyResponses(oneshot::Sender<Result<ResponsesIterationResult, Response>>),
    HarmonyResponsesStreaming(
        oneshot::Sender<Result<(ExecutionResult, Option<LoadGuards>), Response>>,
    ),
}

impl RequestPriority for RoutingQueueEntry {
    fn version_tag(&self) -> i64 {
        self.routing_meta
            .as_ref()
            .map_or(-1, |meta| meta.version_tag)
    }

    fn is_validation(&self) -> bool {
        self.routing_meta
            .as_ref()
            .is_some_and(|meta| meta.is_validate)
    }

    fn input_len(&self) -> usize {
        self.ctx
            .state
            .preparation
            .as_ref()
            .map_or(0, |preparation| preparation.token_ids().len())
    }

    fn request_id(&self) -> Option<i64> {
        self.routing_meta.as_ref().map(|meta| meta.request_id)
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RoutingLoopStatus {
    pub(crate) enabled: bool,
    pub(crate) paused: bool,
    pub(crate) routing: bool,
    pub(crate) selecting: bool,
    pub(crate) routing_epoch: u64,
    pub(crate) active_dispatch_handoffs: usize,
    pub(crate) queue_len: usize,
    pub(crate) pending_request_num: usize,
    pub(crate) running_tasks: usize,
    pub(crate) running_request_num: usize,
    pub(crate) max_running_dispatch_tasks: usize,
    pub(crate) queue_keys: Vec<i32>,
    pub(crate) partition_queue_lens: HashMap<i32, usize>,
}

/// Lightweight metadata snapshot for a single queued request.
///
/// Returned by `RoutingLoopRuntime::filter_queue_by_version_tag` and serialised
/// as JSON by the `GET /routing_loop/filter` endpoint.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct QueuedRequestMeta {
    pub(crate) request_id: i64,
    pub(crate) version_tag: i64,
    pub(crate) prompt_id: i64,
}

pub struct RoutingLoopRuntime {
    queue: Mutex<MultiPriorityRequestQueue<RoutingQueueEntry>>,
    tx: mpsc::UnboundedSender<RoutingQueueEntry>,
    paused: AtomicBool,
    routing_epoch: AtomicU64,
    running_tasks: AtomicUsize,
    active_decisions: AtomicUsize,
    active_dispatch_handoffs: AtomicUsize,
    admission_notify: Notify,
    running_requests: AtomicUsize,
    check_interval_ms: u64,
    receive_batch_size: usize,
    dispatch_batch_size: usize,
    max_running_dispatch_tasks: usize,
    /// `(base_worker_id, dp_rank)` → last synced version tag.
    ///
    /// Uses a sharded concurrent map because Stage 1 is on the hot dispatch
    /// path and should not serialize all readers on a global lock.
    pub(crate) instance_to_version_after_sync: InstanceVersionMap,

    /// `prompt_id` → running `request_id`s sharing that prompt.
    pub(crate) prompt_to_running_request_ids: DashMap<i64, Vec<i64>>,

    /// `prompt_id` → the instance pinned to that prompt group.
    ///
    /// Written once (first placement) via `entry().or_insert()` so that
    /// concurrent Stage 3 reads see either *no entry* (free selection) or a
    /// committed instance — eliminating the TOCTOU window between Stage 3
    /// read and post-selection write.
    pub(crate) prompt_to_pinned_instance: DashMap<i64, (String, usize)>,

    /// PS Manager gRPC client.
    ///
    /// `None` when the strategy is `"naive"` or no address was configured.
    /// Initialised once during router start-up via `connect_ps_manager()`.
    /// `OnceLock` avoids a Mutex on the read hot path.
    pub(crate) ps_manager_client: OnceLock<Arc<psrl_state::PSManagerStateClient>>,

    /// Worker registry used to resolve stable worker IDs during partial-rollout
    /// loopback — specifically to call `reserve_id_for_url` for header injection.
    pub(crate) worker_registry: Arc<WorkerRegistry>,
}

impl RoutingLoopRuntime {
    pub(crate) fn new(
        config: &RoutingLoopConfig,
        instance_to_version_after_sync: InstanceVersionMap,
        worker_registry: Arc<WorkerRegistry>,
    ) -> (Arc<Self>, mpsc::UnboundedReceiver<RoutingQueueEntry>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let runtime = Arc::new(Self {
            queue: Mutex::new(MultiPriorityRequestQueue::new(
                config.request_sort_key,
                config.enable_multi_priority_queue,
            )),
            tx,
            paused: AtomicBool::new(false),
            routing_epoch: AtomicU64::new(0),
            running_tasks: AtomicUsize::new(0),
            active_decisions: AtomicUsize::new(0),
            active_dispatch_handoffs: AtomicUsize::new(0),
            admission_notify: Notify::new(),
            running_requests: AtomicUsize::new(0),
            check_interval_ms: config.check_interval_ms,
            receive_batch_size: config.receive_batch_size.max(1),
            dispatch_batch_size: config.dispatch_batch_size.max(1),
            max_running_dispatch_tasks: config.max_running_dispatch_tasks.max(1),
            instance_to_version_after_sync,
            prompt_to_running_request_ids: DashMap::new(),
            prompt_to_pinned_instance: DashMap::new(),
            ps_manager_client: OnceLock::new(),
            worker_registry,
        });
        (runtime, rx)
    }

    pub(crate) async fn connect_ps_manager(
        &self,
        addr: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.ps_manager_client.get().is_some() {
            return Ok(());
        }
        let client = psrl_state::PSManagerStateClient::connect(addr).await?;
        // If another task raced us here the value is already set; that's fine.
        let _ = self.ps_manager_client.set(Arc::new(client));
        Ok(())
    }

    pub(crate) fn record_selected_instance(
        &self,
        request_id: i64,
        prompt_id: Option<i64>,
        instance: (String, usize),
    ) {
        if let Some(prompt_id) = prompt_id {
            // Atomically claim the group pin for this prompt if not yet set.
            // `entry().or_insert()` holds the DashMap shard write-lock during
            // the check-and-insert, so at most one caller ever writes the value.
            self.prompt_to_pinned_instance
                .entry(prompt_id)
                .or_insert(instance);

            let mut entry = self
                .prompt_to_running_request_ids
                .entry(prompt_id)
                .or_default();
            if !entry.contains(&request_id) {
                entry.push(request_id);
                self.running_requests.fetch_add(1, Ordering::AcqRel);
            }
        }
    }

    pub(crate) fn cleanup_tracking(&self, request_id: Option<i64>, prompt_id: Option<i64>) {
        let Some(request_id) = request_id else {
            return;
        };

        if let Some(prompt_id) = prompt_id {
            let remove_prompt = self
                .prompt_to_running_request_ids
                .get_mut(&prompt_id)
                .map(|mut ids| {
                    let before = ids.len();
                    ids.retain(|rid| *rid != request_id);
                    if ids.len() < before {
                        self.running_requests.fetch_sub(1, Ordering::AcqRel);
                    }
                    ids.is_empty()
                })
                .unwrap_or(false);
            if remove_prompt {
                self.prompt_to_running_request_ids.remove(&prompt_id);
                // All requests in the group have completed; release the pin so
                // a future group for the same prompt_id can be freely placed.
                self.prompt_to_pinned_instance.remove(&prompt_id);
            }
        }
    }

    pub(crate) fn instance_id_for_worker(&self, worker: &Arc<dyn Worker>) -> (String, usize) {
        let base_worker_id = self
            .worker_registry
            .reserve_id_for_url(worker.base_url())
            .as_str()
            .to_string();
        (base_worker_id, worker.dp_rank().unwrap_or(0))
    }

    pub(crate) fn enqueue(&self, entry: RoutingQueueEntry) -> Result<(), Box<RoutingQueueEntry>> {
        self.tx.send(entry).map_err(|err| Box::new(err.0))
    }

    pub(crate) fn pause(&self) {
        if !self.paused.swap(true, Ordering::AcqRel) {
            self.routing_epoch.fetch_add(1, Ordering::AcqRel);
        }
    }

    pub(crate) fn resume(&self) {
        self.paused.store(false, Ordering::Release);
        self.admission_notify.notify_waiters();
    }

    pub(crate) fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Acquire)
    }

    /// Returns `true` while at least one dispatch task is still inside the
    /// worker-selection stage.
    pub(crate) fn is_selecting(&self) -> bool {
        self.active_decisions.load(Ordering::Acquire) > 0
    }

    pub(crate) fn is_routing(&self) -> bool {
        self.is_selecting() || self.active_dispatch_handoffs.load(Ordering::Acquire) > 0
    }

    pub(crate) fn routing_epoch(&self) -> u64 {
        self.routing_epoch.load(Ordering::Acquire)
    }

    /// Atomically admit a new side-effect-free routing decision.
    pub(crate) fn try_acquire_decision(self: &Arc<Self>) -> Option<DecisionPermit> {
        loop {
            if self.is_paused() {
                return None;
            }
            let epoch = self.routing_epoch();
            self.active_decisions.fetch_add(1, Ordering::AcqRel);
            if !self.is_paused() && self.routing_epoch() == epoch {
                gauge!("smg_routing_loop_running_selections")
                    .set(self.active_decisions.load(Ordering::Acquire) as f64);
                return Some(DecisionPermit {
                    runtime: Arc::clone(self),
                    epoch,
                });
            }
            self.release_decision();
        }
    }

    /// Admit the commit/backend-dispatch handoff for a previously made decision.
    pub(crate) fn try_acquire_dispatch_handoff(
        self: &Arc<Self>,
        epoch: u64,
    ) -> Option<DispatchHandoffPermit> {
        if self.is_paused() || self.routing_epoch() != epoch {
            return None;
        }
        self.active_dispatch_handoffs.fetch_add(1, Ordering::AcqRel);
        if self.is_paused() || self.routing_epoch() != epoch {
            self.release_dispatch_handoff();
            return None;
        }
        Some(DispatchHandoffPermit {
            runtime: Arc::clone(self),
        })
    }

    fn release_decision(&self) {
        self.active_decisions.fetch_sub(1, Ordering::AcqRel);
        gauge!("smg_routing_loop_running_selections")
            .set(self.active_decisions.load(Ordering::Acquire) as f64);
        self.admission_notify.notify_waiters();
    }

    fn release_dispatch_handoff(&self) {
        self.active_dispatch_handoffs.fetch_sub(1, Ordering::AcqRel);
        self.admission_notify.notify_waiters();
    }

    pub(crate) async fn wait_for_pause_barrier(&self) {
        loop {
            let notified = self.admission_notify.notified();
            if !self.is_routing() {
                return;
            }
            notified.await;
        }
    }

    pub(crate) async fn status(&self) -> RoutingLoopStatus {
        let queue = self.queue.lock().await;
        let queue_len = queue.len();
        RoutingLoopStatus {
            enabled: true,
            paused: self.is_paused(),
            routing: self.is_routing(),
            selecting: self.is_selecting(),
            routing_epoch: self.routing_epoch(),
            active_dispatch_handoffs: self.active_dispatch_handoffs.load(Ordering::Acquire),
            queue_len,
            pending_request_num: queue_len,
            running_tasks: self.running_tasks.load(Ordering::Acquire),
            running_request_num: self.running_requests.load(Ordering::Acquire),
            max_running_dispatch_tasks: self.max_running_dispatch_tasks,
            queue_keys: queue.queue_keys(),
            partition_queue_lens: queue.per_partition_sizes(),
        }
    }

    /// Return metadata for all queued entries whose `version_tag ≤ max_version_tag`.
    ///
    /// Used by `GET /routing_loop/filter` to let the sync coordinator check
    /// whether old-version requests have fully drained from the queue.
    pub(crate) async fn filter_queue_by_version_tag(
        &self,
        max_version_tag: i64,
    ) -> Vec<QueuedRequestMeta> {
        let queue = self.queue.lock().await;
        queue
            .iter_requests()
            .filter_map(|entry| {
                entry.routing_meta.as_ref().and_then(|meta| {
                    (meta.version_tag <= max_version_tag).then_some(QueuedRequestMeta {
                        request_id: meta.request_id,
                        version_tag: meta.version_tag,
                        prompt_id: meta.prompt_id,
                    })
                })
            })
            .collect()
    }

    async fn push_entries(&self, entries: Vec<RoutingQueueEntry>) {
        let mut queue = self.queue.lock().await;
        for entry in entries {
            queue.push(entry);
        }
        let queue_len = queue.len() as f64;
        // Update per-version-partition gauges.
        for (key, size) in queue.per_partition_sizes() {
            gauge!(
                "smg_routing_loop_partition_queue_length",
                "version" => key.to_string(),
            )
            .set(size as f64);
        }
        gauge!("smg_routing_loop_queue_length").set(queue_len);
    }

    async fn pop_entries(&self, max_entries: usize) -> Vec<RoutingQueueEntry> {
        let mut queue = self.queue.lock().await;
        let mut entries = Vec::with_capacity(max_entries);
        for _ in 0..max_entries {
            let Some(entry) = queue.pop() else { break };
            entries.push(entry);
        }
        queue.remove_empty_partitions();
        gauge!("smg_routing_loop_queue_length").set(queue.len() as f64);
        entries
    }

    fn idle_interval(&self) -> Duration {
        Duration::from_millis(self.check_interval_ms.max(1))
    }

    fn task_started(&self) {
        let running = self.running_tasks.fetch_add(1, Ordering::AcqRel) + 1;
        gauge!("smg_routing_loop_running_tasks").set(running as f64);
    }

    fn task_finished(&self) {
        let previous = self.running_tasks.fetch_sub(1, Ordering::AcqRel);
        let running = if previous > 0 { previous - 1 } else { 0 };
        gauge!("smg_routing_loop_running_tasks").set(running as f64);
    }
}

pub(crate) async fn run_routing_loop(
    runtime: Arc<RoutingLoopRuntime>,
    mut rx: mpsc::UnboundedReceiver<RoutingQueueEntry>,
) {
    let mut dispatch_tasks = JoinSet::new();

    loop {
        while let Some(join_result) = dispatch_tasks.try_join_next() {
            if let Err(err) = join_result {
                error!(error = %err, "routing loop dispatch task failed");
            }
        }

        let entries = drain_receiver(&mut rx, runtime.receive_batch_size);
        if !entries.is_empty() {
            runtime.push_entries(entries).await;
        }

        if runtime.is_paused() {
            tokio::select! {
                maybe_entry = rx.recv() => {
                    match maybe_entry {
                        Some(entry) => {
                            let mut entries = Vec::with_capacity(runtime.receive_batch_size);
                            entries.push(entry);
                            entries.extend(drain_receiver(&mut rx, runtime.receive_batch_size - 1));
                            runtime.push_entries(entries).await;
                        }
                        None => break,
                    }
                }
                () = runtime.admission_notify.notified() => {}
                () = sleep(runtime.idle_interval()) => {}
            }
            continue;
        }

        let running_tasks = runtime.running_tasks.load(Ordering::Acquire);
        if running_tasks >= runtime.max_running_dispatch_tasks {
            tokio::select! {
                maybe_entry = rx.recv() => {
                    match maybe_entry {
                        Some(entry) => {
                            let mut entries = Vec::with_capacity(runtime.receive_batch_size);
                            entries.push(entry);
                            entries.extend(drain_receiver(&mut rx, runtime.receive_batch_size - 1));
                            runtime.push_entries(entries).await;
                        }
                        None => break,
                    }
                }
                join_result = dispatch_tasks.join_next(), if !dispatch_tasks.is_empty() => {
                    if let Some(Err(err)) = join_result {
                        error!(error = %err, "routing loop dispatch task failed");
                    }
                }
                () = sleep(runtime.idle_interval()) => {}
            }
            continue;
        }

        let dispatch_limit = runtime
            .dispatch_batch_size
            .min(runtime.max_running_dispatch_tasks - running_tasks);
        let entries = runtime.pop_entries(dispatch_limit).await;
        let dispatched = !entries.is_empty();
        let aborted_ids = check_aborted_requests(&runtime, &entries).await;
        for entry in entries {
            let entry_request_id = entry.routing_meta.as_ref().map(|meta| meta.request_id);
            if entry_request_id.is_some_and(|request_id| aborted_ids.contains(&request_id)) {
                let prompt_id = entry.routing_meta.as_ref().map(|meta| meta.prompt_id);
                send_aborted(entry);
                runtime.cleanup_tracking(entry_request_id, prompt_id);
                continue;
            }

            let runtime_for_task = Arc::clone(&runtime);
            runtime_for_task.task_started();
            dispatch_tasks.spawn(async move {
                let dispatch_start = std::time::Instant::now();
                dispatch_entry(Arc::clone(&runtime_for_task), entry).await;
                histogram!("smg_routing_loop_dispatch_duration_seconds")
                    .record(dispatch_start.elapsed().as_secs_f64());
                runtime_for_task.task_finished();
            });
        }
        if dispatched {
            yield_now().await;
            continue;
        }

        tokio::select! {
            maybe_entry = rx.recv() => {
                match maybe_entry {
                    Some(entry) => {
                        let mut entries = Vec::with_capacity(runtime.receive_batch_size);
                        entries.push(entry);
                        entries.extend(drain_receiver(&mut rx, runtime.receive_batch_size - 1));
                        runtime.push_entries(entries).await;
                    }
                    None => break,
                }
            }
            join_result = dispatch_tasks.join_next(), if !dispatch_tasks.is_empty() => {
                if let Some(Err(err)) = join_result {
                    error!(error = %err, "routing loop dispatch task failed");
                }
            }
            () = sleep(runtime.idle_interval()) => {}
        }
    }

    while let Some(join_result) = dispatch_tasks.join_next().await {
        if let Err(err) = join_result {
            error!(error = %err, "routing loop dispatch task failed during shutdown");
        }
    }
}

async fn check_aborted_requests(
    runtime: &Arc<RoutingLoopRuntime>,
    entries: &[RoutingQueueEntry],
) -> HashSet<i64> {
    let Some(ps_client) = runtime.ps_manager_client.get() else {
        return HashSet::new();
    };

    let request_ids: Vec<i64> = entries
        .iter()
        .filter_map(|entry| entry.routing_meta.as_ref().map(|meta| meta.request_id))
        .collect();
    if request_ids.is_empty() {
        return HashSet::new();
    }

    match ps_client
        .check_aborted_requests(request_ids.clone(), true)
        .await
    {
        Ok(flags) => request_ids
            .into_iter()
            .zip(flags)
            .filter_map(|(request_id, aborted)| aborted.then_some(request_id))
            .collect(),
        Err(err) => {
            error!(error = %err, "check_aborted_requests RPC failed; continuing dispatch");
            HashSet::new()
        }
    }
}

fn send_aborted(entry: RoutingQueueEntry) {
    let response = router_error::bad_request("request_aborted", "Request aborted by PS Manager");
    match entry.completion {
        RoutingLoopCompletion::Http(tx) => {
            let _ = tx.send(response);
        }
        RoutingLoopCompletion::ChatForResponses(tx) => {
            let _ = tx.send(Err(response));
        }
        RoutingLoopCompletion::HarmonyResponses(tx) => {
            let _ = tx.send(Err(response));
        }
        RoutingLoopCompletion::HarmonyResponsesStreaming(tx) => {
            let _ = tx.send(Err(response));
        }
    }
}

async fn dispatch_entry(runtime: Arc<RoutingLoopRuntime>, mut entry: RoutingQueueEntry) {
    let request_id = entry.routing_meta.as_ref().map(|meta| meta.request_id);
    let prompt_id = entry.routing_meta.as_ref().map(|meta| meta.prompt_id);

    if entry.routing_meta.is_none() {
        let Some(decision_permit) = runtime.try_acquire_decision() else {
            runtime.push_entries(vec![entry]).await;
            return;
        };
        let decision_epoch = decision_permit.epoch();
        let select_result = entry
            .pipeline
            .execute_worker_selection(&mut entry.ctx)
            .await;
        drop(decision_permit);
        if select_result.is_err() {
            runtime.push_entries(vec![entry]).await;
            return;
        }

        let Some(handoff_permit) = runtime.try_acquire_dispatch_handoff(decision_epoch) else {
            entry.ctx.state.workers = None;
            entry.ctx.state.load_guards = None; // drop guard before re-enqueue
            runtime.push_entries(vec![entry]).await;
            return;
        };
        if let Err(response) = entry.pipeline.commit_worker_selection(&mut entry.ctx).await {
            drop(handoff_permit);
            send_completion_error(entry.completion, response);
            runtime.cleanup_tracking(request_id, prompt_id);
            return;
        }
        if let Err(response) = entry
            .pipeline
            .execute_post_selection_execution(&mut entry.ctx)
            .await
        {
            drop(handoff_permit);
            send_completion_error(entry.completion, response);
            runtime.cleanup_tracking(request_id, prompt_id);
            return;
        }
        drop(handoff_permit);

        // ── Execution succeeded — run post-execution stages and send result ──
        match entry.completion {
            RoutingLoopCompletion::Http(tx) => {
                let response = match entry
                    .pipeline
                    .execute_remaining_stages(&mut entry.ctx)
                    .await
                {
                    Ok(Some(r)) => r,
                    Ok(None) => extract_final_response(&mut entry.ctx),
                    Err(r) => r,
                };
                let _ = tx.send(response);
            }
            RoutingLoopCompletion::ChatForResponses(tx) => {
                let result = match entry
                    .pipeline
                    .execute_remaining_stages(&mut entry.ctx)
                    .await
                {
                    Ok(_) => match entry.ctx.state.response.final_response.take() {
                        Some(FinalResponse::Chat(r)) => Ok(r),
                        Some(_) => Err(router_error::internal_error(
                            "wrong_response_type",
                            "Wrong response type for ChatForResponses",
                        )),
                        None => Err(router_error::internal_error(
                            "no_response_produced",
                            "No response produced",
                        )),
                    },
                    Err(r) => Err(r),
                };
                let _ = tx.send(result);
            }
            RoutingLoopCompletion::HarmonyResponses(tx) => {
                let result = match entry
                    .pipeline
                    .execute_remaining_stages(&mut entry.ctx)
                    .await
                {
                    Ok(_) => entry
                        .ctx
                        .state
                        .response
                        .responses_iteration_result
                        .take()
                        .ok_or_else(|| {
                            router_error::internal_error(
                                "no_responses_iteration_result",
                                "No ResponsesIterationResult produced",
                            )
                        }),
                    Err(r) => Err(r),
                };
                let _ = tx.send(result);
            }
            RoutingLoopCompletion::HarmonyResponsesStreaming(tx) => {
                let result = match entry
                    .pipeline
                    .execute_remaining_stages(&mut entry.ctx)
                    .await
                {
                    Ok(_) => match entry.ctx.state.response.execution_result.take() {
                        Some(execution_result) => {
                            let load_guards = entry.ctx.state.load_guards.take();
                            Ok((execution_result, load_guards))
                        }
                        None => Err(router_error::internal_error(
                            "no_execution_result",
                            "No execution result produced",
                        )),
                    },
                    Err(r) => Err(r),
                };
                let _ = tx.send(result);
            }
        }
        runtime.cleanup_tracking(request_id, prompt_id);
    } else {
        dispatch_entry_with_partial_rollout(runtime, entry, request_id, prompt_id).await;
    }
}

fn send_completion_error(completion: RoutingLoopCompletion, response: Response) {
    match completion {
        RoutingLoopCompletion::Http(tx) => {
            let _ = tx.send(response);
        }
        RoutingLoopCompletion::ChatForResponses(tx) => {
            let _ = tx.send(Err(response));
        }
        RoutingLoopCompletion::HarmonyResponses(tx) => {
            let _ = tx.send(Err(response));
        }
        RoutingLoopCompletion::HarmonyResponsesStreaming(tx) => {
            let _ = tx.send(Err(response));
        }
    }
}

/// Extract a `Response` from `ctx.state.response.final_response`
fn extract_final_response(ctx: &mut RequestContext) -> Response {
    match ctx.state.response.final_response.take() {
        Some(FinalResponse::Chat(r)) => axum::Json(r).into_response(),
        Some(FinalResponse::Generate(r)) => axum::Json(r).into_response(),
        Some(FinalResponse::Completion(r)) => axum::Json(r).into_response(),
        Some(FinalResponse::Embedding(r)) => axum::Json(r).into_response(),
        Some(FinalResponse::Classify(r)) => axum::Json(r).into_response(),
        Some(FinalResponse::Messages(r)) => axum::Json(r).into_response(),
        None => router_error::internal_error("no_response_produced", "No response produced"),
    }
}

/// Send the final `Response` through the appropriate completion channel.
fn send_http_completion(completion: RoutingLoopCompletion, response: Response) {
    match completion {
        RoutingLoopCompletion::Http(tx) => {
            let _ = tx.send(response);
        }
        // PSRL dispatch only uses Http completion; other variants are not
        // reachable here, but we log an error rather than panic.
        other => {
            error!(
                "dispatch_entry_with_partial_rollout: unexpected completion type; dropping response"
            );
            drop(other);
        }
    }
}

/// Emit a `pop` decision trace when a request leaves the router: either
/// finished (`completed`) or temporarily pulled out by a partial-rollout /
/// migration interruption (`interrupted`). Uses the dedicated `route_trace`
/// target so it lands in its own log file.
fn log_pop(
    runtime: &RoutingLoopRuntime,
    worker: &Arc<dyn Worker>,
    meta: Option<&RoutingMeta>,
    prompt_len: u32,
    response_tokens: usize,
    status: &str,
    finish_reason: &str,
    iteration_count: u32,
) {
    let (base_id, dp) = runtime.instance_id_for_worker(worker);
    let stats = worker.engine_stats().scheduler_stats;
    info!(
        target: "route_trace",
        event = "pop",
        request_id = meta.map(|m| m.request_id),
        prompt_id = meta.map(|m| m.prompt_id),
        version_tag = meta.map(|m| m.version_tag),
        prompt_tokens = prompt_len,
        response_tokens,
        total_tokens = prompt_len as usize + response_tokens,
        instance = %base_id,
        dp_rank = dp,
        running = stats.num_running_reqs,
        waiting = stats.num_waiting_reqs,
        kv_cache_usage = stats.kv_cache_usage,
        status,
        finish_reason,
        iteration_count,
        "pop"
    );
}

/// Full partial-rollout loopback loop for PSRL requests.
///
/// Runs pipeline stages through execution on each iteration.  If the stream
/// finishes with `"abort"` (PS weight-sync interrupted generation), the
/// accumulated tokens are preserved and the request is re-routed to the
/// newly-synced instance.  The loop terminates on `"stop"` / `"length"`.
async fn dispatch_entry_with_partial_rollout(
    runtime: Arc<RoutingLoopRuntime>,
    entry: RoutingQueueEntry,
    request_id: Option<i64>,
    prompt_id: Option<i64>,
) {
    let RoutingQueueEntry {
        mut ctx,
        pipeline,
        completion,
        mut routing_meta,
    } = entry;

    let mut partial_state = ctx.state.partial_rollout_state.take().unwrap_or_default();

    if partial_state.iteration_count == 0 {
        partial_state.first_iter_prompt_start = ctx
            .state
            .partial_rollout_overrides
            .routed_experts_prompt_start
            .unwrap_or_else(|| ctx.input.request_type.routed_experts_prompt_start());
        partial_state.prompt_len = ctx
            .state
            .preparation
            .as_ref()
            .map(|p| p.token_ids().len() as u32)
            .unwrap_or(0);
    }

    // Snapshot `preparation` before the loop starts
    ctx.state.preparation_snapshot = ctx.state.preparation.clone();

    // ── Step 1: publish accumulated response-token count for selection ───
    let headers = ctx.input.headers.get_or_insert_with(Default::default);
    if let Ok(v) = HeaderValue::from_str(&partial_state.response_token_count().to_string()) {
        headers.insert("x-response-token-count", v);
    }

    // ── Step 2: run decision, fenced commit, and backend dispatch ────────
    let Some(decision_permit) = runtime.try_acquire_decision() else {
        ctx.state.partial_rollout_state = Some(partial_state);
        runtime
            .push_entries(vec![RoutingQueueEntry {
                ctx,
                pipeline,
                completion,
                routing_meta,
            }])
            .await;
        return;
    };
    let decision_epoch = decision_permit.epoch();
    let select_result = pipeline.execute_worker_selection(&mut ctx).await;
    drop(decision_permit);
    if select_result.is_err() {
        ctx.state.partial_rollout_state = Some(partial_state);
        runtime
            .push_entries(vec![RoutingQueueEntry {
                ctx,
                pipeline,
                completion,
                routing_meta,
            }])
            .await;
        return;
    }

    let Some(handoff_permit) = runtime.try_acquire_dispatch_handoff(decision_epoch) else {
        ctx.state.workers = None;
        ctx.state.load_guards = None; // drop guard before re-enqueue
        ctx.state.partial_rollout_state = Some(partial_state);
        runtime
            .push_entries(vec![RoutingQueueEntry {
                ctx,
                pipeline,
                completion,
                routing_meta,
            }])
            .await;
        return;
    };
    if let Err(response) = pipeline.commit_worker_selection(&mut ctx).await {
        drop(handoff_permit);
        send_http_completion(completion, response);
        runtime.cleanup_tracking(request_id, prompt_id);
        return;
    }
    // Commit may have pinned `x-version-tag` (an unversioned `-1` request gets
    // pinned to the selected instance's version). Refresh `routing_meta` from
    // the context headers so the pop/route traces and any re-enqueue carry the
    // pinned version instead of the stale `-1`.
    if let Some(refreshed) = parse_routing_request_meta_from_context(&ctx) {
        routing_meta = Some(refreshed);
    }
    if let Err(response) = pipeline.execute_post_selection_execution(&mut ctx).await {
        drop(handoff_permit);
        send_http_completion(completion, response);
        runtime.cleanup_tracking(request_id, prompt_id);
        return;
    }
    drop(handoff_permit);

    // ── Step 3: extract the raw gRPC stream ──────────────────────────────
    let stream = match ctx.state.response.execution_result.take() {
        Some(ExecutionResult::Single { stream }) => stream,
        Some(ExecutionResult::Dual { decode, .. }) => *decode,
        other => {
            // Non-generate result (embedding, etc.) — put it back and run
            // post-execution stages normally (no loopback needed).
            ctx.state.response.execution_result = other;
            let response = match pipeline.execute_remaining_stages(&mut ctx).await {
                Ok(Some(r)) => r,
                Ok(None) => extract_final_response(&mut ctx),
                Err(r) => r,
            };
            send_http_completion(completion, response);
            runtime.cleanup_tracking(request_id, prompt_id);
            return;
        }
    };

    // ── Step 4: drain the stream, collecting tokens and finish_reason ────
    let mut stream = stream;
    let drained = match drain_stream_for_partial_rollout(&mut stream).await {
        Ok(d) => d,
        Err(msg) => {
            error!(
                function = "dispatch_entry_with_partial_rollout",
                request_id = request_id,
                prompt_id = prompt_id,
                error = %msg,
                "Partial rollout stream drain failed"
            );
            let response = router_error::internal_error("stream_drain_failed", msg.as_str());
            send_http_completion(completion, response);
            runtime.cleanup_tracking(request_id, prompt_id);
            return;
        }
    };

    // ── Step 5: branch on finish_reason ─────────────────────────────────
    match drained.finish_reason.as_str() {
        "stop" | "length" => {
            // ── ROLLOUT_COMPLETED ────────────────────────────────────────
            if let Err(re_err) = merge_into_partial_state(&mut partial_state, &drained) {
                let response =
                    router_error::internal_error(re_err.error_code(), re_err.to_string().as_str());
                send_http_completion(completion, response);
                runtime.cleanup_tracking(request_id, prompt_id);
                return;
            }

            // Emit completion metrics.
            histogram!("smg_partial_rollout_loopback_count")
                .record(partial_state.iteration_count as f64);
            let accumulated_tokens = partial_state.token_ids.len();
            histogram!("smg_partial_rollout_accumulated_tokens").record(accumulated_tokens as f64);
            counter!("smg_partial_rollout_completed_total").increment(1);

            // Pop trace: request finished. `ctx.state.workers` still holds the
            // serving worker here (execute_remaining_stages resets it below).
            if let Some(worker) = ctx.state.workers.as_ref().map(|sel| match sel {
                WorkerSelection::Single { worker } => worker.clone(),
                WorkerSelection::Dual { decode, .. } => decode.clone(),
            }) {
                log_pop(
                    &runtime,
                    &worker,
                    routing_meta.as_ref(),
                    partial_state.prompt_len,
                    accumulated_tokens,
                    "completed",
                    drained.finish_reason.as_str(),
                    partial_state.iteration_count,
                );
            }

            // Override the output_ids in the final complete frame with the
            // full accumulated token sequence from all loopback iterations.
            //
            // Multi-iteration *or* RE opt-in: rewrite `output_logprobs`
            // and `routed_experts` together so the frame is internally
            // consistent.
            let mut complete = drained.complete;
            let merged_token_ids = std::mem::take(&mut partial_state.token_ids);
            let needs_full_rewrite =
                partial_state.iteration_count > 1 || partial_state.routed_experts.is_some();
            if needs_full_rewrite {
                let merged_re = partial_state
                    .routed_experts
                    .take()
                    .map(|acc| acc.into_proto());
                if let Err(re_err) = complete.set_partial_rollout_outputs(
                    merged_token_ids,
                    partial_state.logprobs.take(),
                    merged_re,
                    partial_state.prompt_len,
                    partial_state.first_iter_prompt_start,
                ) {
                    let response = router_error::internal_error(
                        re_err.error_code(),
                        re_err.to_string().as_str(),
                    );
                    send_http_completion(completion, response);
                    runtime.cleanup_tracking(request_id, prompt_id);
                    return;
                }
            } else {
                complete.set_output_ids(merged_token_ids);
            }

            // Place the assembled complete frame back for PostExecution stages.
            ctx.state.response.execution_result = Some(ExecutionResult::Complete(complete));

            // Capture the served instance BEFORE execute_remaining_stages, which
            // resets ctx.state.workers. The instance id is echoed back as response
            // headers so the SessionRouter can pin subsequent turns of the same
            // trajectory to this instance (trajectory sticky).
            let served_instance = ctx
                .state
                .workers
                .as_ref()
                .map(|sel| match sel {
                    WorkerSelection::Single { worker } => worker.clone(),
                    WorkerSelection::Dual { decode, .. } => decode.clone(),
                })
                .map(|worker| runtime.instance_id_for_worker(&worker));

            let mut response = match pipeline.execute_remaining_stages(&mut ctx).await {
                Ok(Some(r)) => r,
                Ok(None) => extract_final_response(&mut ctx),
                Err(r) => r,
            };
            if let Some((base_id, dp_rank)) = served_instance {
                let headers = response.headers_mut();
                if let Ok(v) = HeaderValue::from_str(&base_id) {
                    headers.insert("x-base-worker-id", v);
                }
                if let Ok(v) = HeaderValue::from_str(&dp_rank.to_string()) {
                    headers.insert("x-target-dp-rank", v);
                }
                // Echo the pinned version so the SessionRouter can carry it into
                // the next turn of the trajectory (keeping the request on an
                // instance at least as fresh as the pinned version).
                if let Some(version_tag) = routing_meta.as_ref().map(|m| m.version_tag) {
                    if let Ok(v) = HeaderValue::from_str(&version_tag.to_string()) {
                        headers.insert("x-version-tag", v);
                    }
                }
                debug!(
                    request_id = ?request_id,
                    base_worker_id = %base_id,
                    target_dp_rank = dp_rank,
                    "PSRL trajectory sticky: echoed served instance to response headers"
                );
            }
            send_http_completion(completion, response);
            runtime.cleanup_tracking(request_id, prompt_id);
        }
        "abort" => {
            // ── ROLLOUT_INTERRUPTED — loopback to newly-synced instance ──
            if let Err(re_err) = merge_into_partial_state(&mut partial_state, &drained) {
                let response =
                    router_error::internal_error(re_err.error_code(), re_err.to_string().as_str());
                send_http_completion(completion, response);
                runtime.cleanup_tracking(request_id, prompt_id);
                return;
            }
            counter!("smg_partial_rollout_abort_total").increment(1);

            // Determine the instance that was just used so we can pin the
            // loopback request to the same worker (which now has the fresh
            // weights after the sync that triggered the abort).
            let worker = match ctx.state.workers.as_ref() {
                Some(WorkerSelection::Single { worker }) => worker.clone(),
                Some(WorkerSelection::Dual { decode, .. }) => decode.clone(),
                None => {
                    let response =
                        router_error::internal_error("loopback_no_instance", "no worker selected");
                    send_http_completion(completion, response);
                    runtime.cleanup_tracking(request_id, prompt_id);
                    return;
                }
            };
            let (base_id, dp_rank) = runtime.instance_id_for_worker(&worker);

            // Pop trace: request temporarily pulled out for partial rollout /
            // migration loopback. Accumulated response length so far is echoed.
            log_pop(
                &runtime,
                &worker,
                routing_meta.as_ref(),
                partial_state.prompt_len,
                partial_state.token_ids.len(),
                "interrupted",
                "abort",
                partial_state.iteration_count,
            );

            // Reset ctx for the next iteration (clears workers, clients,
            // proto_request, dispatch, load_guards, and execution_result).
            reset_ctx_for_loopback(&mut ctx);

            // Inject the loopback `routed_experts_prompt_start` override
            // so the next iteration's vLLM `SamplingParams` only captures
            // RE for token positions not already covered by prior
            // iterations' segments.
            let prompt_start_next = partial_state.first_iter_prompt_start
                + partial_state
                    .routed_experts
                    .as_ref()
                    .map(|acc| acc.num_tokens() as u32)
                    .unwrap_or(0);
            ctx.state
                .partial_rollout_overrides
                .routed_experts_prompt_start = Some(prompt_start_next);

            // Inject loopback routing headers so that on the next iteration
            // `parse_routing_request_meta_from_context` picks up the hint.
            let headers = ctx.input.headers.get_or_insert_with(Default::default);
            if let Ok(v) = HeaderValue::from_str(&base_id) {
                headers.insert("x-base-worker-id", v);
            }
            if let Ok(v) = HeaderValue::from_str(&dp_rank.to_string()) {
                headers.insert("x-target-dp-rank", v);
            }
            // Clear the one-shot pin: `force_pin_once` only pins the FIRST
            // selection of a turn. Once this request loops back (partial rollout
            // / preemption re-dispatch), the scheduler's readmission intent has
            // been honoured, so subsequent iterations fall back to free routing.
            headers.remove("x-force-pin-once");

            counter!("smg_partial_rollout_abort_reenqueue_total").increment(1);
            ctx.state.partial_rollout_state = Some(partial_state);
            runtime
                .push_entries(vec![RoutingQueueEntry {
                    ctx,
                    pipeline,
                    completion,
                    routing_meta,
                }])
                .await;
        }
        other => {
            error!(
                finish_reason = other,
                request_id = ?request_id,
                "unexpected finish_reason in partial rollout; terminating"
            );
            let response = router_error::internal_error("unexpected_finish_reason", other);
            send_http_completion(completion, response);
            runtime.cleanup_tracking(request_id, prompt_id);
        }
    }
}

fn drain_receiver(
    rx: &mut mpsc::UnboundedReceiver<RoutingQueueEntry>,
    max_entries: usize,
) -> Vec<RoutingQueueEntry> {
    let mut entries = Vec::with_capacity(max_entries);
    for _ in 0..max_entries {
        let Ok(entry) = rx.try_recv() else {
            break;
        };
        entries.push(entry);
    }
    entries
}

pub(crate) struct DecisionPermit {
    runtime: Arc<RoutingLoopRuntime>,
    epoch: u64,
}

impl DecisionPermit {
    pub(crate) fn epoch(&self) -> u64 {
        self.epoch
    }
}

impl Drop for DecisionPermit {
    fn drop(&mut self) {
        self.runtime.release_decision();
    }
}

pub(crate) struct DispatchHandoffPermit {
    runtime: Arc<RoutingLoopRuntime>,
}

impl Drop for DispatchHandoffPermit {
    fn drop(&mut self) {
        self.runtime.release_dispatch_handoff();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_runtime() -> Arc<RoutingLoopRuntime> {
        RoutingLoopRuntime::new(
            &RoutingLoopConfig::default(),
            Arc::new(DashMap::new()),
            Arc::new(WorkerRegistry::new()),
        )
        .0
    }

    #[tokio::test]
    async fn pause_barrier_waits_for_active_decision() {
        let runtime = make_runtime();
        let permit = runtime
            .try_acquire_decision()
            .expect("decision should be admitted");
        runtime.pause();

        assert!(
            tokio::time::timeout(Duration::from_millis(10), runtime.wait_for_pause_barrier())
                .await
                .is_err()
        );

        drop(permit);
        tokio::time::timeout(Duration::from_secs(1), runtime.wait_for_pause_barrier())
            .await
            .expect("pause barrier should complete after decision exits");
    }

    #[tokio::test]
    async fn pause_barrier_waits_for_active_dispatch_handoff() {
        let runtime = make_runtime();
        let decision = runtime
            .try_acquire_decision()
            .expect("decision should be admitted");
        let epoch = decision.epoch();
        drop(decision);
        let handoff = runtime
            .try_acquire_dispatch_handoff(epoch)
            .expect("handoff should be admitted");
        runtime.pause();

        assert!(
            tokio::time::timeout(Duration::from_millis(10), runtime.wait_for_pause_barrier())
                .await
                .is_err()
        );

        drop(handoff);
        tokio::time::timeout(Duration::from_secs(1), runtime.wait_for_pause_barrier())
            .await
            .expect("pause barrier should complete after handoff exits");
    }

    #[test]
    fn pause_invalidates_old_decision_before_handoff() {
        let runtime = make_runtime();
        let permit = runtime
            .try_acquire_decision()
            .expect("decision should be admitted");
        let epoch = permit.epoch();
        drop(permit);

        runtime.pause();
        assert!(runtime.try_acquire_dispatch_handoff(epoch).is_none());
        assert!(runtime.try_acquire_decision().is_none());

        runtime.resume();
        assert!(runtime.try_acquire_decision().is_some());
    }
}
