//! Partial-rollout state management and stream drain utilities.
//!
//! When a PS weight-sync interrupts generation (`finish_reason == "abort"`), the
//! accumulated tokens from the aborted stream are preserved here so the
//! re-routed request can continue generation from the right offset.
//!
//! # Life-cycle
//!
//! 1. `dispatch_entry_with_partial_rollout` detects `is_psrl = true` and enters
//!    the loopback loop.
//! 2. After each execution, `drain_stream_for_partial_rollout` consumes the
//!    raw gRPC stream into a `DrainedStreamResult`.
//! 3. If `finish_reason == "abort"`:
//!    - `merge_into_partial_state` appends the new tokens to `PartialRolloutState`.
//!    - `reset_ctx_for_loopback` rebuilds `ctx` for the next iteration.
//! 4. Otherwise the loop exits and the caller forwards the result to the client.
//!
//! # Routed experts contract
//!
//! When the request opted in to `return_routed_experts`, every iteration's
//! `Complete` frame carries a `routed_experts` segment whose token count equals
//! `(prompt_len_iter - prompt_start_iter) + completion_len_iter - 1` (the
//! `-1` accounts for the final sampled token having no captured RE — vLLM
//! invariant).  Across loopback iterations the segments are concatenated
//! rather than overlapped:
//!     (`prompt_start_iter_{k+1} = first_iter_prompt_start + accumulator.num_tokens()`)
//! ensures vLLM only captures *new* positions on each restart.
//! The merge logic in this module is therefore a pure append;
//! misalignments / late-arrival / shape changes are bug paths
//! and surface as fail-hard `RoutedExpertsError`.

use tracing::{info, warn};

use crate::routers::grpc::{
    context::RequestContext,
    proto_wrapper::{
        ProtoGenerateComplete, ProtoResponseVariant, ProtoRoutedExperts, ProtoRoutedExpertsShape,
        ProtoStream, RoutedExpertsDtype, RoutedExpertsError,
    },
};

/// Append-only accumulator for routed-experts bytes across loopback iterations.
#[derive(Debug, Clone)]
pub(crate) struct RoutedExpertsAccumulator {
    pub data: Vec<u8>,
    pub num_layers: u32,
    pub top_k: u32,
    pub dtype: RoutedExpertsDtype,
    pub index: u32,
}

impl RoutedExpertsAccumulator {
    /// Seed the accumulator from the first non-empty segment.
    pub fn from_segment(seg: &ProtoRoutedExperts, capacity_hint_bytes: usize) -> Self {
        let cap = capacity_hint_bytes.max(seg.data.len());
        let mut data = Vec::with_capacity(cap);
        data.extend_from_slice(&seg.data);
        Self {
            data,
            num_layers: seg.num_layers,
            top_k: seg.top_k,
            dtype: seg.dtype,
            index: seg.index,
        }
    }

    #[inline]
    pub const fn token_bytes(&self) -> usize {
        self.num_layers as usize * self.top_k as usize * self.dtype.size()
    }

    /// Number of tokens (= token positions covered) accumulated so far.
    #[inline]
    pub fn num_tokens(&self) -> usize {
        let r = self.token_bytes();
        if r == 0 {
            0
        } else {
            self.data.len() / r
        }
    }

    /// Shape descriptor used in error reporting.
    #[inline]
    pub const fn shape(&self) -> ProtoRoutedExpertsShape {
        ProtoRoutedExpertsShape {
            num_layers: self.num_layers,
            top_k: self.top_k,
            dtype: self.dtype,
            index: self.index,
        }
    }

    /// Whether `seg` is shape-compatible with the accumulator and can be
    /// appended in-place.
    #[inline]
    pub fn shape_compatible(&self, seg: &ProtoRoutedExperts) -> bool {
        self.num_layers == seg.num_layers
            && self.top_k == seg.top_k
            && self.dtype == seg.dtype
            && self.index == seg.index
    }

    /// Convert into a wire-form `ProtoRoutedExperts` for terminal-frame rewrite.
    pub fn into_proto(self) -> ProtoRoutedExperts {
        ProtoRoutedExperts {
            data: bytes::Bytes::from(self.data),
            num_layers: self.num_layers,
            top_k: self.top_k,
            dtype: self.dtype,
            index: self.index,
        }
    }
}

/// Accumulated state carried across partial-rollout loopback iterations.
#[derive(Debug, Clone)]
pub(crate) struct PartialRolloutState {
    /// All output token IDs accumulated so far across every loopback iteration.
    pub token_ids: Vec<u32>,
    /// Per-token log-probabilities, parallel to `token_ids`.
    /// `None` if any iteration did not return log-prob data.
    pub logprobs: Option<Vec<f32>>,
    /// Routed-experts accumulator.  `None` when the request did not opt in,
    /// or before the first iteration that produces a non-empty RE segment.
    pub routed_experts: Option<RoutedExpertsAccumulator>,
    /// Number of loopback iterations completed so far (excluding the final one).
    ///
    /// Incremented by `merge_into_partial_state` on each `"abort"` cycle.
    /// At completion this equals the number of PS weight-sync interruptions.
    pub iteration_count: u32,
    /// The caller's generation cap, captured once at iteration 1.
    ///
    /// A loopback folds the accumulated output into the prompt, so each
    /// continuation must be charged `original_max_tokens - tokens_produced`
    /// rather than the original cap, which was sized for a sequence that had
    /// generated nothing. `None` when the caller set no cap.
    pub original_max_tokens: Option<u32>,

    // ─── Routed-experts metadata (immutable across loopback) ───────────
    /// Original prompt length at iter 1 dispatch time, before any loopback
    /// augmented the prompt with accumulated completion tokens.
    pub prompt_len: u32,
    /// `routed_experts_prompt_start` from the very first iteration's request
    /// (default 0 when not supplied).  Captured once at iter 1 entry and
    /// never overwritten.  Used both as the first-iter dispatch
    /// `prompt_start` and as the offset baseline for the loopback formula
    /// `prompt_start_iter_{k+1} = first_iter_prompt_start + accumulator.num_tokens()`.
    pub first_iter_prompt_start: u32,
    /// Capacity hint bytes for the RE accumulator, sized once at dispatch
    /// from `(prompt_len + max_new_tokens - first_iter_prompt_start) * token_bytes`.
    /// Avoids reallocations on long rollouts.  `0` when the request did not
    /// opt in to RE.
    pub expected_final_re_bytes: usize,
}

impl Default for PartialRolloutState {
    fn default() -> Self {
        Self {
            token_ids: Vec::new(),
            logprobs: None,
            routed_experts: None,
            iteration_count: 0,
            original_max_tokens: None,
            prompt_len: 0,
            first_iter_prompt_start: 0,
            expected_final_re_bytes: 0,
        }
    }
}

impl PartialRolloutState {
    #[inline]
    pub fn response_token_count(&self) -> usize {
        self.token_ids.len()
    }

    /// Tokens the next continuation may generate, or `None` when uncapped.
    ///
    /// The loopback folds everything produced so far into the prompt, so the
    /// caller's cap has to be reduced by that much. Reusing the original cap
    /// would grant a second near-full budget and the trajectory would run past
    /// the response limit the caller asked for.
    pub fn remaining_max_tokens(&self) -> Option<u32> {
        self.original_max_tokens
            .map(|original| original.saturating_sub(self.response_token_count() as u32))
    }
}

/// The result of draining a gRPC stream until either a `Complete` frame or
/// end-of-stream is reached.
pub(crate) struct DrainedStreamResult {
    /// New token IDs emitted by *this* stream segment (not yet accumulated).
    pub new_token_ids: Vec<u32>,
    /// New per-token log-probabilities for this segment (if available).
    pub new_logprobs: Option<Vec<f32>>,
    /// Routed-experts segment from this iteration's `Complete` frame.
    pub new_routed_experts: Option<ProtoRoutedExperts>,
    /// The finish reason from the `Complete` frame.
    ///
    /// `"abort"` → PS weight-sync interrupted generation (loopback needed).
    /// `"stop"` / `"length"` → generation finished normally.
    pub finish_reason: String,
    /// The raw `Complete` proto frame, forwarded to response-processing stages.
    pub complete: ProtoGenerateComplete,
}

/// Consume `stream` frame-by-frame until a `Complete` frame is received and
/// return the accumulated result.
pub(crate) async fn drain_stream_for_partial_rollout(
    stream: &mut ProtoStream,
) -> Result<DrainedStreamResult, String> {
    let mut new_token_ids: Vec<u32> = Vec::new();
    let mut new_logprobs: Option<Vec<f32>> = None;

    loop {
        let Some(response_result) = stream.next().await else {
            return Err("stream ended without Complete frame".to_owned());
        };

        let response = match response_result {
            Ok(r) => r,
            Err(status) => return Err(format!("gRPC stream error: {status}")),
        };

        match response.into_response() {
            ProtoResponseVariant::Chunk(chunk) => {
                new_token_ids.extend_from_slice(chunk.token_ids());

                if let Some(lp) = chunk.output_logprobs() {
                    new_logprobs
                        .get_or_insert_with(Vec::new)
                        .extend_from_slice(&lp.token_logprobs);
                }
            }
            ProtoResponseVariant::Complete(complete) => {
                if new_token_ids.is_empty() {
                    new_token_ids = complete.output_ids().to_vec();
                }
                let finish_reason = complete.finish_reason().to_owned();
                let new_routed_experts = complete
                    .routed_experts()
                    .map_err(|error| error.to_string())?;
                return Ok(DrainedStreamResult {
                    new_token_ids,
                    new_logprobs,
                    new_routed_experts,
                    finish_reason,
                    complete,
                });
            }
            ProtoResponseVariant::None => {
                warn!("partial rollout drain: received empty proto frame, skipping");
            }
        }
    }
}

/// Append the tokens from `drained` into `state`.
pub(crate) fn merge_into_partial_state(
    state: &mut PartialRolloutState,
    drained: &DrainedStreamResult,
) -> Result<(), RoutedExpertsError> {
    let prior_completion = state.token_ids.len();
    state.token_ids.extend_from_slice(&drained.new_token_ids);
    state.iteration_count += 1;

    match (&mut state.logprobs, &drained.new_logprobs) {
        (Some(acc), Some(new)) => acc.extend_from_slice(new),
        (None, Some(_new)) if state.iteration_count == 1 => {
            // First iteration produced log-probs; initialise the field.
            state.logprobs.clone_from(&drained.new_logprobs);
        }
        (Some(_), None) => {
            // Previous iterations had log-probs but this one didn't — drop them
            // to keep the vecs in sync.
            warn!(
                "partial rollout merge: log-prob mismatch (had logprobs, new segment missing); \
                 dropping logprobs"
            );
            state.logprobs = None;
        }
        _ => {}
    }

    let seg = drained.new_routed_experts.as_ref();
    let no_new_tokens = drained.new_token_ids.is_empty();

    match (state.routed_experts.as_mut(), seg) {
        // Aborted before any forward; recoverable.
        (_, None) if no_new_tokens => {
            metrics::counter!(
                "smg_routed_experts_recoverable_total",
                "reason" => "early_abort",
            )
            .increment(1);
            Ok(())
        }
        // Append into existing accumulator.
        (Some(acc), Some(seg)) => {
            if !acc.shape_compatible(seg) {
                metrics::counter!(
                    "smg_routed_experts_failed_total",
                    "reason" => "shape_mismatch",
                )
                .increment(1);
                return Err(RoutedExpertsError::ShapeMismatch {
                    accumulator: acc.shape(),
                    segment: seg.shape(),
                });
            }
            acc.data.extend_from_slice(&seg.data);
            Ok(())
        }
        // First time we see an RE segment.
        (None, Some(seg)) => {
            if prior_completion > 0 {
                metrics::counter!(
                    "smg_routed_experts_failed_total",
                    "reason" => "late_arrival",
                )
                .increment(1);
                return Err(RoutedExpertsError::LateArrival {
                    prior_completion_tokens: prior_completion,
                    current_segment_tokens: seg.num_tokens(),
                });
            }
            state.routed_experts = Some(RoutedExpertsAccumulator::from_segment(
                seg,
                state.expected_final_re_bytes,
            ));
            Ok(())
        }
        // Accumulator exists but this iteration produced new tokens without
        // RE — bug path.
        (Some(acc), None) => {
            metrics::counter!(
                "smg_routed_experts_failed_total",
                "reason" => "missing_segment",
            )
            .increment(1);
            Err(RoutedExpertsError::MissingSegment {
                accumulator_tokens_so_far: acc.num_tokens(),
                tokens_in_segment: drained.new_token_ids.len(),
            })
        }
        // No accumulator, no segment, new tokens emitted: engine simply
        // isn't capturing RE for this request — no error.
        (None, None) => Ok(()),
    }
}

/// Reset `ctx` so it is ready for the next loopback iteration.
///
/// Clears the per-iteration stage outputs that must be re-computed
/// (worker selection, client, execution plan, dispatch, load-guards, execution
/// result) and restores `preparation` from the snapshot stashed by
/// `dispatch_entry_with_partial_rollout` before the loop began —
/// `request_building` consumes the live `preparation` via `.take()`, so
/// without this restore the next iteration's `worker_selection` would
/// observe `None` and abort the pipeline with `preparation_stage_not_completed`.
///
/// The restored prompt is then **extended with the tokens accumulated so far**,
/// so the next iteration continues the sequence instead of generating it again
/// from the original prompt. The snapshot is taken before any generation, so
/// restoring it alone would discard every token produced before the abort: the
/// engine would re-decode the whole response under the new weights and the
/// earlier work would be thrown away. `state.token_ids` already holds the
/// accumulated output (appended by `merge_into_partial_state`), which is
/// exactly the continuation prefix the next iteration needs.
pub(crate) fn reset_ctx_for_loopback(ctx: &mut RequestContext, state: &PartialRolloutState) {
    ctx.state.workers = None;
    ctx.state.clients = None;
    ctx.state.execution_plan = None;
    ctx.state.dispatch = None;
    ctx.state.load_guards = None;
    // Restore preparation from the snapshot for the next iteration's
    // worker_selection + request_building.
    ctx.state.preparation = ctx.state.preparation_snapshot.clone();
    // Continue the sequence: prompt + everything generated so far.
    if let Some(preparation) = ctx.state.preparation.as_mut() {
        let prompt_only = preparation.token_ids().len();
        preparation.token_ids_mut().extend_from_slice(&state.token_ids);
        // Emitted on the `route_trace` target so it lands beside the route/pop
        // events a loopback produces, making the continuation verifiable from
        // the logs rather than only by inference.
        info!(
            target: "route_trace",
            event = "loopback_prompt",
            prompt_only_tokens = prompt_only,
            accumulated_tokens = state.token_ids.len(),
            continued_prompt_tokens = preparation.token_ids().len(),
            iteration_count = state.iteration_count,
            "loopback prompt extended with accumulated output"
        );
    }
    // Clear only per-iteration response fields; keep preparation-stage
    // products (`stop_decoder`, `skip_special_tokens`) intact so the final
    // post-execution stage can use them after the last loopback iteration.
    ctx.state.response.execution_result = None;
    ctx.state.response.final_response = None;
    ctx.state.response.responses_iteration_result = None;
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── PartialRolloutState ────────────────────────────────────────────

    #[test]
    fn response_token_count_empty() {
        let state = PartialRolloutState::default();
        assert_eq!(state.response_token_count(), 0);
    }

    #[test]
    fn response_token_count_with_tokens() {
        let state = PartialRolloutState {
            token_ids: vec![1, 2, 3],
            ..PartialRolloutState::default()
        };
        assert_eq!(state.response_token_count(), 3);
    }

    // ─── remaining_max_tokens ───────────────────────────────────────────

    #[test]
    fn remaining_max_tokens_is_none_when_uncapped() {
        let state = PartialRolloutState {
            token_ids: vec![1, 2, 3],
            ..PartialRolloutState::default()
        };
        assert_eq!(state.remaining_max_tokens(), None);
    }

    #[test]
    fn remaining_max_tokens_subtracts_what_was_produced() {
        let state = PartialRolloutState {
            token_ids: vec![0; 6009],
            original_max_tokens: Some(11110),
            ..PartialRolloutState::default()
        };
        assert_eq!(state.remaining_max_tokens(), Some(11110 - 6009));
    }

    #[test]
    fn remaining_max_tokens_saturates_at_zero() {
        // A segment may overshoot the cap slightly; the next leg gets nothing
        // rather than wrapping around to a huge budget.
        let state = PartialRolloutState {
            token_ids: vec![0; 12000],
            original_max_tokens: Some(11110),
            ..PartialRolloutState::default()
        };
        assert_eq!(state.remaining_max_tokens(), Some(0));
    }

    #[test]
    fn remaining_max_tokens_never_exceeds_the_original_cap() {
        // Summed across iterations, generation stays inside one budget.
        let cap = 11110u32;
        let mut produced = 0u32;
        for _ in 0..8 {
            let state = PartialRolloutState {
                token_ids: vec![0; produced as usize],
                original_max_tokens: Some(cap),
                ..PartialRolloutState::default()
            };
            let granted = state.remaining_max_tokens().expect("capped");
            if granted == 0 {
                break;
            }
            // Take half the allowance so several iterations are exercised.
            produced += (granted / 2).max(1);
            assert!(
                produced <= cap,
                "produced {produced} exceeded the cap {cap}"
            );
        }
        assert!(produced <= cap);
    }

    // ─── merge_into_partial_state ────────────────────────────────────────────

    fn make_drained(tokens: &[u32], logprobs: Option<Vec<f32>>) -> DrainedStreamResult {
        make_drained_full(tokens, logprobs, None, "stop")
    }

    fn make_drained_full(
        tokens: &[u32],
        logprobs: Option<Vec<f32>>,
        routed_experts: Option<ProtoRoutedExperts>,
        finish_reason: &str,
    ) -> DrainedStreamResult {
        use smg_grpc_client::sglang_proto::GenerateComplete as SglangGenerateComplete;

        // We need a concrete ProtoGenerateComplete for DrainedStreamResult.
        // Use the Sglang variant — `merge_into_partial_state` doesn't read
        // `complete` at all, so the variant choice is purely cosmetic.
        let complete = ProtoGenerateComplete::Sglang(SglangGenerateComplete {
            finish_reason: finish_reason.to_owned(),
            ..Default::default()
        });
        DrainedStreamResult {
            new_token_ids: tokens.to_vec(),
            new_logprobs: logprobs,
            new_routed_experts: routed_experts,
            finish_reason: finish_reason.to_owned(),
            complete,
        }
    }

    /// Build a synthetic RE segment whose row stride matches the given shape
    /// and whose token count equals `num_tokens`.  Bytes are filled with `fill`
    /// so equality checks remain meaningful across appends.
    fn make_re(num_tokens: usize, num_layers: u32, top_k: u32, fill: u8) -> ProtoRoutedExperts {
        let token_bytes = num_layers as usize * top_k as usize;
        let data = vec![fill; num_tokens * token_bytes];
        ProtoRoutedExperts {
            data: bytes::Bytes::from(data),
            num_layers,
            top_k,
            dtype: RoutedExpertsDtype::U8,
            index: 0,
        }
    }

    #[test]
    fn merge_tokens_no_logprobs() {
        let mut state = PartialRolloutState::default();
        let drained = make_drained(&[10, 20, 30], None);
        merge_into_partial_state(&mut state, &drained).unwrap();
        assert_eq!(state.token_ids, vec![10, 20, 30]);
        assert!(state.logprobs.is_none());
        assert!(state.routed_experts.is_none());
    }

    #[test]
    fn merge_tokens_with_logprobs_two_iterations() {
        let mut state = PartialRolloutState::default();

        // Iteration 1 — with logprobs
        let d1 = make_drained(&[1, 2], Some(vec![0.1, 0.2]));
        merge_into_partial_state(&mut state, &d1).unwrap();
        assert_eq!(state.token_ids, vec![1, 2]);
        assert_eq!(state.logprobs, Some(vec![0.1, 0.2]));

        // Iteration 2 — append
        let d2 = make_drained(&[3, 4], Some(vec![0.3, 0.4]));
        merge_into_partial_state(&mut state, &d2).unwrap();
        assert_eq!(state.token_ids, vec![1, 2, 3, 4]);
        assert_eq!(state.logprobs, Some(vec![0.1, 0.2, 0.3, 0.4]));
    }

    #[test]
    fn merge_logprob_mismatch_drops_logprobs() {
        let mut state = PartialRolloutState {
            token_ids: vec![1],
            logprobs: Some(vec![0.5]),
            iteration_count: 1,
            ..PartialRolloutState::default()
        };
        // Second segment has no logprobs → drop the accumulated ones
        let drained = make_drained(&[2], None);
        merge_into_partial_state(&mut state, &drained).unwrap();
        assert_eq!(state.token_ids, vec![1, 2]);
        assert!(state.logprobs.is_none());
    }

    /// `merge_into_partial_state` increments `iteration_count` on every call.
    #[test]
    fn merge_increments_iteration_count() {
        let mut state = PartialRolloutState::default();
        assert_eq!(state.iteration_count, 0);

        let d1 = make_drained(&[1], None);
        merge_into_partial_state(&mut state, &d1).unwrap();
        assert_eq!(state.iteration_count, 1);

        let d2 = make_drained(&[2], None);
        merge_into_partial_state(&mut state, &d2).unwrap();
        assert_eq!(state.iteration_count, 2);

        // Token accumulation is also correct
        assert_eq!(state.token_ids, vec![1, 2]);
    }

    /// `response_token_count` reflects tokens accumulated across iterations.
    #[test]
    fn response_token_count_accumulates_across_merges() {
        let mut state = PartialRolloutState::default();
        assert_eq!(state.response_token_count(), 0);

        merge_into_partial_state(&mut state, &make_drained(&[10, 20], None)).unwrap();
        assert_eq!(state.response_token_count(), 2);

        merge_into_partial_state(&mut state, &make_drained(&[30, 40, 50], None)).unwrap();
        assert_eq!(state.response_token_count(), 5);
    }

    /// Merging when `state` is empty and new segment has log-probs initialises the field.
    #[test]
    fn merge_initialises_logprobs_from_first_segment() {
        let mut state = PartialRolloutState::default();
        let d = make_drained(&[1, 2], Some(vec![0.1, 0.2]));
        merge_into_partial_state(&mut state, &d).unwrap();
        assert_eq!(state.logprobs, Some(vec![0.1, 0.2]));
    }

    /// When both sides have no log-probs, the field stays `None`.
    #[test]
    fn merge_no_logprobs_stays_none() {
        let mut state = PartialRolloutState::default();
        merge_into_partial_state(&mut state, &make_drained(&[1], None)).unwrap();
        merge_into_partial_state(&mut state, &make_drained(&[2], None)).unwrap();
        assert!(state.logprobs.is_none());
    }

    // ─── Routed experts merge behaviour ──────────────────────────────────

    /// First iteration seeds the accumulator from a non-empty segment.
    #[test]
    fn re_seeds_accumulator_from_first_segment() {
        let mut state = PartialRolloutState {
            expected_final_re_bytes: 1024,
            ..PartialRolloutState::default()
        };
        let seg = make_re(3, 2, 4, 0xAA); // 3 tokens × 8 bytes
        let d = make_drained_full(&[1, 2, 3], None, Some(seg.clone()), "abort");
        merge_into_partial_state(&mut state, &d).unwrap();

        let acc = state.routed_experts.as_ref().expect("accumulator seeded");
        assert_eq!(acc.num_tokens(), 3);
        assert_eq!(acc.num_layers, 2);
        assert_eq!(acc.top_k, 4);
        assert_eq!(acc.dtype, RoutedExpertsDtype::U8);
        assert!(acc.data.iter().all(|&b| b == 0xAA));
    }

    /// Two compatible segments concatenate.
    #[test]
    fn re_appends_compatible_segments() {
        let mut state = PartialRolloutState::default();
        let s1 = make_re(2, 2, 4, 0x11);
        let s2 = make_re(3, 2, 4, 0x22);

        merge_into_partial_state(
            &mut state,
            &make_drained_full(&[1, 2], None, Some(s1), "abort"),
        )
        .unwrap();
        merge_into_partial_state(
            &mut state,
            &make_drained_full(&[3, 4, 5], None, Some(s2), "stop"),
        )
        .unwrap();

        let acc = state.routed_experts.as_ref().expect("accumulator");
        assert_eq!(acc.num_tokens(), 5);
        // First 16 bytes are 0x11, next 24 bytes are 0x22.
        let row = acc.token_bytes();
        assert!(acc.data[..2 * row].iter().all(|&b| b == 0x11));
        assert!(acc.data[2 * row..].iter().all(|&b| b == 0x22));
    }

    /// A shape change between two iterations is fail-hard.
    #[test]
    fn re_shape_mismatch_returns_err() {
        let mut state = PartialRolloutState::default();
        let s1 = make_re(2, 2, 4, 0x01);
        let s2 = make_re(2, 2, 8, 0x02); // top_k changed

        merge_into_partial_state(
            &mut state,
            &make_drained_full(&[1, 2], None, Some(s1), "abort"),
        )
        .unwrap();
        let err = merge_into_partial_state(
            &mut state,
            &make_drained_full(&[3, 4], None, Some(s2), "abort"),
        )
        .unwrap_err();
        assert!(matches!(err, RoutedExpertsError::ShapeMismatch { .. }));
    }

    /// §1.A step 5: abort with no tokens and no segment is recoverable.
    #[test]
    fn re_early_abort_no_tokens_no_segment_is_ok() {
        let mut state = PartialRolloutState::default();
        let d = make_drained_full(&[], None, None, "abort");
        merge_into_partial_state(&mut state, &d).unwrap();
        assert!(state.routed_experts.is_none());
        assert_eq!(state.token_ids, Vec::<u32>::new());
    }

    /// Late arrival: tokens accumulated in earlier iter without RE; a later
    /// segment cannot retroactively cover them — fail-hard.
    #[test]
    fn re_late_arrival_returns_err() {
        let mut state = PartialRolloutState::default();
        // Iter 1: tokens but no RE (engine not yet capturing).  This is
        // legal at face value — the (None, None, tokens > 0) arm returns
        // Ok because the engine simply isn't producing RE.
        merge_into_partial_state(&mut state, &make_drained_full(&[1, 2], None, None, "abort"))
            .unwrap();
        // Iter 2: a segment arrives — but prior tokens are already in
        // state.token_ids → LateArrival, because the prompt segment
        // covering iter 1 is permanently lost.
        let s = make_re(2, 1, 1, 0xFF);
        let err = merge_into_partial_state(
            &mut state,
            &make_drained_full(&[3, 4], None, Some(s), "stop"),
        )
        .unwrap_err();
        assert!(matches!(err, RoutedExpertsError::LateArrival { .. }));
    }

    /// Missing segment: accumulator exists, new iter has tokens but no RE.
    #[test]
    fn re_missing_segment_returns_err() {
        let mut state = PartialRolloutState::default();
        let s = make_re(2, 1, 1, 0x10);
        merge_into_partial_state(
            &mut state,
            &make_drained_full(&[1, 2], None, Some(s), "abort"),
        )
        .unwrap();
        let err =
            merge_into_partial_state(&mut state, &make_drained_full(&[3, 4], None, None, "stop"))
                .unwrap_err();
        assert!(matches!(err, RoutedExpertsError::MissingSegment { .. }));
    }

    /// When the engine isn't capturing RE (every iteration returns `None`),
    /// the accumulator stays empty and merge succeeds — no opt-in flag is
    /// required because RE capture is decided engine-side.
    #[test]
    fn re_no_segments_no_accumulator_is_ok() {
        let mut state = PartialRolloutState::default();
        merge_into_partial_state(&mut state, &make_drained_full(&[1, 2], None, None, "stop"))
            .unwrap();
        assert!(state.routed_experts.is_none());
    }

    // ─── reset_ctx_for_loopback ──────────────────────────────────────────────
    #[test]
    fn reset_preserves_preparation_products_in_response_state() {
        use std::sync::Arc;

        use llm_tokenizer::{stop::StopSequenceDecoderBuilder, MockTokenizer};
        use openai_protocol::chat::ChatCompletionRequest;
        use reasoning_parser::ParserFactory as ReasoningParserFactory;
        use tool_parser::ParserFactory as ToolParserFactory;

        use crate::routers::grpc::context::{
            FinalResponse, PreparationOutput, ProcessingState, RequestContext, RequestInput,
            RequestType, SharedComponents,
        };

        // Minimal RequestContext built directly so the test stays inside the
        // crate (no dependency on the integration test helpers).  We use a
        // Chat request because `ChatCompletionRequest` derives `Default`
        // whereas `GenerateRequest` does not — the field-clearing logic under
        // test does not depend on the request variant.
        let components = Arc::new(SharedComponents {
            tokenizer_registry: Arc::new(llm_tokenizer::registry::TokenizerRegistry::new()),
            tool_parser_factory: ToolParserFactory::default(),
            reasoning_parser_factory: ReasoningParserFactory::default(),
            configured_tool_parser: None,
            multimodal: None,
        });
        let mut ctx = RequestContext {
            input: RequestInput {
                request_type: RequestType::Chat(Arc::new(ChatCompletionRequest::default())),
                headers: None,
                model_id: "test-model".to_string(),
                tenant_request_meta: None,
            },
            components,
            state: ProcessingState::default(),
        };

        // Simulate Preparation stage outputs that live in `ResponseState`.
        let tokenizer: Arc<dyn llm_tokenizer::traits::Tokenizer> = Arc::new(MockTokenizer::new());
        let stop_decoder = StopSequenceDecoderBuilder::new(tokenizer)
            .skip_special_tokens(true)
            .build();
        ctx.state.preparation = None;
        ctx.state.preparation_snapshot = Some(PreparationOutput::Completion {
            original_text: "prompt".to_string(),
            token_ids: vec![1, 2, 3],
        });
        ctx.state.response.stop_decoder = Some(stop_decoder);
        ctx.state.response.skip_special_tokens = Some(true);

        // Simulate a per-iteration response field being populated; reset must
        // clear it so the next iteration starts clean.
        ctx.state.response.final_response = Some(FinalResponse::Generate(vec![]));

        // No tokens generated yet, so the restored prompt is the prompt alone.
        reset_ctx_for_loopback(&mut ctx, &PartialRolloutState::default());

        // Preparation is restored from the snapshot for the next loopback
        // iteration's worker_selection + request_building.
        let restored_preparation = ctx
            .state
            .preparation
            .as_ref()
            .expect("preparation must survive loopback reset");
        let PreparationOutput::Completion {
            original_text,
            token_ids,
        } = restored_preparation
        else {
            panic!("expected Completion preparation output");
        };
        assert_eq!(original_text, "prompt");
        assert_eq!(token_ids, &[1, 2, 3]);

        // Preparation-stage products preserved.
        assert!(
            ctx.state.response.stop_decoder.is_some(),
            "stop_decoder must survive loopback reset (set once during preparation)"
        );
        assert_eq!(
            ctx.state.response.skip_special_tokens,
            Some(true),
            "skip_special_tokens must survive loopback reset"
        );

        // Per-iteration response fields cleared.
        assert!(
            ctx.state.response.execution_result.is_none(),
            "execution_result must be cleared between iterations"
        );
        assert!(
            ctx.state.response.final_response.is_none(),
            "final_response must be cleared between iterations"
        );
        assert!(
            ctx.state.response.responses_iteration_result.is_none(),
            "responses_iteration_result must be cleared between iterations"
        );
    }

    #[test]
    fn reset_appends_accumulated_tokens_to_the_prompt() {
        use std::sync::Arc;

        use openai_protocol::chat::ChatCompletionRequest;
        use reasoning_parser::ParserFactory as ReasoningParserFactory;
        use tool_parser::ParserFactory as ToolParserFactory;

        use crate::routers::grpc::context::{
            PreparationOutput, ProcessingState, RequestContext, RequestInput, RequestType,
            SharedComponents,
        };

        let components = Arc::new(SharedComponents {
            tokenizer_registry: Arc::new(llm_tokenizer::registry::TokenizerRegistry::new()),
            tool_parser_factory: ToolParserFactory::default(),
            reasoning_parser_factory: ReasoningParserFactory::default(),
            configured_tool_parser: None,
            multimodal: None,
        });
        let mut ctx = RequestContext {
            input: RequestInput {
                request_type: RequestType::Chat(Arc::new(ChatCompletionRequest::default())),
                headers: None,
                model_id: "test-model".to_string(),
                tenant_request_meta: None,
            },
            components,
            state: ProcessingState::default(),
        };
        ctx.state.preparation_snapshot = Some(PreparationOutput::Completion {
            original_text: "prompt".to_string(),
            token_ids: vec![1, 2, 3],
        });

        // Two tokens were generated before the abort.
        let mut state = PartialRolloutState::default();
        state.token_ids = vec![7, 8];

        reset_ctx_for_loopback(&mut ctx, &state);

        let PreparationOutput::Completion { token_ids, .. } = ctx
            .state
            .preparation
            .as_ref()
            .expect("preparation must survive loopback reset")
        else {
            panic!("expected Completion preparation output");
        };
        // The next iteration continues the sequence instead of regenerating it.
        // Restoring the snapshot alone would leave [1, 2, 3] and throw away the
        // tokens produced before the abort.
        assert_eq!(
            token_ids,
            &[1, 2, 3, 7, 8],
            "loopback prompt must be the original prompt plus the accumulated output"
        );
    }
}
