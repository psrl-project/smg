//! Encoding helpers for the sampler top-k head returned on the /generate boundary.
//!
//! The /generate response carries the sampler head as base64 little-endian `int32` token ids and
//! `float32` log probabilities plus a `[rows, width]` shape, mirroring the routed-experts payload.
//! Rows are the generated token positions, each truncated to `width` entries in rank order.
//!
//! `width` is the per-token capture width the caller requested via `sampler_top_logprobs`. The
//! request value and the wire width agree only because the servicer's `_build_top_logprobs` filters
//! the engine's entries to `rank <= width` (the engine returns one extra column for the sampled
//! token). The trainer's `capture_width` is the authoritative definition of that width; keep the
//! request, the filter and the returned shape in lockstep or an off-by-one misaligns the head.

use base64::Engine as _;

use crate::routers::grpc::proto_wrapper::ProtoOutputLogProbs;

/// Encode the sampler top-k head into `(ids_b64, log_probs_b64, shape)`.
///
/// Rows are sorted by descending log-probability before encoding, so the finite support is a strict
/// prefix (the contract the trainer's compact transport relies on) and the truncation keeps the true
/// top-`width` entries. Returns `None` when no head was captured, when no width was requested, or
/// when a position has fewer than `width` entries (a partial capture must not be served silently).
pub(crate) fn encode_sampler_topk_for_response(
    lp: &ProtoOutputLogProbs,
    width: usize,
) -> Option<(String, String, Vec<u32>)> {
    if lp.top_logprobs.is_empty() || width == 0 {
        return None;
    }
    let rows = lp.top_logprobs.len();
    let mut id_bytes = Vec::with_capacity(rows * width * 4);
    let mut log_prob_bytes = Vec::with_capacity(rows * width * 4);
    // Reused per-row buffers so the byte copy is one vectorized cast per row, not per element.
    let mut row_ids: Vec<u32> = Vec::with_capacity(width);
    let mut row_log_probs: Vec<f32> = Vec::with_capacity(width);
    for row in &lp.top_logprobs {
        if row.token_ids.len() != row.values.len() || row.token_ids.len() < width {
            return None;
        }
        let mut entries: Vec<(u32, f32)> = row
            .token_ids
            .iter()
            .copied()
            .zip(row.values.iter().copied())
            .collect();
        entries.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        row_ids.clear();
        row_log_probs.clear();
        for (token_id, log_prob) in entries.iter().take(width) {
            row_ids.push(*token_id);
            row_log_probs.push(*log_prob);
        }
        // Native-endian cast (SMG targets little-endian hosts), matching the routed-experts payload.
        id_bytes.extend_from_slice(bytemuck::cast_slice(&row_ids));
        log_prob_bytes.extend_from_slice(bytemuck::cast_slice(&row_log_probs));
    }
    Some((
        base64::engine::general_purpose::STANDARD.encode(id_bytes),
        base64::engine::general_purpose::STANDARD.encode(log_prob_bytes),
        vec![rows as u32, width as u32],
    ))
}
