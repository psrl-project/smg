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
use openai_protocol::sampler_capture::SamplerCaptureWidth;
use smg_tito::validate_sampler_row;

use crate::routers::grpc::proto_wrapper::ProtoOutputLogProbs;

/// Encode the sampler top-k head into `(ids_b64, log_probs_b64, shape)`.
///
/// Rows are sorted by descending log-probability before encoding, so the finite support is a strict
/// prefix (the contract the trainer's compact transport relies on) and the truncation keeps the true
/// top-`width` entries. Returns `None` when no head was captured, when no width was requested, or
/// when a position has fewer than `width` entries (a partial capture must not be served silently).
pub(crate) fn encode_sampler_topk_for_response(
    lp: &ProtoOutputLogProbs,
    width: SamplerCaptureWidth,
) -> Option<(String, String, Vec<u32>)> {
    let width = usize::from(width);
    if lp.top_logprobs.is_empty() || width == 0 {
        return None;
    }
    let rows = lp.top_logprobs.len();
    if lp
        .top_logprobs
        .iter()
        .any(|row| row.token_ids.len() != row.values.len() || row.token_ids.len() < width)
    {
        return None;
    }
    let byte_capacity = rows.checked_mul(width)?.checked_mul(size_of::<u32>())?;
    let mut id_bytes = Vec::new();
    let mut log_prob_bytes = Vec::new();
    id_bytes.try_reserve_exact(byte_capacity).ok()?;
    log_prob_bytes.try_reserve_exact(byte_capacity).ok()?;
    // Reused per-row buffers so the byte copy is one vectorized cast per row, not per element.
    let mut row_ids: Vec<i32> = Vec::new();
    let mut row_log_probs: Vec<f32> = Vec::new();
    row_ids.try_reserve_exact(width).ok()?;
    row_log_probs.try_reserve_exact(width).ok()?;
    for row in &lp.top_logprobs {
        let mut entries = Vec::new();
        entries.try_reserve_exact(row.token_ids.len()).ok()?;
        entries.extend(
            row.token_ids
                .iter()
                .copied()
                .zip(row.values.iter().copied()),
        );
        entries.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        row_ids.clear();
        row_log_probs.clear();
        for (token_id, log_prob) in entries.iter().take(width) {
            row_ids.push(i32::try_from(*token_id).ok()?);
            row_log_probs.push(*log_prob);
        }
        validate_sampler_row(bytemuck::cast_slice::<i32, u32>(&row_ids), &row_log_probs)?;
        // Native-endian cast (SMG targets little-endian hosts), matching the routed-experts payload.
        id_bytes.extend_from_slice(bytemuck::cast_slice(&row_ids));
        log_prob_bytes.extend_from_slice(bytemuck::cast_slice(&row_log_probs));
    }
    Some((
        base64::engine::general_purpose::STANDARD.encode(id_bytes),
        base64::engine::general_purpose::STANDARD.encode(log_prob_bytes),
        vec![u32::try_from(rows).ok()?, u32::try_from(width).ok()?],
    ))
}

#[cfg(test)]
mod tests {
    use openai_protocol::sampler_capture::SamplerCaptureWidth;

    use super::*;
    use crate::routers::grpc::proto_wrapper::ProtoTopLogProbs;

    fn output(ids: Vec<u32>, values: Vec<f32>) -> ProtoOutputLogProbs {
        ProtoOutputLogProbs {
            token_logprobs: vec![0.0],
            token_ids: vec![ids[0]],
            top_logprobs: vec![ProtoTopLogProbs {
                values,
                token_ids: ids,
            }],
        }
    }

    fn width(value: u32) -> SamplerCaptureWidth {
        value.try_into().unwrap()
    }

    #[test]
    fn rejects_invalid_sampler_rows_before_encoding() {
        assert!(encode_sampler_topk_for_response(
            &output(vec![1, 2], vec![-0.1, f32::NAN]),
            width(2)
        )
        .is_none());
        assert!(encode_sampler_topk_for_response(
            &output(vec![1, 2], vec![-0.1, f32::INFINITY]),
            width(2)
        )
        .is_none());
        assert!(
            encode_sampler_topk_for_response(&output(vec![1, 1], vec![-0.1, -0.2]), width(2))
                .is_none()
        );
        assert!(encode_sampler_topk_for_response(
            &output(vec![1, i32::MAX as u32 + 1], vec![-0.1, -0.2]),
            width(2),
        )
        .is_none());
    }

    #[test]
    fn accepts_negative_infinity_as_the_support_sentinel() {
        assert!(encode_sampler_topk_for_response(
            &output(vec![1, 2, 3], vec![-0.1, -0.2, f32::NEG_INFINITY]),
            width(3),
        )
        .is_some());
    }
}
