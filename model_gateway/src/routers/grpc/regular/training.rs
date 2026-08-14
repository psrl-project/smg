//! Protocol-independent training helpers.
//!
//! Native endpoints keep their public wire format, while TITO consumes one
//! canonical, completed Chat turn with sampled-token log probabilities.

use axum::http::HeaderMap;
use openai_protocol::chat::ChatCompletionRequest;
use smg_tito::TITO_SESSION_HEADER;

pub(crate) fn is_tito_request(headers: Option<&HeaderMap>) -> bool {
    headers.is_some_and(|headers| headers.contains_key(TITO_SESSION_HEADER))
}

pub(crate) fn configure_canonical_turn(request: &mut ChatCompletionRequest) {
    request.stream = false;
    request.logprobs = true;
    request.top_logprobs = Some(request.top_logprobs.unwrap_or(1).max(1));
}
