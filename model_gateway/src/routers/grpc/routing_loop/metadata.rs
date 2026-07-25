//! Request routing metadata parsing.

use axum::http::HeaderMap;
use serde_json::Value;

use crate::routers::grpc::context::RequestContext;

/// Routing metadata supplied by callers that use the routing loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RoutingMeta {
    pub request_id: i64,
    pub prompt_id: i64,
    pub version_tag: i64,
    pub is_validate: bool,
    pub is_sticky: bool,
    /// One-shot pin: when set, this request is pinned to `rollout_instance_hint`
    /// for its FIRST worker selection only, independent of `is_sticky`. Unlike
    /// `is_sticky` (which pins for the whole trajectory), this is cleared on the
    /// first partial-rollout / preemption loopback so re-dispatch falls back to
    /// free routing. Set by the session hang/continue scheduler when it readmits
    /// a hung session onto a chosen instance.
    pub force_pin_once: bool,
    pub rollout_instance_hint: Option<(String, usize)>,
    /// Number of tokens already generated in previous partial-rollout iterations.
    pub response_token_count: Option<usize>,
}

/// Parse routing metadata from transport headers and a JSON body.
///
/// Headers take precedence over body fields.  Returns `None` when no routing
/// metadata is present so ordinary requests can skip routing-loop bookkeeping.
pub(crate) fn parse_routing_request_meta(
    headers: Option<&HeaderMap>,
    body: Option<&Value>,
) -> Option<RoutingMeta> {
    let request_id = parse_i64_header(headers, "x-request-id")
        .or_else(|| parse_i64_from_body(body, "request_id"))?;
    let prompt_id = parse_i64_header(headers, "x-prompt-id")
        .or_else(|| parse_i64_from_body(body, "prompt_id"))?;
    let version_tag = parse_i64_header(headers, "x-version-tag")
        .or_else(|| parse_i64_from_body(body, "version_tag"))
        .unwrap_or(-1);
    let is_validate = parse_bool_header(headers, "x-is-validate")
        .or_else(|| parse_bool_from_body(body, "is_validate"))
        .unwrap_or(false);
    let is_sticky = parse_bool_header(headers, "x-is-sticky")
        .or_else(|| parse_bool_from_body(body, "is_sticky"))
        .unwrap_or(false);
    let force_pin_once = parse_bool_header(headers, "x-force-pin-once")
        .or_else(|| parse_bool_from_body(body, "force_pin_once"))
        .unwrap_or(false);
    let rollout_instance_hint = parse_rollout_instance_hint_from_headers(headers)
        .or_else(|| parse_rollout_instance_hint_from_body(body));
    let response_token_count = parse_usize_header(headers, "x-response-token-count");

    Some(RoutingMeta {
        request_id,
        prompt_id,
        version_tag,
        is_validate,
        is_sticky,
        force_pin_once,
        rollout_instance_hint,
        response_token_count,
    })
}

pub(crate) fn parse_routing_request_meta_from_context(ctx: &RequestContext) -> Option<RoutingMeta> {
    parse_routing_request_meta(ctx.input.headers.as_ref(), None)
}

fn parse_i64_header(headers: Option<&HeaderMap>, key: &'static str) -> Option<i64> {
    headers
        .and_then(|h| h.get(key))
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<i64>().ok())
}

fn parse_bool_header(headers: Option<&HeaderMap>, key: &'static str) -> Option<bool> {
    headers
        .and_then(|h| h.get(key))
        .and_then(|v| v.to_str().ok())
        .and_then(parse_bool_like)
}

fn parse_usize_header(headers: Option<&HeaderMap>, key: &'static str) -> Option<usize> {
    headers
        .and_then(|h| h.get(key))
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<usize>().ok())
}

pub(crate) fn parse_i64_from_body(body: Option<&Value>, key: &str) -> Option<i64> {
    let value = body?.get(key)?;
    value
        .as_i64()
        .or_else(|| value.as_u64().and_then(|v| i64::try_from(v).ok()))
        .or_else(|| value.as_str().and_then(|s| s.trim().parse::<i64>().ok()))
}

fn parse_usize_from_json(value: &Value) -> Option<usize> {
    value
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .or_else(|| value.as_i64().and_then(|v| usize::try_from(v).ok()))
        .or_else(|| value.as_str().and_then(|s| s.trim().parse::<usize>().ok()))
}

fn parse_bool_from_body(body: Option<&Value>, key: &str) -> Option<bool> {
    let value = body?.get(key)?;
    value
        .as_bool()
        .or_else(|| value.as_str().and_then(parse_bool_like))
}

fn parse_bool_like(s: &str) -> Option<bool> {
    let s = s.trim();
    if s == "1"
        || s.eq_ignore_ascii_case("true")
        || s.eq_ignore_ascii_case("t")
        || s.eq_ignore_ascii_case("yes")
        || s.eq_ignore_ascii_case("y")
    {
        Some(true)
    } else if s == "0"
        || s.eq_ignore_ascii_case("false")
        || s.eq_ignore_ascii_case("f")
        || s.eq_ignore_ascii_case("no")
        || s.eq_ignore_ascii_case("n")
    {
        Some(false)
    } else {
        None
    }
}

fn parse_rollout_instance_hint_from_headers(
    headers: Option<&HeaderMap>,
) -> Option<(String, usize)> {
    let headers = headers?;
    let base_worker_id = headers
        .get("x-base-worker-id")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())?
        .to_string();
    let target_dp_rank = headers
        .get("x-target-dp-rank")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<usize>().ok())?;
    Some((base_worker_id, target_dp_rank))
}

fn parse_rollout_instance_hint_from_body(body: Option<&Value>) -> Option<(String, usize)> {
    let body = body?;

    if let Some(instance) = body
        .get("rollout_instance_id")
        .or_else(|| body.get("rollout_instance_hint"))
    {
        if let Some(arr) = instance.as_array() {
            if arr.len() >= 2 {
                let base_worker_id = arr
                    .first()
                    .and_then(|v| v.as_str())
                    .map(str::trim)
                    .filter(|s| !s.is_empty())?
                    .to_string();
                let target_dp_rank = arr.get(1).and_then(parse_usize_from_json)?;
                return Some((base_worker_id, target_dp_rank));
            }
        }

        if let Some(obj) = instance.as_object() {
            let base_worker_id = obj
                .get("base_worker_id")
                .or_else(|| obj.get("worker_id"))
                .or_else(|| obj.get("replica_id"))
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty())?
                .to_string();
            let target_dp_rank = obj
                .get("target_dp_rank")
                .or_else(|| obj.get("dp_rank"))
                .or_else(|| obj.get("data_parallel_rank"))
                .and_then(parse_usize_from_json)?;
            return Some((base_worker_id, target_dp_rank));
        }
    }

    let base_worker_id = body
        .get("base_worker_id")
        .or_else(|| body.get("x-base-worker-id"))
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())?
        .to_string();
    let target_dp_rank = body
        .get("target_dp_rank")
        .or_else(|| body.get("x-target-dp-rank"))
        .and_then(parse_usize_from_json)?;
    Some((base_worker_id, target_dp_rank))
}

#[cfg(test)]
mod tests {
    use axum::http::{HeaderMap, HeaderValue};
    use serde_json::json;

    use super::*;

    #[test]
    fn parse_returns_none_without_metadata() {
        assert_eq!(
            parse_routing_request_meta(None, Some(&json!({"model": "m"}))),
            None
        );
    }

    #[test]
    fn parse_headers_override_body() {
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", HeaderValue::from_static("42"));
        headers.insert("x-prompt-id", HeaderValue::from_static("7"));
        headers.insert("x-version-tag", HeaderValue::from_static("3"));
        headers.insert("x-is-validate", HeaderValue::from_static("yes"));
        headers.insert("x-is-sticky", HeaderValue::from_static("yes"));
        headers.insert("x-base-worker-id", HeaderValue::from_static("worker-a"));
        headers.insert("x-target-dp-rank", HeaderValue::from_static("2"));

        let meta = parse_routing_request_meta(
            Some(&headers),
            Some(&json!({
                "request_id": 1,
                "prompt_id": 1,
                "version_tag": 1,
                "is_validate": false,
                "is_sticky": false,
                "rollout_instance_id": ["worker-b", 5]
            })),
        );

        assert_eq!(
            meta,
            Some(RoutingMeta {
                request_id: 42,
                prompt_id: 7,
                version_tag: 3,
                is_validate: true,
                is_sticky: true,
                force_pin_once: false,
                rollout_instance_hint: Some(("worker-a".to_string(), 2)),
                response_token_count: None,
            })
        );
    }

    #[test]
    fn parse_body_rollout_instance_object_aliases() {
        // request_id is present but prompt_id is absent → returns None
        // because both are required for RoutingMeta to be constructed.
        let meta = parse_routing_request_meta(
            None,
            Some(&json!({
                "request_id": "5",
                "rollout_instance_hint": {
                    "replica_id": "worker-c",
                    "data_parallel_rank": "4"
                }
            })),
        );

        assert_eq!(meta, None);
    }

    #[test]
    fn parse_body_both_ids_with_rollout_hint() {
        let meta = parse_routing_request_meta(
            None,
            Some(&json!({
                "request_id": "5",
                "prompt_id": "3",
                "rollout_instance_hint": {
                    "replica_id": "worker-c",
                    "data_parallel_rank": "4"
                }
            })),
        );

        assert_eq!(
            meta,
            Some(RoutingMeta {
                request_id: 5,
                prompt_id: 3,
                version_tag: -1,
                is_validate: false,
                is_sticky: false,
                force_pin_once: false,
                rollout_instance_hint: Some(("worker-c".to_string(), 4)),
                response_token_count: None,
            })
        );
    }

    #[test]
    fn parse_force_pin_once_header() {
        // force_pin_once is set independently of is_sticky and pins the request
        // to the hinted instance for its first selection.
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", HeaderValue::from_static("9"));
        headers.insert("x-prompt-id", HeaderValue::from_static("9"));
        headers.insert("x-force-pin-once", HeaderValue::from_static("true"));
        headers.insert("x-base-worker-id", HeaderValue::from_static("worker-d"));
        headers.insert("x-target-dp-rank", HeaderValue::from_static("1"));

        let meta = parse_routing_request_meta(Some(&headers), None).unwrap();
        assert!(meta.force_pin_once);
        assert!(!meta.is_sticky);
        assert_eq!(meta.rollout_instance_hint, Some(("worker-d".to_string(), 1)));
    }

    #[test]
    fn parse_force_pin_once_defaults_false() {
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", HeaderValue::from_static("9"));
        headers.insert("x-prompt-id", HeaderValue::from_static("9"));

        let meta = parse_routing_request_meta(Some(&headers), None).unwrap();
        assert!(!meta.force_pin_once);
    }
}
