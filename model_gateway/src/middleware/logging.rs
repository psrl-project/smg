//! Tracing/logging integration for the HTTP layer.
//!
//! Wires `tower_http::trace::TraceLayer` with custom span/request/response
//! handlers that propagate W3C trace context, attach the request ID into
//! the span, and record HTTP-level metrics via the observability layer.

use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};

use axum::{extract::Request, response::Response};
use tower::{Layer, Service};
use tower_http::trace::{MakeSpan, OnRequest, OnResponse, TraceLayer};
use tracing::{debug, error, field::Empty, info, info_span, warn, Span};
use tracing_opentelemetry::OpenTelemetrySpanExt;

use super::{metrics::matched_path_label, request_id::RequestId};
use crate::observability::{
    metrics::{method_to_static_str, Metrics},
    otel_trace::extract_trace_context_http,
};
use crate::routers::error::{
    extract_error_code_from_response, PROMPT_OVERFLOW_ERROR_CODE,
};

/// The request line a completed response answers, carried on the response so the
/// completion log can name the caller.
///
/// The default (non-JSON) log format prints no span fields, so the `method`/`uri`
/// recorded on the request span never reach the log line, and `OnResponse` sees
/// only the response. [`RequestLineBridge`] therefore captures these on the
/// request and hands them to `on_response` through the response extensions —
/// the one channel both sides share.
#[derive(Clone, Debug)]
pub struct RequestLine {
    /// HTTP method, interned through [`method_to_static_str`].
    pub method: &'static str,
    /// Raw request path as the caller sent it (`uri.path()`).
    pub uri: String,
    /// Matched axum route template, or `"other"` when nothing matched.
    pub matched: String,
}

/// Layer that copies the request line from a request onto its response.
///
/// Apply it *inside* [`create_logging_layer`]: the response it annotates is the
/// one `ResponseLogger` then reads. A request that never produces a response
/// (client disconnect) simply drops its line with the request, so nothing is
/// retained between requests.
#[derive(Clone, Copy, Debug, Default)]
pub struct RequestLineBridge;

impl<S> Layer<S> for RequestLineBridge {
    type Service = RequestLineBridgeService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RequestLineBridgeService { inner }
    }
}

/// Service produced by [`RequestLineBridge`].
#[derive(Clone, Debug)]
pub struct RequestLineBridgeService<S> {
    inner: S,
}

impl<S, B> Service<Request<B>> for RequestLineBridgeService<S>
where
    S: Service<Request<B>, Response = Response<B>> + Send + Clone + 'static,
    S::Future: Send + 'static,
    B: Send + 'static,
{
    type Response = Response<B>;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<B>) -> Self::Future {
        // `matched_path_label` borrows the request extensions, so resolve the
        // whole line before the request moves into the inner service.
        let line = RequestLine {
            method: method_to_static_str(req.method().as_str()),
            uri: req.uri().path().to_owned(),
            matched: matched_path_label(req.extensions()).to_owned(),
        };
        let mut inner = self.inner.clone();
        Box::pin(async move {
            let mut response = inner.call(req).await?;
            response.extensions_mut().insert(line);
            Ok(response)
        })
    }
}

/// Build the request-line bridge layer.
pub fn create_request_line_bridge() -> RequestLineBridge {
    RequestLineBridge
}


/// Custom span maker that includes request ID
#[derive(Clone, Debug)]
pub struct RequestSpan;

impl<B> MakeSpan<B> for RequestSpan {
    fn make_span(&mut self, request: &Request<B>) -> Span {
        // Extract incoming W3C trace context (traceparent/tracestate) so that
        // server-side spans become children of the caller's distributed trace.
        let parent_cx = extract_trace_context_http(request.headers());

        // Don't try to extract request ID here - it won't be available yet
        // The RequestIdLayer runs after TraceLayer creates the span
        let span = info_span!(
            target: "smg::otel-trace",
            "http_request",
            method = %request.method(),
            uri = %request.uri(),
            version = ?request.version(),
            request_id = Empty,  // Will be set later
            status_code = Empty,
            latency = Empty,
            error = Empty,
            module = "smg"
        );

        // 0.33 returns a Result; a missing/empty parent context is not actionable here.
        let _ = span.set_parent(parent_cx);
        span
    }
}

/// Custom on_request handler
#[derive(Clone, Debug)]
pub struct RequestLogger;

impl<B> OnRequest<B> for RequestLogger {
    fn on_request(&mut self, request: &Request<B>, span: &Span) {
        let _enter = span.enter();

        // Try to get the request ID from extensions
        // This will work if RequestIdLayer has already run
        if let Some(request_id) = request.extensions().get::<RequestId>() {
            span.record("request_id", request_id.0.as_str());
        }

        let method = method_to_static_str(request.method().as_str());
        let path = matched_path_label(request.extensions());
        Metrics::record_http_request(method, path);

        // Log the request start
        info!(
            target: "smg::request",
            "started processing request"
        );
    }
}

/// Custom on_response handler
#[derive(Clone, Debug, Default)]
pub struct ResponseLogger;

impl<B> OnResponse<B> for ResponseLogger {
    fn on_response(self, response: &Response<B>, latency: Duration, span: &Span) {
        let status = response.status();
        let status_code = status.as_u16();

        // Record these in the span for structured logging/observability tools
        span.record("status_code", status_code);
        // Use microseconds as integer to avoid format! string allocation
        span.record("latency", latency.as_micros() as u64);

        // Log the response completion
        let _enter = span.enter();
        // The caller's method and route, when the bridge ran. Absent only for a
        // response that bypassed the layer stack, which is worth saying.
        let line = response.extensions().get::<RequestLine>();
        let method = line.map_or("<unknown>", |line| line.method);
        let uri = line.map_or("<unknown>", |line| line.uri.as_str());
        let matched_route = line.map_or("<unknown>", |line| line.matched.as_str());
        if status.is_server_error() {
            error!(
                target: "smg::response",
                method,
                uri,
                matched_route,
                status_code,
                "request failed with server error"
            );
        } else if status.is_client_error() {
            // prompt_overflow (engine overflow) and prompt_too_long (reactive
            // compaction) are expected RL terminations; do not treat them as
            // generic client errors. Other 4xx responses keep the WARN.
            let error_code = extract_error_code_from_response(response);
            if error_code == PROMPT_OVERFLOW_ERROR_CODE || error_code == "prompt_too_long" {
                debug!(
                    target: "smg::response",
                    error_code,
                    "request terminated with expected RL termination error"
                );
            } else {
                warn!(
                    target: "smg::response",
                    method,
                    uri,
                    matched_route,
                    status_code,
                    error_code,
                    "request failed with client error"
                );
            }
        } else {
            info!(
                target: "smg::response",
                "finished processing request"
            );
        }
    }
}

/// Create a configured TraceLayer for HTTP logging
/// Note: Actual request/response logging with request IDs is done in RequestIdService
pub fn create_logging_layer() -> TraceLayer<
    tower_http::classify::SharedClassifier<tower_http::classify::ServerErrorsAsFailures>,
    RequestSpan,
    RequestLogger,
    ResponseLogger,
> {
    TraceLayer::new_for_http()
        .make_span_with(RequestSpan)
        .on_request(RequestLogger)
        .on_response(ResponseLogger)
}
