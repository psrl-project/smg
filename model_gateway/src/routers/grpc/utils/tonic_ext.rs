//! Extension traits for tonic gRPC types.

use axum::response::Response;
use http::StatusCode;
use tonic::Code;

use crate::routers::error::{self, PROMPT_OVERFLOW_ERROR_CODE};

const PROMPT_OVERFLOW_MARKERS: &[&str] = &[
    "longer than the maximum model length",
    "exceeds the model's maximum context length",
];

/// Return whether `status` is a prompt-overflow INVALID_ARGUMENT.
///
/// Prefers the servicer's `x-smg-error-code` trailing metadata when present,
/// then falls back to well-known vLLM / servicer message markers.
pub(crate) fn is_prompt_overflow_status(status: &tonic::Status) -> bool {
    if status.code() != Code::InvalidArgument {
        return false;
    }
    if let Some(value) = status.metadata().get("x-smg-error-code") {
        if value.to_str().ok() == Some(PROMPT_OVERFLOW_ERROR_CODE) {
            return true;
        }
    }
    let message = status.message();
    PROMPT_OVERFLOW_MARKERS
        .iter()
        .any(|marker| message.contains(marker))
}

/// Extension methods for `tonic::Status`.
pub(crate) trait TonicStatusExt {
    /// Map gRPC status code to the corresponding HTTP status code.
    fn http_status(&self) -> StatusCode;

    /// Convert this gRPC error into an HTTP error response with the appropriate status code.
    fn to_http_error(&self, code: &str, msg: String) -> Response;
}

impl TonicStatusExt for tonic::Status {
    fn http_status(&self) -> StatusCode {
        match self.code() {
            Code::Ok => StatusCode::OK,
            Code::InvalidArgument
            | Code::FailedPrecondition
            | Code::OutOfRange
            | Code::Cancelled => StatusCode::BAD_REQUEST,
            Code::Unauthenticated => StatusCode::UNAUTHORIZED,
            Code::PermissionDenied => StatusCode::FORBIDDEN,
            Code::NotFound => StatusCode::NOT_FOUND,
            Code::AlreadyExists | Code::Aborted => StatusCode::CONFLICT,
            Code::ResourceExhausted => StatusCode::TOO_MANY_REQUESTS,
            Code::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
            Code::DeadlineExceeded => StatusCode::GATEWAY_TIMEOUT,
            Code::Unimplemented => StatusCode::NOT_IMPLEMENTED,
            // Internal, Unknown, DataLoss
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn to_http_error(&self, code: &str, msg: String) -> Response {
        error::create_error(self.http_status(), code, msg)
    }
}

/// Extension for `Result<T, tonic::Status>` to extract HTTP status for CB recording.
pub(crate) trait TonicResultExt {
    /// Returns the HTTP status code for circuit breaker recording.
    /// `Ok` → 200, `Err(status)` → mapped HTTP status code.
    fn cb_status_code(&self) -> u16;
}

impl<T> TonicResultExt for Result<T, tonic::Status> {
    fn cb_status_code(&self) -> u16 {
        self.as_ref()
            .map_or_else(|e| e.http_status().as_u16(), |_| 200)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tonic::metadata::MetadataMap;

    #[test]
    fn detects_prompt_overflow_from_message() {
        let status = tonic::Status::invalid_argument(
            "The prompt (length 32828) is longer than the maximum model length of 32768.",
        );
        assert!(is_prompt_overflow_status(&status));
    }

    #[test]
    fn detects_prompt_overflow_from_metadata() {
        let mut metadata = MetadataMap::new();
        metadata.insert("x-smg-error-code", PROMPT_OVERFLOW_ERROR_CODE.parse().unwrap());
        let status = tonic::Status::with_metadata(
            Code::InvalidArgument,
            "overlong",
            metadata,
        );
        assert!(is_prompt_overflow_status(&status));
    }

    #[test]
    fn ignores_other_invalid_argument() {
        let status = tonic::Status::invalid_argument("temperature must be >= 0");
        assert!(!is_prompt_overflow_status(&status));
    }
}
