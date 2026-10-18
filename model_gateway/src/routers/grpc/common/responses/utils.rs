//! Utility functions for /v1/responses endpoint

use std::sync::Arc;

use axum::response::Response;
use openai_protocol::{
    common::{Function, Tool},
    responses::{NamespaceTool, ResponseTool, ResponsesRequest, ResponsesResponse},
};
use serde_json::{json, to_value};
use smg_data_connector::{
    ConversationItemStorage, ConversationStorage, RequestContext as StorageRequestContext,
    ResponseStorage,
};
use smg_mcp::{McpOrchestrator, McpServerBinding};
use tracing::{debug, error, warn};

use crate::{
    routers::{
        common::{
            mcp_utils::ensure_request_mcp_client, openai_bridge,
            persistence_utils::persist_conversation_items,
        },
        error,
    },
    worker::WorkerRegistry,
};

const NAMESPACE_TOOL_SEPARATOR: &str = "__";

pub(crate) fn flatten_namespace_tool_name(namespace: &str, name: &str) -> String {
    format!("{namespace}{NAMESPACE_TOOL_SEPARATOR}{name}")
}

/// Restore a model-facing flattened function name to its Responses namespace.
///
/// Resolution is driven by the original typed tool definitions instead of
/// splitting arbitrary names, so ordinary function names containing `__`
/// remain unchanged.
pub(crate) fn resolve_namespace_tool_name(
    response_tools: Option<&[ResponseTool]>,
    flattened_name: &str,
) -> (String, Option<String>) {
    let Some(tools) = response_tools else {
        return (flattened_name.to_string(), None);
    };
    for tool in tools {
        let ResponseTool::Namespace(namespace) = tool else {
            continue;
        };
        for nested in &namespace.tools {
            let name = match nested {
                NamespaceTool::Function(function) => function.function.name.as_str(),
                NamespaceTool::Custom(custom) => custom.name.as_str(),
            };
            if flatten_namespace_tool_name(&namespace.name, name) == flattened_name {
                return (name.to_string(), Some(namespace.name.clone()));
            }
        }
    }
    (flattened_name.to_string(), None)
}

/// Ensure MCP connection succeeds if MCP tools or builtin tools are declared.
///
/// Checks if the request declares MCP tools or builtin tool types
/// (`web_search_preview`, `code_interpreter`, `image_generation`) and,
/// if so, validates that the MCP clients can be created and connected.
///
/// Returns Ok((has_mcp_tools, mcp_servers)) on success.
pub(crate) async fn ensure_mcp_connection(
    mcp_orchestrator: &Arc<McpOrchestrator>,
    format_registry: &openai_bridge::FormatRegistry,
    tools: Option<&[ResponseTool]>,
) -> Result<(bool, Vec<McpServerBinding>), Response> {
    // Check for explicit MCP tools (must error if connection fails)
    let has_explicit_mcp_tools = tools
        .map(|t| t.iter().any(|tool| matches!(tool, ResponseTool::Mcp(_))))
        .unwrap_or(false);

    // Check for builtin tools that MAY have MCP routing configured.
    //
    // `ImageGeneration` is included here because gpt-oss via the
    // harmony pipeline, and Qwen/Llama via the regular pipeline, both
    // dispatch hosted `image_generation` calls through the same MCP
    // routing path — the only difference is how the tool is advertised in
    // the prompt. Without this arm, the short-circuit below would return
    // `(false, Vec::new())`, the MCP loop would never be entered, and the
    // registered `image_generation` MCP server would receive zero
    // dispatches.
    let has_builtin_tools = tools
        .map(|t| {
            t.iter()
                .any(|tool| openai_bridge::builtin_type_for_response_tool(tool).is_some())
        })
        .unwrap_or(false);

    // Only process if we have MCP or builtin tools
    if !has_explicit_mcp_tools && !has_builtin_tools {
        return Ok((false, Vec::new()));
    }

    if let Some(tools) = tools {
        // TODO: Thread real request headers through the gRPC responses path if/when
        // gRPC MCP flows need the same forwarded-header preservation contract.
        match ensure_request_mcp_client(mcp_orchestrator, format_registry, tools).await {
            Some(mcp_servers) => {
                return Ok((true, mcp_servers));
            }
            None => {
                // No MCP servers available
                if has_explicit_mcp_tools {
                    // Explicit MCP tools MUST have working connections
                    error!(
                        function = "ensure_mcp_connection",
                        "Failed to connect to MCP servers"
                    );
                    return Err(error::failed_dependency(
                        "connect_mcp_server_failed",
                        "Failed to connect to MCP servers. Check server_url and authorization.",
                    ));
                }
                // Builtin tools without MCP routing - pass through to model
                debug!(
                    function = "ensure_mcp_connection",
                    "No MCP routing configured for builtin tools, passing through to model"
                );
                return Ok((false, Vec::new()));
            }
        }
    }

    Ok((false, Vec::new()))
}

/// Validate that workers are available for the requested model
pub(crate) fn validate_worker_availability(
    worker_registry: &Arc<WorkerRegistry>,
    model: &str,
) -> Option<Response> {
    let available_models = worker_registry.get_models();

    if !available_models.contains(&model.to_string()) {
        return Some(error::model_not_found(model));
    }

    None
}

/// Extract function tools from ResponseTools
///
/// This utility consolidates the logic for extracting tools with schemas from ResponseTools.
/// It's used by both Harmony and Regular routers for different purposes:
///
/// - **Harmony router**: Extracts function tools because MCP tools are exposed to the model as
///   function tools (via `convert_mcp_tools_to_response_tools()`), and those are used to
///   generate structural constraints in the Harmony preparation stage.
///
/// - **Regular router**: Extracts function tools during the initial conversion from
///   ResponsesRequest to ChatCompletionRequest. MCP tools are merged later by the tool loop.
pub(crate) fn extract_tools_from_response_tools(
    response_tools: Option<&[ResponseTool]>,
) -> Vec<Tool> {
    let Some(tools) = response_tools else {
        return Vec::new();
    };

    tools
        .iter()
        .flat_map(|rt| match rt {
            ResponseTool::Function(ft) => vec![Tool {
                tool_type: "function".to_string(),
                function: ft.function.clone(),
            }],
            // A user-defined `type:"custom"` tool (e.g. Codex's `apply_patch`)
            // carries a free-form `input` string rather than a JSON schema. The
            // chat template can only render function tools, so expose it as a
            // single-`input` function; `chat_to_responses` maps the call back to
            // a `custom_tool_call` for tools listed here.
            ResponseTool::Custom(custom) => vec![custom_tool_as_function(custom.name.clone(), custom.description.clone())],
            ResponseTool::Namespace(namespace) => namespace
                .tools
                .iter()
                .map(|tool| match tool {
                    NamespaceTool::Function(function) => {
                        let mut function = function.function.clone();
                        function.name =
                            flatten_namespace_tool_name(&namespace.name, &function.name);
                        Tool {
                            tool_type: "function".to_string(),
                            function,
                        }
                    }
                    NamespaceTool::Custom(custom) => custom_tool_as_function(
                        flatten_namespace_tool_name(&namespace.name, &custom.name),
                        custom.description.clone(),
                    ),
                })
                .collect(),
            _ => Vec::new(),
        })
        .collect()
}

/// Model-facing name of a custom tool, flattened under its namespace if any.
fn custom_tool_flat_name(namespace: Option<&str>, name: &str) -> String {
    match namespace {
        Some(namespace) => flatten_namespace_tool_name(namespace, name),
        None => name.to_string(),
    }
}

/// Expose a `type:"custom"` tool to the chat model as a single-`input` function.
///
/// The model cannot be grammar-constrained through the chat template, so the
/// free-form payload is carried as one string parameter. `chat_to_responses`
/// unwraps it back into the custom tool's `input` field.
fn custom_tool_as_function(name: String, description: Option<String>) -> Tool {
    Tool {
        tool_type: "function".to_string(),
        function: Function {
            name,
            description,
            parameters: json!({
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "The free-form payload for this tool.",
                    }
                },
                "required": ["input"],
            }),
            strict: Some(false),
        },
    }
}

/// Names (model-facing, namespace-flattened) of every `type:"custom"` tool in
/// the request. `chat_to_responses` uses this set to decide whether a tool call
/// is emitted as a `custom_tool_call` or a plain `function_call`.
pub(crate) fn custom_tool_names(response_tools: Option<&[ResponseTool]>) -> std::collections::HashSet<String> {
    let Some(tools) = response_tools else {
        return std::collections::HashSet::new();
    };
    let mut names = std::collections::HashSet::new();
    for tool in tools {
        match tool {
            ResponseTool::Custom(custom) => {
                names.insert(custom_tool_flat_name(None, &custom.name));
            }
            ResponseTool::Namespace(namespace) => {
                for nested in &namespace.tools {
                    if let NamespaceTool::Custom(custom) = nested {
                        names.insert(custom_tool_flat_name(Some(&namespace.name), &custom.name));
                    }
                }
            }
            _ => {}
        }
    }
    names
}

/// Persist response to storage if store=true
///
/// Common helper function to avoid duplication across sync and streaming paths
/// in both harmony and regular responses implementations.
pub(crate) async fn persist_response_if_needed(
    conversation_storage: Arc<dyn ConversationStorage>,
    conversation_item_storage: Arc<dyn ConversationItemStorage>,
    response_storage: Arc<dyn ResponseStorage>,
    response: &ResponsesResponse,
    original_request: &ResponsesRequest,
    request_context: Option<StorageRequestContext>,
) {
    if !original_request.store.unwrap_or(true) {
        return;
    }

    if let Ok(response_json) = to_value(response) {
        if let Err(e) = persist_conversation_items(
            conversation_storage,
            conversation_item_storage,
            response_storage,
            &response_json,
            original_request,
            request_context,
        )
        .await
        {
            warn!("Failed to persist response: {}", e);
        } else {
            debug!("Persisted response: {}", response.id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tools(value: serde_json::Value) -> Vec<ResponseTool> {
        serde_json::from_value(value).expect("valid tools")
    }

    #[test]
    fn custom_tool_becomes_single_input_function() {
        let tools = tools(json!([
            { "type": "custom", "name": "apply_patch", "description": "patch files" }
        ]));

        let extracted = extract_tools_from_response_tools(Some(&tools));

        assert_eq!(extracted.len(), 1);
        assert_eq!(extracted[0].function.name, "apply_patch");
        assert_eq!(extracted[0].function.description.as_deref(), Some("patch files"));
        assert_eq!(extracted[0].function.parameters["required"], json!(["input"]));
        assert_eq!(
            extracted[0].function.parameters["properties"]["input"]["type"],
            "string"
        );
    }

    #[test]
    fn custom_tool_names_flatten_namespace() {
        let tools = tools(json!([
            { "type": "custom", "name": "apply_patch" },
            { "type": "namespace", "name": "ns", "description": "group",
              "tools": [{ "type": "custom", "name": "shell" }] }
        ]));

        let names = custom_tool_names(Some(&tools));

        assert!(names.contains("apply_patch"));
        assert!(names.contains("ns__shell"));
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn namespaced_custom_extracted_with_flattened_name() {
        let tools = tools(json!([
            { "type": "namespace", "name": "ns", "description": "group",
              "tools": [{ "type": "custom", "name": "shell" }] }
        ]));

        let extracted = extract_tools_from_response_tools(Some(&tools));

        assert_eq!(extracted.len(), 1);
        assert_eq!(extracted[0].function.name, "ns__shell");
    }
}
