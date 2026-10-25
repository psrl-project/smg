//! Conversion utilities for translating between /v1/responses and /v1/chat/completions formats
//!
//! This module implements the conversion approach where:
//! 1. ResponsesRequest → ChatCompletionRequest (for backend processing)
//! 2. ChatCompletionResponse → ResponsesResponse (for client response)
//!
//! This allows the gRPC router to reuse the existing chat pipeline infrastructure
//! without requiring Python backend changes.

use std::collections::HashMap;
use openai_protocol::{
    chat::{thinking_from_reasoning_effort, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, MessageContent},
    common::{FunctionCallResponse, JsonSchemaFormat, ResponseFormat, ToolCall, UsageInfo},
    responses::{
        CustomToolCallOutputContent, ResponseContentPart, ResponseInput, ResponseInputOutputItem,
        ResponseOutputItem, ResponseReasoningContent::ReasoningText, ResponseStatus,
        ResponsesRequest, ResponsesResponse, ResponsesUsage, StringOrContentParts, TextConfig,
        TextFormat, ResponseTool,
    },
    UNKNOWN_MODEL_ID,
};
use serde_json::Value;
use tracing::warn;

use crate::routers::grpc::common::responses::utils::{
    custom_tool_names, extract_tools_from_response_tools, flatten_namespace_tool_name,
    resolve_namespace_tool_name,
};

/// Accumulates one assistant turn from adjacent Responses input items.
///
/// Codex splits a single assistant turn into several Responses items
/// (`reasoning`, an assistant `message`, and one or more tool-call items), but
/// the TITO store keeps that turn as ONE `ChatMessage::Assistant`. Folding the
/// items back into one message on the request side is what lets the stored
/// prefix hash match; without it every replay forks a phantom trajectory.
///
/// A tool result (`function_call_output` / `custom_tool_call_output`, or a
/// `function_call` carrying its own `output`) always terminates the assistant
/// turn whose call it answers, so it must **flush** the pending assistant before
/// the next turn's items arrive. Buffering tool results without flushing folds
/// every turn of the replayed history into a single assistant message (last
/// content wins, all tool calls accumulated), which never hash-matches the
/// per-turn nodes TITO stored at generation time — the replayed prefix misses
/// and the session forks on every turn.
#[derive(Default)]
struct PendingAssistant {
    content: Option<String>,
    reasoning: Option<String>,
    tool_calls: Vec<ToolCall>,
}

impl PendingAssistant {
    fn set_content(&mut self, text: String) {
        if !text.is_empty() {
            self.content = Some(text);
        }
    }

    fn append_reasoning(&mut self, text: String) {
        if text.is_empty() {
            return;
        }
        match self.reasoning.as_mut() {
            Some(existing) => {
                existing.push('\n');
                existing.push_str(&text);
            }
            None => self.reasoning = Some(text),
        }
    }

    fn push_tool_call(&mut self, call: ToolCall) {
        self.tool_calls.push(call);
    }

    /// Emit the batched assistant message, if any.
    fn flush(&mut self, out: &mut Vec<ChatMessage>) {
        let has_assistant =
            self.content.is_some() || self.reasoning.is_some() || !self.tool_calls.is_empty();
        if has_assistant {
            out.push(ChatMessage::Assistant {
                content: self.content.take().map(MessageContent::Text),
                name: None,
                tool_calls: if self.tool_calls.is_empty() {
                    None
                } else {
                    Some(std::mem::take(&mut self.tool_calls))
                },
                reasoning_content: self.reasoning.take(),
            });
        }
    }

    /// Close the current assistant turn with a tool result: emit the pending
    /// assistant (carrying the call this result answers), then the tool message.
    fn flush_turn_with_tool(&mut self, out: &mut Vec<ChatMessage>, tool: ChatMessage) {
        self.flush(out);
        out.push(tool);
    }
}

/// Text form of a `custom_tool_call_output` payload (string or input-typed parts).
fn custom_tool_output_text(output: &CustomToolCallOutputContent) -> String {
    match output {
        CustomToolCallOutputContent::Text(text) => text.clone(),
        CustomToolCallOutputContent::Parts(parts) => parts
            .iter()
            .filter_map(|part| match part {
                openai_protocol::responses::CustomToolInputContentPart::InputText { text } => {
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// Unwrap a custom tool's arguments back into the free-form payload the client
/// expects in a `custom_tool_call` item.
///
/// The model is asked for a single-`input` function call, but Codex's own base
/// instructions also document an `{"command": ["apply_patch", "<patch>"]}` shape,
/// so accept that (and a lone string value) before falling back to the raw text.
fn custom_tool_input(arguments: &str) -> String {
    let Ok(value) = serde_json::from_str::<Value>(arguments) else {
        return arguments.to_string();
    };
    if let Some(input) = value.get("input").and_then(Value::as_str) {
        return input.to_string();
    }
    if let Some(second) = value
        .get("command")
        .and_then(Value::as_array)
        .and_then(|parts| parts.get(1))
        .and_then(Value::as_str)
    {
        return second.to_string();
    }
    let mut strings = value
        .as_object()
        .into_iter()
        .flatten()
        .filter_map(|(_, v)| v.as_str());
    match (strings.next(), strings.next()) {
        (Some(only), None) => only.to_string(),
        _ => arguments.to_string(),
    }
}

/// Build the Responses output item for one model tool call.
///
/// Tools declared `type:"custom"` in the request are surfaced to the model as
/// single-`input` functions; their calls must be returned as `custom_tool_call`
/// items (with the payload unwrapped) so the client executes them natively.
/// Every other call is a plain `function_call`.
pub(super) fn tool_call_output_item(
    call_id: String,
    name_flat: String,
    arguments: String,
    tools: Option<&[ResponseTool]>,
) -> ResponseOutputItem {
    let (name, namespace) = resolve_namespace_tool_name(tools, &name_flat);
    if custom_tool_names(tools).contains(&name_flat) {
        ResponseOutputItem::CustomToolCall {
            id: Some(call_id.clone()),
            call_id,
            name,
            namespace,
            input: custom_tool_input(&arguments),
            status: Some("in_progress".to_string()),
        }
    } else {
        ResponseOutputItem::FunctionToolCall {
            id: Some(call_id.clone()),
            call_id,
            name,
            namespace,
            arguments,
            output: None, // Tool hasn't been executed yet
            status: "in_progress".to_string(),
        }
    }
}

/// Convert a ResponsesRequest to ChatCompletionRequest for processing through the chat pipeline
///
/// # Conversion Logic
/// - `input` (text/items) → `messages` (chat messages)
/// - `instructions` → system message (prepended)
/// - `max_output_tokens` → `max_completion_tokens`
/// - `tools` → function tools extracted from ResponseTools
/// - `tool_choice` → passed through from request
/// - Response-specific fields (previous_response_id, conversation) are handled by router
pub(crate) fn responses_to_chat(req: &ResponsesRequest) -> Result<ChatCompletionRequest, String> {
    let mut messages = Vec::new();

    // 1. Add system message if instructions provided
    if let Some(instructions) = &req.instructions {
        messages.push(ChatMessage::System {
            content: MessageContent::Text(instructions.clone()),
            name: None,
        });
    }

    // 2. Convert input to chat messages.
    //
    // Adjacent assistant-side items are folded into ONE `ChatMessage::Assistant`
    // (see `PendingAssistant`) so the request-side structure matches the single
    // assistant turn the TITO store recorded at generation time.
    match &req.input {
        ResponseInput::Text(text) => {
            // Simple text input → user message
            messages.push(ChatMessage::User {
                content: MessageContent::Text(text.clone()),
                name: None,
            });
        }
        ResponseInput::Items(items) => {
            let mut pending = PendingAssistant::default();
            for item in items {
                match item {
                    // --- assistant-side items: fold into the pending turn ---
                    ResponseInputOutputItem::Reasoning { content, .. } => {
                        let reasoning_text = content
                            .iter()
                            .map(|c| match c {
                                ReasoningText { text } => text.as_str(),
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        pending.append_reasoning(reasoning_text);
                    }
                    ResponseInputOutputItem::Message { role, content, .. } => {
                        let text = extract_text_from_content(content);
                        if role == "assistant" {
                            pending.set_content(text);
                        } else {
                            pending.flush(&mut messages);
                            messages.push(role_to_chat_message(role.as_str(), text));
                        }
                    }
                    ResponseInputOutputItem::SimpleInputMessage { content, role, .. } => {
                        let text = match content {
                            StringOrContentParts::String(s) => s.clone(),
                            StringOrContentParts::Array(parts) => parts
                                .iter()
                                .filter_map(|part| match part {
                                    ResponseContentPart::InputText { text } => Some(text.as_str()),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join(" "),
                        };
                        if role == "assistant" {
                            pending.set_content(text);
                        } else {
                            pending.flush(&mut messages);
                            messages.push(role_to_chat_message(role.as_str(), text));
                        }
                    }
                    ResponseInputOutputItem::FunctionToolCall {
                        call_id,
                        name,
                        namespace,
                        arguments,
                        output,
                        ..
                    } => {
                        let name = namespace.as_ref().map_or_else(
                            || name.clone(),
                            |namespace| flatten_namespace_tool_name(namespace, name),
                        );
                        pending.push_tool_call(ToolCall {
                            id: call_id.clone(),
                            tool_type: "function".to_string(),
                            function: FunctionCallResponse {
                                name,
                                arguments: Some(arguments.clone()),
                            },
                        });

                        if let Some(output_text) = output {
                            pending.flush_turn_with_tool(
                                &mut messages,
                                ChatMessage::Tool {
                                    content: MessageContent::Text(output_text.clone()),
                                    tool_call_id: call_id.clone(),
                                },
                            );
                        }
                    }
                    ResponseInputOutputItem::CustomToolCall {
                        call_id,
                        name,
                        namespace,
                        input,
                        ..
                    } => {
                        // A `type:"custom"` tool call (e.g. Codex's apply_patch).
                        // The model saw it as a single-`input` function, so re-wrap
                        // the free-form payload as that function's arguments; the
                        // hash then agrees with the stored assistant turn.
                        let name = namespace.as_ref().map_or_else(
                            || name.clone(),
                            |namespace| flatten_namespace_tool_name(namespace, name),
                        );
                        let arguments =
                            serde_json::to_string(&serde_json::json!({ "input": input })).map_err(
                                |error| format!("failed to serialize custom tool input: {error}"),
                            )?;
                        pending.push_tool_call(ToolCall {
                            id: call_id.clone(),
                            tool_type: "function".to_string(),
                            function: FunctionCallResponse {
                                name,
                                arguments: Some(arguments),
                            },
                        });
                    }
                    // --- tool outputs: buffer until the turn boundary ---
                    ResponseInputOutputItem::FunctionCallOutput {
                        call_id, output, ..
                    } => {
                        pending.flush_turn_with_tool(
                            &mut messages,
                            ChatMessage::Tool {
                                content: MessageContent::Text(output.clone()),
                                tool_call_id: call_id.clone(),
                            },
                        );
                    }
                    ResponseInputOutputItem::CustomToolCallOutput {
                        call_id, output, ..
                    } => {
                        pending.flush_turn_with_tool(
                            &mut messages,
                            ChatMessage::Tool {
                                content: MessageContent::Text(custom_tool_output_text(output)),
                                tool_call_id: call_id.clone(),
                            },
                        );
                    }
                    ResponseInputOutputItem::McpApprovalResponse { .. }
                    | ResponseInputOutputItem::McpApprovalRequest { .. }
                    | ResponseInputOutputItem::ComputerCall { .. }
                    | ResponseInputOutputItem::ComputerCallOutput { .. }
                    | ResponseInputOutputItem::McpCall { .. }
                    | ResponseInputOutputItem::McpListTools { .. } => {
                        warn!(
                            function = "responses_to_chat",
                            "Approval item reached chat conversion"
                        );
                        return Err("Unsupported input item type".to_string());
                    }
                    ResponseInputOutputItem::ImageGenerationCall { .. } => {
                        warn!(
                            function = "responses_to_chat",
                            "image_generation_call input item reached chat conversion"
                        );
                        return Err("Unsupported input item type".to_string());
                    }
                    ResponseInputOutputItem::Compaction { .. }
                    | ResponseInputOutputItem::ItemReference { .. } => {
                        return Err("Unsupported input item type".to_string());
                    }
                    ResponseInputOutputItem::ShellCall { .. }
                    | ResponseInputOutputItem::ShellCallOutput { .. } => {
                        warn!(
                            function = "responses_to_chat",
                            "Shell tool item reached chat conversion"
                        );
                        return Err("Unsupported input item type".to_string());
                    }
                    ResponseInputOutputItem::ApplyPatchCall { .. }
                    | ResponseInputOutputItem::ApplyPatchCallOutput { .. } => {
                        warn!(
                            function = "responses_to_chat",
                            "apply_patch item reached chat conversion"
                        );
                        return Err("Unsupported input item type".to_string());
                    }
                    // T5 schema-only: forced-cascade arm, no behavior.
                    ResponseInputOutputItem::LocalShellCall { .. }
                    | ResponseInputOutputItem::LocalShellCallOutput { .. } => {
                        return Err("Unsupported input item type".to_string());
                    }
                }
            }
            pending.flush(&mut messages);
        }
    }

    // Ensure we have at least one message
    if messages.is_empty() {
        return Err("Request must contain at least one message".to_string());
    }

    // 3. Extract function tools from ResponseTools.
    // MCP tools are merged later by the tool loop (see tool_loop.rs:prepare_chat_tools_and_choice).
    let function_tools = extract_tools_from_response_tools(req.tools.as_deref());
    let tools = if function_tools.is_empty() {
        None
    } else {
        Some(function_tools)
    };

    // 4. Build ChatCompletionRequest.
    //
    // `tool_choice` is meaningless without tools, and Codex sends `tool_choice: "auto"`
    // unconditionally (its tools-less compaction turn included), so drop it here rather than
    // forward an inconsistent pair to the chat backend.
    let tool_choice = if tools.is_some() {
        req.tool_choice.as_ref().map(|tc| tc.to_chat_tool_choice())
    } else {
        None
    };
    let is_streaming = req.stream.unwrap_or(false);
    // Forward the Responses `reasoning` effort so the chat render path and the TITO
    // incremental render agree on the template's thinking toggle. When the effort maps to a
    // concrete on/off decision (Codex's `none`/`minimal` disable thinking) pin it through
    // `chat_template_kwargs`, matching the Messages training adapter.
    let reasoning_effort = req
        .reasoning
        .as_ref()
        .and_then(|reasoning| reasoning.effort.as_ref())
        .map(|effort| effort.as_str().to_string());
    let chat_template_kwargs = thinking_from_reasoning_effort(reasoning_effort.as_deref()).map(
        |enabled| {
            HashMap::from([
                ("enable_thinking".to_string(), Value::Bool(enabled)),
                ("thinking".to_string(), Value::Bool(enabled)),
            ])
        },
    );

    Ok(ChatCompletionRequest {
        messages,
        model: if req.model.is_empty() {
            UNKNOWN_MODEL_ID.to_string()
        } else {
            req.model.clone()
        },
        temperature: req.temperature,
        max_completion_tokens: req.max_output_tokens,
        stream: is_streaming,
        // Preserve caller-provided stream_options (e.g. `include_obfuscation: false`
        // on the Responses API) and only default `include_usage` when the caller
        // did not set it. Non-streaming requests intentionally drop stream_options.
        stream_options: if is_streaming {
            let mut opts = req.stream_options.clone().unwrap_or_default();
            if opts.include_usage.is_none() {
                opts.include_usage = Some(true);
            }
            Some(opts)
        } else {
            None
        },
        parallel_tool_calls: req.parallel_tool_calls,
        top_logprobs: req.top_logprobs,
        top_p: req.top_p,
        sampler_top_logprobs: req.sampler_top_logprobs,
        skip_special_tokens: true,
        reasoning_effort,
        chat_template_kwargs,
        tools,
        tool_choice,
        response_format: map_text_to_response_format(req.text.as_ref()),
        ..Default::default()
    })
}

/// Extract text content from ResponseContentPart array. `Refusal` is
/// losslessly representable as text and is preserved verbatim. Image / file
/// parts are currently dropped; the gRPC regular path is text-only and
/// relies on the multimodal pipeline for media handling (R1/R2/R3 will
/// implement full media handling).
fn extract_text_from_content(content: &[ResponseContentPart]) -> String {
    content
        .iter()
        .filter_map(|part| match part {
            ResponseContentPart::InputText { text } => Some(text.as_str()),
            ResponseContentPart::OutputText { text, .. } => Some(text.as_str()),
            ResponseContentPart::Refusal { refusal } => Some(refusal.as_str()),
            // R1/R2/R3 will implement full media handling
            ResponseContentPart::InputImage { .. } | ResponseContentPart::InputFile { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Convert role and text to ChatMessage
fn role_to_chat_message(role: &str, text: String) -> ChatMessage {
    match role {
        "user" => ChatMessage::User {
            content: MessageContent::Text(text),
            name: None,
        },
        "assistant" => ChatMessage::Assistant {
            content: Some(MessageContent::Text(text)),
            name: None,
            tool_calls: None,
            reasoning_content: None,
        },
        "system" => ChatMessage::System {
            content: MessageContent::Text(text),
            name: None,
        },
        _ => {
            // Unknown role, treat as user message
            ChatMessage::User {
                content: MessageContent::Text(text),
                name: None,
            }
        }
    }
}

/// Map TextConfig from Responses API to ResponseFormat for Chat API
///
/// Converts the structured output configuration from the Responses API format
/// to the Chat API format for non-Harmony models.
fn map_text_to_response_format(text: Option<&TextConfig>) -> Option<ResponseFormat> {
    let text_config = text?;
    let format = text_config.format.as_ref()?;

    match format {
        TextFormat::Text => Some(ResponseFormat::Text),
        TextFormat::JsonObject => Some(ResponseFormat::JsonObject),
        TextFormat::JsonSchema {
            name,
            schema,
            description: _,
            strict,
        } => Some(ResponseFormat::JsonSchema {
            json_schema: JsonSchemaFormat {
                name: name.clone(),
                schema: schema.clone(),
                strict: *strict,
            },
        }),
    }
}

/// Convert a ChatCompletionResponse to ResponsesResponse
///
/// # Conversion Logic
/// - `id` → `response_id_override` if provided, otherwise `chat_resp.id`
/// - `model` → `model` (pass through)
/// - `choices[0].message` → `output` array (convert to ResponseOutputItem::Message)
/// - `choices[0].finish_reason` → determines `status` (stop/length → Completed)
/// - `created` timestamp → `created_at`
pub(crate) fn chat_to_responses(
    chat_resp: &ChatCompletionResponse,
    original_req: &ResponsesRequest,
    response_id_override: Option<String>,
) -> Result<ResponsesResponse, String> {
    // Extract the first choice (responses API doesn't support n>1)
    let choice = chat_resp
        .choices
        .first()
        .ok_or_else(|| "Chat response contains no choices".to_string())?;

    // Convert assistant message to output items
    let mut output: Vec<ResponseOutputItem> = Vec::new();

    // Convert message content to output item
    if let Some(content) = &choice.message.content {
        if !content.is_empty() {
            output.push(ResponseOutputItem::Message {
                id: format!("msg_{}", chat_resp.id),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: content.clone(),
                    annotations: vec![],
                    logprobs: choice.logprobs.clone(),
                }],
                status: "completed".to_string(),
                phase: None,
            });
        }
    }

    // Convert reasoning content if present (O1-style models)
    if let Some(reasoning) = &choice.message.reasoning_content {
        if !reasoning.is_empty() {
            output.push(ResponseOutputItem::new_reasoning(
                format!("reasoning_{}", chat_resp.id),
                vec![],
                vec![ReasoningText {
                    text: reasoning.clone(),
                }],
                Some("completed".to_string()),
            ));
        }
    }

    // Convert tool calls if present
    if let Some(tool_calls) = &choice.message.tool_calls {
        for tool_call in tool_calls {
            output.push(tool_call_output_item(
                tool_call.id.clone(),
                tool_call.function.name.clone(),
                tool_call.function.arguments.clone().unwrap_or_default(),
                original_req.tools.as_deref(),
            ));
        }
    }

    // Determine response status based on finish_reason
    let status = match choice.finish_reason.as_deref() {
        Some("stop") | Some("length") => ResponseStatus::Completed,
        Some("tool_calls") => ResponseStatus::InProgress, // Waiting for tool execution
        Some("failed") | Some("error") => ResponseStatus::Failed,
        _ => ResponseStatus::Completed, // Default to completed
    };

    // Convert usage from Usage to UsageInfo, then wrap in ResponsesUsage.
    //
    // The wire shape must be the Responses API one (`input_tokens`/`output_tokens`): a
    // chat-shaped `{prompt_tokens, ...}` object makes strict clients fail to parse
    // `response.completed` and abort the turn (Codex: "missing field `input_tokens`").
    let usage = chat_resp.usage.as_ref().map(|u| {
        let usage_info = UsageInfo {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
            reasoning_tokens: u
                .completion_tokens_details
                .as_ref()
                .and_then(|d| d.reasoning_tokens),
            prompt_tokens_details: None, // Chat response doesn't have this
        };
        ResponsesUsage::Modern(usage_info.to_response_usage())
    });

    // Generate response
    let response_id = response_id_override.unwrap_or_else(|| chat_resp.id.clone());
    Ok(ResponsesResponse::builder(&response_id, &chat_resp.model)
        .copy_from_request(original_req)
        .created_at(chat_resp.created as i64)
        .status(status)
        .output(output)
        .maybe_text(original_req.text.clone())
        .maybe_usage(usage)
        .build())
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::StreamOptions;

    use super::*;

    #[test]
    fn test_text_input_conversion() {
        let req = ResponsesRequest {
            input: ResponseInput::Text("Hello, world!".to_string()),
            instructions: Some("You are a helpful assistant.".to_string()),
            model: "gpt-4".to_string(),
            temperature: Some(0.7),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        assert_eq!(chat_req.messages.len(), 2); // system + user
        assert_eq!(chat_req.model, "gpt-4");
        assert_eq!(chat_req.temperature, Some(0.7));
    }

    #[test]
    fn test_items_input_conversion() {
        let req = ResponsesRequest {
            input: ResponseInput::Items(vec![
                ResponseInputOutputItem::Message {
                    id: "msg_1".to_string(),
                    role: "user".to_string(),
                    content: vec![ResponseContentPart::InputText {
                        text: "Hello!".to_string(),
                    }],
                    status: None,
                    phase: None,
                },
                ResponseInputOutputItem::Message {
                    id: "msg_2".to_string(),
                    role: "assistant".to_string(),
                    content: vec![ResponseContentPart::OutputText {
                        text: "Hi there!".to_string(),
                        annotations: vec![],
                        logprobs: None,
                    }],
                    status: None,
                    phase: None,
                },
            ]),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        assert_eq!(chat_req.messages.len(), 2); // user + assistant
    }

    #[test]
    fn test_function_call_history_uses_call_id_for_chat_tool_messages() {
        let req = ResponsesRequest {
            input: ResponseInput::Items(vec![ResponseInputOutputItem::FunctionToolCall {
                id: Some("fc_item_id".to_string()),
                call_id: "call_tool_id".to_string(),
                name: "lookup".to_string(),
                namespace: None,
                arguments: "{\"q\":\"rust\"}".to_string(),
                output: Some("done".to_string()),
                status: Some("completed".to_string()),
            }]),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        assert_eq!(chat_req.messages.len(), 2);

        match &chat_req.messages[0] {
            ChatMessage::Assistant {
                tool_calls: Some(tool_calls),
                ..
            } => assert_eq!(tool_calls[0].id, "call_tool_id"),
            other => panic!("expected assistant tool call, got {other:?}"),
        }

        match &chat_req.messages[1] {
            ChatMessage::Tool { tool_call_id, .. } => {
                assert_eq!(tool_call_id, "call_tool_id");
            }
            other => panic!("expected tool message, got {other:?}"),
        }
    }

    #[test]
    fn multi_turn_history_keeps_one_assistant_per_turn() {
        // Codex replays each assistant turn as `reasoning` + `message` +
        // `function_call`, immediately followed by that turn's
        // `function_call_output`, then the next turn. The conversion must keep
        // the per-turn boundaries — Assistant(its own tool call) followed by its
        // Tool output, repeated — because TITO stored exactly that shape at
        // generation time. Merging every turn into one Assistant (last content
        // wins, all tool calls accumulated) makes the replayed prefix hash never
        // match the stored node, so every turn forks.
        let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
            "model": "actor",
            "input": [
                {"type":"message","id":"u1","role":"user",
                 "content":[{"type":"input_text","text":"do it"}]},
                {"type":"reasoning","id":"r1","summary":[],
                 "content":[{"type":"reasoning_text","text":"think1"}]},
                {"type":"message","id":"a1","role":"assistant",
                 "content":[{"type":"output_text","text":"answer1","annotations":[]}]},
                {"type":"function_call","id":"fc1","call_id":"c1","name":"exec_command",
                 "arguments":"{\"cmd\":\"ls\"}"},
                {"type":"function_call_output","call_id":"c1","output":"out1"},
                {"type":"reasoning","id":"r2","summary":[],
                 "content":[{"type":"reasoning_text","text":"think2"}]},
                {"type":"message","id":"a2","role":"assistant",
                 "content":[{"type":"output_text","text":"answer2","annotations":[]}]},
                {"type":"function_call","id":"fc2","call_id":"c2","name":"exec_command",
                 "arguments":"{\"cmd\":\"pwd\"}"},
                {"type":"function_call_output","call_id":"c2","output":"out2"}
            ]
        }))
        .expect("valid multi-turn request");

        let chat = responses_to_chat(&req).expect("multi-turn history converts");

        let assistants = chat
            .messages
            .iter()
            .filter(|m| matches!(m, ChatMessage::Assistant { .. }))
            .count();
        let tools = chat
            .messages
            .iter()
            .filter(|m| matches!(m, ChatMessage::Tool { .. }))
            .count();
        assert_eq!(
            assistants, 2,
            "each turn must keep its own assistant message; got {:?}",
            chat.messages
        );
        assert_eq!(
            tools, 2,
            "each turn must keep its own tool output; got {:?}",
            chat.messages
        );
        assert!(matches!(
            &chat.messages[1],
            ChatMessage::Assistant { .. }
        ));
        assert!(matches!(
            &chat.messages[2],
            ChatMessage::Tool { tool_call_id, .. } if tool_call_id == "c1"
        ));
        assert!(matches!(
            &chat.messages[4],
            ChatMessage::Tool { tool_call_id, .. } if tool_call_id == "c2"
        ));
    }

    #[test]
    fn test_namespace_function_roundtrip_preserves_wire_identity() {
        let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
            "model": "actor",
            "input": "delegate",
            "tools": [{
                "type": "namespace",
                "name": "multi_agent_v1",
                "description": "Manage agents",
                "tools": [{
                    "type": "function",
                    "name": "spawn_agent",
                    "description": "Spawn an agent",
                    "parameters": {"type": "object", "properties": {}}
                }]
            }]
        }))
        .expect("valid namespace request");

        let chat_req = responses_to_chat(&req).expect("namespace converts to Chat IR");
        assert_eq!(
            chat_req.tools.as_ref().expect("tool")[0].function.name,
            "multi_agent_v1__spawn_agent"
        );

        let chat_resp: ChatCompletionResponse = serde_json::from_value(serde_json::json!({
            "id": "chatcmpl_namespace",
            "object": "chat.completion",
            "created": 1,
            "model": "actor",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "multi_agent_v1__spawn_agent",
                            "arguments": "{\"task\":\"inspect\"}"
                        }
                    }],
                    "reasoning_content": null
                },
                "finish_reason": "tool_calls"
            }]
        }))
        .expect("valid Chat response");
        let response =
            chat_to_responses(&chat_resp, &req, None).expect("Chat IR converts to Responses");
        let ResponseOutputItem::FunctionToolCall {
            name, namespace, ..
        } = &response.output[0]
        else {
            panic!("expected function call output");
        };
        assert_eq!(name, "spawn_agent");
        assert_eq!(namespace.as_deref(), Some("multi_agent_v1"));
    }

    #[test]
    fn test_empty_input_error() {
        let req = ResponsesRequest {
            input: ResponseInput::Text(String::new()),
            ..Default::default()
        };

        // Empty text should still create a user message, so this should succeed
        let result = responses_to_chat(&req);
        assert!(result.is_ok());
    }

    #[test]
    fn test_stream_options_include_obfuscation_roundtrip() {
        // Regression: ensure caller-provided stream_options (e.g. `include_obfuscation`)
        // are preserved through the Responses → Chat conversion when streaming.
        let req = ResponsesRequest {
            input: ResponseInput::Text("hi".to_string()),
            stream: Some(true),
            stream_options: Some(StreamOptions {
                include_usage: None,
                include_obfuscation: Some(false),
                ..StreamOptions::default()
            }),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        assert!(chat_req.stream);
        let opts = chat_req
            .stream_options
            .expect("stream_options populated when streaming");
        // Caller-provided value is preserved verbatim.
        assert_eq!(opts.include_obfuscation, Some(false));
        // include_usage defaults to true when absent so downstream consumers
        // still emit the usage block at end-of-stream.
        assert_eq!(opts.include_usage, Some(true));
    }

    #[test]
    fn test_stream_options_caller_include_usage_preserved() {
        // Caller-set `include_usage` must not be clobbered by the conversion layer.
        let req = ResponsesRequest {
            input: ResponseInput::Text("hi".to_string()),
            stream: Some(true),
            stream_options: Some(StreamOptions {
                include_usage: Some(false),
                include_obfuscation: Some(true),
                ..StreamOptions::default()
            }),
            ..Default::default()
        };

        let opts = responses_to_chat(&req).unwrap().stream_options.unwrap();
        assert_eq!(opts.include_usage, Some(false));
        assert_eq!(opts.include_obfuscation, Some(true));
    }

    #[test]
    fn test_stream_options_non_streaming_dropped() {
        // stream=false must produce None stream_options even if caller set it.
        let req = ResponsesRequest {
            input: ResponseInput::Text("hi".to_string()),
            stream: Some(false),
            stream_options: Some(StreamOptions {
                include_usage: Some(true),
                include_obfuscation: Some(false),
                ..StreamOptions::default()
            }),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        assert!(!chat_req.stream);
        assert!(chat_req.stream_options.is_none());
    }

    #[test]
    fn test_image_generation_call_input_rejected() {
        // Regression: `image_generation_call` items are server-produced
        // output (populated via the shared MCP transformer) and must not
        // be round-tripped back into the chat conversion as input.
        // The regular gRPC path — used by non-Harmony text LLMs that only do
        // function calling — rejects this variant with the same contract as
        // sibling hosted-tool items (Computer/Shell/Custom/ApplyPatch).
        let req = ResponsesRequest {
            input: ResponseInput::Items(vec![ResponseInputOutputItem::ImageGenerationCall {
                id: "ig_test".to_string(),
                action: None,
                background: None,
                output_format: None,
                quality: None,
                result: Some("base64data".to_string()),
                revised_prompt: Some("a cat".to_string()),
                size: None,
                status: None,
            }]),
            ..Default::default()
        };

        let result = responses_to_chat(&req);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Unsupported input item type");
    }

    // --- Codex assistant-item merging (TITO structure) ---

    fn items_request(items: Value) -> ResponsesRequest {
        serde_json::from_value(serde_json::json!({ "model": "actor", "input": items }))
            .expect("valid request")
    }

    fn assistant(messages: &[ChatMessage], index: usize) -> &ChatMessage {
        messages.get(index).expect("message at index")
    }

    #[test]
    fn reasoning_and_tool_call_merge_into_one_assistant() {
        // Codex resends one assistant turn as a reasoning item plus a
        // function_call item; both must fold back into a single ChatMessage.
        let req = items_request(serde_json::json!([
            {
                "type": "reasoning",
                "id": "r1",
                "summary": [],
                "content": [{ "type": "reasoning_text", "text": "thinking" }],
            },
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "exec_command",
                "arguments": "{\"cmd\":\"ls\"}",
            },
        ]));

        let chat = responses_to_chat(&req).expect("converts");
        assert_eq!(chat.messages.len(), 1, "assistant items must merge into one message");
        match assistant(&chat.messages, 0) {
            ChatMessage::Assistant {
                tool_calls: Some(calls),
                reasoning_content,
                ..
            } => {
                assert_eq!(reasoning_content.as_deref(), Some("thinking"));
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "exec_command");
            }
            other => panic!("expected one merged assistant message, got {other:?}"),
        }
    }

    #[test]
    fn custom_tool_call_and_output_roundtrip() {
        // A `type:"custom"` tool call (Codex apply_patch) is re-wrapped as a
        // single-`input` function call, and its output becomes a tool message.
        let req = items_request(serde_json::json!([
            {
                "type": "custom_tool_call",
                "call_id": "p1",
                "name": "apply_patch",
                "input": "*** Begin Patch\n*** End Patch",
            },
            {
                "type": "custom_tool_call_output",
                "call_id": "p1",
                "output": "Done!",
            },
        ]));

        let chat = responses_to_chat(&req).expect("converts");
        assert_eq!(chat.messages.len(), 2);
        match assistant(&chat.messages, 0) {
            ChatMessage::Assistant {
                tool_calls: Some(calls),
                ..
            } => {
                assert_eq!(calls[0].function.name, "apply_patch");
                assert_eq!(
                    calls[0].function.arguments.as_deref(),
                    Some("{\"input\":\"*** Begin Patch\\n*** End Patch\"}")
                );
            }
            other => panic!("expected assistant tool call, got {other:?}"),
        }
        match &chat.messages[1] {
            ChatMessage::Tool { content, tool_call_id } => {
                assert_eq!(tool_call_id, "p1");
                assert!(matches!(content, MessageContent::Text(text) if text == "Done!"));
            }
            other => panic!("expected tool message, got {other:?}"),
        }
    }

    #[test]
    fn parallel_calls_group_before_their_outputs() {
        let req = items_request(serde_json::json!([
            { "type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}" },
            { "type": "function_call", "call_id": "c2", "name": "f", "arguments": "{}" },
            { "type": "function_call_output", "call_id": "c1", "output": "o1" },
            { "type": "function_call_output", "call_id": "c2", "output": "o2" },
        ]));

        let chat = responses_to_chat(&req).expect("converts");
        assert_eq!(chat.messages.len(), 3);
        match assistant(&chat.messages, 0) {
            ChatMessage::Assistant {
                tool_calls: Some(calls),
                ..
            } => assert_eq!(calls.len(), 2),
            other => panic!("expected grouped assistant, got {other:?}"),
        }
        assert!(matches!(chat.messages[1], ChatMessage::Tool { .. }));
        assert!(matches!(chat.messages[2], ChatMessage::Tool { .. }));
    }

    #[test]
    fn user_message_between_assistant_items_splits_turns() {
        let req = items_request(serde_json::json!([
            { "type": "message", "id": "m1", "role": "assistant",
              "content": [{ "type": "output_text", "text": "a" }] },
            { "type": "message", "id": "m2", "role": "user",
              "content": [{ "type": "input_text", "text": "next" }] },
            { "type": "message", "id": "m3", "role": "assistant",
              "content": [{ "type": "output_text", "text": "b" }] },
        ]));

        let chat = responses_to_chat(&req).expect("converts");
        assert_eq!(chat.messages.len(), 3);
        assert!(matches!(chat.messages[0], ChatMessage::Assistant { .. }));
        assert!(matches!(chat.messages[1], ChatMessage::User { .. }));
        assert!(matches!(chat.messages[2], ChatMessage::Assistant { .. }));
    }

    #[test]
    fn chat_to_responses_emits_custom_tool_call_for_custom_tools() {
        let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
            "model": "actor",
            "input": "patch it",
            "tools": [{ "type": "custom", "name": "apply_patch" }],
        }))
        .expect("valid request");

        let chat_resp: ChatCompletionResponse = serde_json::from_value(serde_json::json!({
            "id": "chatcmpl_1",
            "object": "chat.completion",
            "created": 1,
            "model": "actor",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "apply_patch",
                            "arguments": "{\"input\":\"*** Begin Patch\"}"
                        }
                    }],
                    "reasoning_content": null
                },
                "finish_reason": "tool_calls"
            }]
        }))
        .expect("valid chat response");

        let response = chat_to_responses(&chat_resp, &req, None).expect("converts");
        match &response.output[0] {
            ResponseOutputItem::CustomToolCall {
                call_id, name, input, ..
            } => {
                assert_eq!(call_id, "call_1");
                assert_eq!(name, "apply_patch");
                // The `{"input": ...}` shim is unwrapped back to the raw payload.
                assert_eq!(input, "*** Begin Patch");
            }
            other => panic!("expected custom_tool_call, got {other:?}"),
        }
    }

    #[test]
    fn chat_to_responses_keeps_function_call_for_plain_tools() {
        let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
            "model": "actor",
            "input": "run it",
            "tools": [{ "type": "function", "name": "exec_command",
                        "parameters": { "type": "object" } }],
        }))
        .expect("valid request");

        let chat_resp: ChatCompletionResponse = serde_json::from_value(serde_json::json!({
            "id": "chatcmpl_2",
            "object": "chat.completion",
            "created": 1,
            "model": "actor",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_2",
                        "type": "function",
                        "function": { "name": "exec_command", "arguments": "{\"cmd\":\"ls\"}" }
                    }],
                    "reasoning_content": null
                },
                "finish_reason": "tool_calls"
            }]
        }))
        .expect("valid chat response");

        let response = chat_to_responses(&chat_resp, &req, None).expect("converts");
        assert!(matches!(
            &response.output[0],
            ResponseOutputItem::FunctionToolCall { name, .. } if name == "exec_command"
        ));
    }

    #[test]
    fn custom_tool_input_accepts_documented_shapes() {
        assert_eq!(custom_tool_input("{\"input\":\"P\"}"), "P");
        // Codex's base instructions document this command-array shape.
        assert_eq!(custom_tool_input("{\"command\":[\"apply_patch\",\"P\"]}"), "P");
        assert_eq!(custom_tool_input("{\"patch\":\"P\"}"), "P");
        // Anything ambiguous or non-JSON is passed through untouched.
        assert_eq!(custom_tool_input("{\"a\":\"1\",\"b\":\"2\"}"), "{\"a\":\"1\",\"b\":\"2\"}");
        assert_eq!(custom_tool_input("raw patch"), "raw patch");
    }

    #[test]
    fn tool_choice_is_dropped_when_no_tools_reach_the_model() {
        // Codex sends `tool_choice: "auto"` even on tools-less (compaction) turns; the chat
        // backend must not receive the inconsistent pair.
        let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
            "model": "actor",
            "input": "summarize the conversation",
            "tool_choice": "auto",
        }))
        .expect("valid request");
        let chat = responses_to_chat(&req).expect("converts");
        assert!(chat.tool_choice.is_none());
        assert!(chat.tools.is_none());

        // With tools present the choice is preserved.
        let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
            "model": "actor",
            "input": "go",
            "tool_choice": "auto",
            "tools": [{ "type": "function", "name": "exec_command", "parameters": { "type": "object" } }],
        }))
        .expect("valid request");
        let chat = responses_to_chat(&req).expect("converts");
        assert!(chat.tool_choice.is_some());
        assert!(chat.tools.is_some());
    }

    #[test]
    fn responses_usage_uses_the_responses_api_shape() {
        // Codex aborts the turn when `response.completed.usage` lacks `input_tokens`, so the
        // `/responses` reply must carry the Responses-API usage shape, not the chat one.
        let req: ResponsesRequest =
            serde_json::from_value(serde_json::json!({ "model": "actor", "input": "hi" }))
                .expect("valid request");
        let chat_resp: ChatCompletionResponse = serde_json::from_value(serde_json::json!({
            "id": "chatcmpl_u",
            "object": "chat.completion",
            "created": 1,
            "model": "actor",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "done", "tool_calls": null, "reasoning_content": null },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14 }
        }))
        .expect("valid chat response");

        let response = chat_to_responses(&chat_resp, &req, None).expect("converts");
        let usage = serde_json::to_value(&response).expect("serialises")["usage"].clone();
        assert_eq!(usage["input_tokens"], 11);
        assert_eq!(usage["output_tokens"], 3);
        assert_eq!(usage["total_tokens"], 14);
        assert!(usage.get("prompt_tokens").is_none());
    }

    #[test]
    fn custom_tool_round_trips_through_codex_replay() {
        // End-to-end shape check for Codex's apply_patch:
        //   custom tool -> single-`input` function -> model call -> `custom_tool_call`
        //   -> replayed as an assistant tool call whose arguments match the model's
        //   original call, so the TITO prefix hash agrees with the stored turn.
        let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
            "model": "actor",
            "input": "fix it",
            "tools": [{ "type": "custom", "name": "apply_patch", "description": "patch" }],
        }))
        .expect("valid request");

        // 1. apply_patch reaches the model as a single-`input` function.
        let tools = extract_tools_from_response_tools(req.tools.as_deref());
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "apply_patch");
        assert_eq!(tools[0].function.parameters["required"], serde_json::json!(["input"]));

        let patch = "*** Begin Patch\n*** Update File: a.py\n+x\n*** End Patch";
        let chat_resp: ChatCompletionResponse = serde_json::from_value(serde_json::json!({
            "id": "chatcmpl_patch",
            "object": "chat.completion",
            "created": 1,
            "model": "actor",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_p",
                        "type": "function",
                        "function": {
                            "name": "apply_patch",
                            "arguments": serde_json::json!({ "input": patch }).to_string(),
                        }
                    }],
                    "reasoning_content": null
                },
                "finish_reason": "tool_calls"
            }]
        }))
        .expect("valid chat response");

        // 2. The response carries a custom_tool_call with the payload unwrapped.
        let response = chat_to_responses(&chat_resp, &req, None).expect("converts");
        let item = serde_json::to_value(&response.output[0]).expect("serialises");
        assert_eq!(item["type"], "custom_tool_call");
        assert_eq!(item["name"], "apply_patch");
        assert_eq!(item["input"], patch);

        // 3. Codex resends the call plus its output; it collapses back to the same
        //    assistant tool call (arguments equal to the model's original call).
        let replay = items_request(serde_json::json!([
            item,
            { "type": "custom_tool_call_output", "call_id": "call_p", "output": "Done!" }
        ]));
        let chat = responses_to_chat(&replay).expect("replay converts");
        match assistant(&chat.messages, 0) {
            ChatMessage::Assistant {
                tool_calls: Some(calls),
                ..
            } => {
                assert_eq!(calls[0].function.name, "apply_patch");
                let round_tripped: Value =
                    serde_json::from_str(calls[0].function.arguments.as_deref().unwrap()).unwrap();
                assert_eq!(round_tripped, serde_json::json!({ "input": patch }));
            }
            other => panic!("expected assistant tool call, got {other:?}"),
        }
    }
}
