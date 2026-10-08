//! Canonical training adapter for the Anthropic Messages API.
//!
//! Normal Messages traffic keeps using the dedicated Messages pipeline. When
//! a TITO session header is present, the request is lowered inside SMG to the
//! canonical Chat training IR so token hashing, logprobs, partial rollout, and
//! TITO capture have one implementation. The result is rendered back to the
//! native Anthropic wire shape before leaving the gateway.

use std::collections::HashMap;

use openai_protocol::{
    chat::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage, MessageContent},
    common::{ContentPart, FunctionCallResponse, ImageUrl, StringOrArray, ToolCall},
    messages::{
        self, ContentBlock, CreateMessageRequest, DocumentSource, ImageSource, InputContent,
        InputContentBlock, Message, Role, StopReason, SystemContent, ToolResultContent,
        ToolResultContentBlock,
    },
};
use serde_json::{Map, Value};
use tracing::warn;

use crate::routers::grpc::{regular::training, utils::message_utils};

fn strip_billing_header(text: &str) -> String {
    text.lines()
        .filter(|line| {
            !line
                .trim_start()
                .to_ascii_lowercase()
                .starts_with("x-anthropic-billing-header:")
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn system_text(system: Option<&SystemContent>) -> String {
    let text = match system {
        Some(SystemContent::String(text)) => text.clone(),
        Some(SystemContent::Blocks(blocks)) => blocks
            .iter()
            .map(|block| {
                let messages::SystemContentBlock::Text(text) = block;
                text.text.as_str()
            })
            .collect::<Vec<_>>()
            .join("\n"),
        None => String::new(),
    };
    strip_billing_header(&text)
}

fn image_part(source: &ImageSource) -> ContentPart {
    let url = match source {
        ImageSource::Base64 { media_type, data } => {
            format!("data:{media_type};base64,{data}")
        }
        ImageSource::Url { url } => url.clone(),
    };
    ContentPart::ImageUrl {
        image_url: ImageUrl { url, detail: None },
    }
}

fn document_text(source: &DocumentSource) -> Result<String, String> {
    match source {
        DocumentSource::Text { data } => Ok(data.clone()),
        DocumentSource::Content { content } => {
            let mut parts = Vec::new();
            for block in content {
                if let InputContentBlock::Text(text) = block {
                    parts.push(text.text.clone());
                } else {
                    return Err(
                        "TITO Messages document content supports text blocks only".to_string()
                    );
                }
            }
            Ok(parts.join("\n"))
        }
        DocumentSource::Base64 { .. } | DocumentSource::Url { .. } => Err(
            "TITO Messages cannot represent binary or URL documents in the canonical training IR"
                .to_string(),
        ),
    }
}

fn flush_user_parts(parts: &mut Vec<ContentPart>, output: &mut Vec<ChatMessage>) {
    if parts.is_empty() {
        return;
    }
    let content = if parts
        .iter()
        .all(|part| matches!(part, ContentPart::Text { .. }))
    {
        let text = parts
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        MessageContent::Text(text)
    } else {
        MessageContent::Parts(std::mem::take(parts))
    };
    output.push(ChatMessage::User {
        content,
        name: None,
    });
    parts.clear();
}

fn tool_result_text(content: Option<&ToolResultContent>) -> Result<String, String> {
    match content {
        None => Ok(String::new()),
        Some(ToolResultContent::String(text)) => Ok(text.clone()),
        Some(ToolResultContent::Blocks(blocks)) => {
            let mut text = Vec::new();
            for block in blocks {
                match block {
                    ToolResultContentBlock::Text(value) => text.push(value.text.clone()),
                    ToolResultContentBlock::Document(value) => {
                        text.push(document_text(&value.source)?);
                    }
                    ToolResultContentBlock::Image(_) | ToolResultContentBlock::SearchResult(_) => {
                        return Err(
                            "TITO Messages tool_result supports text content only".to_string()
                        );
                    }
                }
            }
            Ok(text.join("\n"))
        }
    }
}

fn user_messages(content: &InputContent) -> Result<Vec<ChatMessage>, String> {
    let blocks = match content {
        InputContent::String(text) => {
            return Ok(vec![ChatMessage::User {
                content: MessageContent::Text(text.clone()),
                name: None,
            }]);
        }
        InputContent::Blocks(blocks) => blocks,
    };
    let mut output = Vec::new();
    let mut parts = Vec::new();
    for block in blocks {
        match block {
            InputContentBlock::Text(text) => {
                parts.push(ContentPart::Text {
                    text: text.text.clone(),
                });
            }
            InputContentBlock::Image(image) => parts.push(image_part(&image.source)),
            InputContentBlock::Document(document) => {
                parts.push(ContentPart::Text {
                    text: document_text(&document.source)?,
                });
            }
            InputContentBlock::ToolResult(result) => {
                flush_user_parts(&mut parts, &mut output);
                output.push(ChatMessage::Tool {
                    content: MessageContent::Text(tool_result_text(result.content.as_ref())?),
                    tool_call_id: result.tool_use_id.clone(),
                });
            }
            InputContentBlock::Thinking(_)
            | InputContentBlock::RedactedThinking(_)
            | InputContentBlock::ToolUse(_)
            | InputContentBlock::ServerToolUse(_)
            | InputContentBlock::SearchResult(_)
            | InputContentBlock::WebSearchToolResult(_)
            | InputContentBlock::ToolSearchToolResult(_)
            | InputContentBlock::ToolReference(_) => {
                return Err("unsupported block in a TITO Messages user message".to_string());
            }
        }
    }
    flush_user_parts(&mut parts, &mut output);
    Ok(output)
}

fn assistant_message(content: &InputContent) -> Result<ChatMessage, String> {
    let blocks = match content {
        InputContent::String(text) => {
            return Ok(ChatMessage::Assistant {
                content: Some(MessageContent::Text(text.clone())),
                name: None,
                tool_calls: None,
                reasoning_content: None,
            });
        }
        InputContent::Blocks(blocks) => blocks,
    };
    let mut text = Vec::new();
    let mut reasoning = Vec::new();
    let mut tool_calls = Vec::new();
    for block in blocks {
        match block {
            InputContentBlock::Text(value) => text.push(value.text.clone()),
            InputContentBlock::Thinking(value) => reasoning.push(value.thinking.clone()),
            InputContentBlock::ToolUse(value) => tool_calls.push(ToolCall {
                id: value.id.clone(),
                tool_type: "function".to_string(),
                function: FunctionCallResponse {
                    name: value.name.clone(),
                    arguments: Some(
                        serde_json::to_string(&value.input)
                            .map_err(|error| format!("failed to serialize tool input: {error}"))?,
                    ),
                },
            }),
            InputContentBlock::RedactedThinking(_)
            | InputContentBlock::Image(_)
            | InputContentBlock::Document(_)
            | InputContentBlock::ToolResult(_)
            | InputContentBlock::ServerToolUse(_)
            | InputContentBlock::SearchResult(_)
            | InputContentBlock::WebSearchToolResult(_)
            | InputContentBlock::ToolSearchToolResult(_)
            | InputContentBlock::ToolReference(_) => {
                return Err("unsupported block in a TITO Messages assistant message".to_string());
            }
        }
    }
    Ok(ChatMessage::Assistant {
        content: (!text.is_empty()).then(|| MessageContent::Text(text.join(""))),
        name: None,
        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
        reasoning_content: (!reasoning.is_empty()).then(|| reasoning.join("\n")),
    })
}

fn parallel_tool_calls(choice: Option<&messages::ToolChoice>) -> Option<bool> {
    let disabled = match choice {
        Some(
            messages::ToolChoice::Auto {
                disable_parallel_tool_use,
            }
            | messages::ToolChoice::Any {
                disable_parallel_tool_use,
            }
            | messages::ToolChoice::Tool {
                disable_parallel_tool_use,
                ..
            },
        ) => *disable_parallel_tool_use,
        Some(messages::ToolChoice::None) | None => None,
    };
    disabled.map(|disabled| !disabled)
}

pub(crate) fn messages_to_training_chat(
    request: &CreateMessageRequest,
) -> Result<ChatCompletionRequest, String> {
    let mut messages = Vec::new();
    let system = system_text(request.system.as_ref());
    if !system.is_empty() {
        messages.push(ChatMessage::System {
            content: MessageContent::Text(system),
            name: None,
        });
    }

    for message in &request.messages {
        match message.role {
            Role::User => messages.extend(user_messages(&message.content)?),
            Role::Assistant => messages.push(assistant_message(&message.content)?),
            Role::System => {
                let content = match &message.content {
                    InputContent::String(text) => text.clone(),
                    InputContent::Blocks(blocks) => blocks
                        .iter()
                        .filter_map(|block| match block {
                            InputContentBlock::Text(text) => Some(text.text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n"),
                };
                messages.push(ChatMessage::System {
                    content: MessageContent::Text(content),
                    name: None,
                });
            }
        }
    }

    let tools = request
        .tools
        .as_deref()
        .map(message_utils::extract_chat_tools)
        .filter(|tools| !tools.is_empty());
    let tool_choice = request
        .tool_choice
        .as_ref()
        .map(message_utils::convert_message_tool_choice);
    let thinking = request
        .thinking
        .as_ref()
        .map(|thinking| !matches!(thinking, messages::ThinkingConfig::Disabled));
    let chat_template_kwargs = thinking.map(|enabled| {
        HashMap::from([
            ("enable_thinking".to_string(), Value::Bool(enabled)),
            ("thinking".to_string(), Value::Bool(enabled)),
        ])
    });

    let mut chat_request = ChatCompletionRequest {
        messages,
        model: request.model.clone(),
        max_completion_tokens: Some(request.max_tokens),
        parallel_tool_calls: parallel_tool_calls(request.tool_choice.as_ref()),
        stop: request
            .stop_sequences
            .as_ref()
            .map(|stop| StringOrArray::Array(stop.clone())),
        temperature: request.temperature.map(|value| value as f32),
        tool_choice,
        tools,
        top_p: request.top_p.map(|value| value as f32),
        top_k: request.top_k.map(|value| value as i32),
        chat_template_kwargs,
        ..Default::default()
    };
    training::configure_canonical_turn(&mut chat_request);
    Ok(chat_request)
}

pub(crate) fn training_chat_to_message(
    response: &ChatCompletionResponse,
    request: &CreateMessageRequest,
) -> Result<Message, String> {
    let choice = response
        .choices
        .first()
        .ok_or_else(|| "canonical Chat response contains no choices".to_string())?;
    let mut content = Vec::new();
    if let Some(reasoning) = choice
        .message
        .reasoning_content
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        content.push(ContentBlock::Thinking {
            thinking: reasoning.to_string(),
            // Self-hosted models do not produce Anthropic's cryptographic
            // signature. The client treats this value as opaque and echoes it;
            // the inbound canonical adapter intentionally ignores it.
            signature: "smg".to_string(),
        });
    }
    if let Some(text) = choice
        .message
        .content
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        content.push(ContentBlock::Text {
            text: text.to_string(),
            citations: None,
        });
    }
    if let Some(tool_calls) = &choice.message.tool_calls {
        for tool_call in tool_calls {
            let input = tool_call
                .function
                .arguments
                .as_deref()
                .map(serde_json::from_str)
                .transpose()
                .map_err(|error| format!("invalid tool arguments from canonical Chat: {error}"))?
                .unwrap_or_else(|| Value::Object(Map::new()));
            content.push(ContentBlock::ToolUse {
                id: tool_call.id.clone(),
                name: tool_call.function.name.clone(),
                input,
            });
        }
    }

    let matched_stop = choice
        .matched_stop
        .as_ref()
        .and_then(Value::as_str)
        .map(String::from);
    let stop_reason = if choice.message.tool_calls.is_some()
        || choice.finish_reason.as_deref() == Some("tool_calls")
    {
        Some(StopReason::ToolUse)
    } else if matched_stop.is_some() {
        Some(StopReason::StopSequence)
    } else if choice.finish_reason.as_deref() == Some("length") {
        Some(StopReason::MaxTokens)
    } else {
        Some(StopReason::EndTurn)
    };
    let usage = response.usage.as_ref();

    if choice.logprobs.is_some() {
        warn!("TITO canonical Messages response unexpectedly exposed outbound logprobs");
    }

    Ok(Message {
        id: format!("msg_{}", response.id),
        message_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: request.model.clone(),
        stop_reason,
        stop_sequence: matched_stop,
        usage: messages::Usage {
            input_tokens: usage.map_or(0, |usage| usage.prompt_tokens),
            output_tokens: usage.map_or(0, |usage| usage.completion_tokens),
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
            cache_creation: None,
            server_tool_use: None,
            service_tier: None,
        },
    })
}

#[cfg(test)]
mod tests {
    use openai_protocol::{
        chat::{ChatChoice, ChatCompletionMessage},
        common::Usage,
        messages::{
            CustomTool, InputMessage, InputSchema, ThinkingConfig, ToolChoice, ToolUseBlock,
        },
    };
    use serde_json::json;

    use super::*;

    fn request() -> CreateMessageRequest {
        CreateMessageRequest {
            model: "actor".to_string(),
            messages: vec![
                InputMessage {
                    role: Role::User,
                    content: InputContent::String("inspect".to_string()),
                },
                InputMessage {
                    role: Role::Assistant,
                    content: InputContent::Blocks(vec![
                        InputContentBlock::Thinking(messages::ThinkingBlock {
                            thinking: "reason".to_string(),
                            signature: "wire-only".to_string(),
                        }),
                        InputContentBlock::ToolUse(ToolUseBlock {
                            id: "toolu_1".to_string(),
                            name: "Read".to_string(),
                            input: json!({"path": "README.md"}),
                            cache_control: None,
                        }),
                    ]),
                },
            ],
            max_tokens: 128,
            system: Some(SystemContent::String(
                "x-anthropic-billing-header: ignored\nBe useful".to_string(),
            )),
            thinking: Some(ThinkingConfig::Enabled {
                budget_tokens: 1024,
                display: None,
            }),
            tool_choice: Some(ToolChoice::Auto {
                disable_parallel_tool_use: Some(true),
            }),
            tools: Some(vec![messages::Tool::Custom(CustomTool {
                name: "Read".to_string(),
                tool_type: None,
                description: Some("Read a file".to_string()),
                input_schema: InputSchema {
                    schema_type: "object".to_string(),
                    properties: None,
                    required: None,
                    additional: HashMap::new(),
                },
                defer_loading: None,
                cache_control: None,
            })]),
            ..serde_json::from_value(json!({
                "model": "actor",
                "messages": [{"role": "user", "content": "placeholder"}],
                "max_tokens": 128
            }))
            .expect("valid request defaults")
        }
    }

    #[test]
    fn converts_messages_to_non_streaming_training_chat() {
        let chat = messages_to_training_chat(&request()).expect("conversion succeeds");
        assert!(!chat.stream);
        assert!(chat.logprobs);
        assert_eq!(chat.top_logprobs, Some(1));
        assert_eq!(chat.max_completion_tokens, Some(128));
        assert_eq!(chat.parallel_tool_calls, Some(false));
        assert!(matches!(
            chat.messages.first(),
            Some(ChatMessage::System {
                content: MessageContent::Text(text),
                ..
            }) if text == "Be useful"
        ));
        assert!(matches!(
            chat.messages.last(),
            Some(ChatMessage::Assistant {
                tool_calls: Some(calls),
                reasoning_content: Some(reasoning),
                ..
            }) if calls[0].function.name == "Read" && reasoning == "reason"
        ));
    }

    #[test]
    fn converts_training_chat_back_to_native_message() {
        let response = ChatCompletionResponse {
            id: "chatcmpl_1".to_string(),
            object: "chat.completion".to_string(),
            created: 1,
            model: "actor".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content: Some("done".to_string()),
                    tool_calls: Some(vec![ToolCall {
                        id: "call_1".to_string(),
                        tool_type: "function".to_string(),
                        function: FunctionCallResponse {
                            name: "Read".to_string(),
                            arguments: Some(r#"{"path":"README.md"}"#.to_string()),
                        },
                    }]),
                    reasoning_content: Some("check".to_string()),
                },
                logprobs: None,
                finish_reason: Some("tool_calls".to_string()),
                matched_stop: None,
                hidden_states: None,
                routed_experts: None,
            }],
            usage: Some(Usage::from_counts(11, 3)),
            system_fingerprint: None,
        };

        let message = training_chat_to_message(&response, &request()).expect("conversion succeeds");
        assert_eq!(message.stop_reason, Some(StopReason::ToolUse));
        assert_eq!(message.usage.input_tokens, 11);
        assert!(matches!(
            message.content.as_slice(),
            [
                ContentBlock::Thinking { .. },
                ContentBlock::Text { .. },
                ContentBlock::ToolUse { name, .. }
            ] if name == "Read"
        ));
    }
}
