use std::collections::{BTreeMap, HashMap};

use openai_protocol::chat::ChatMessage;
use serde_json::Value;

/// Hash type for content-addressed prefix tree nodes
pub type PrefixHash = [u8; 32];

/// Incremental hasher used by the prefix-lookup path.
pub type PrefixHasher = blake3::Hasher;

/// Rendering inputs that affect chat-template tokenization in addition to messages.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RenderContext {
    pub tools: Option<Vec<Value>>,
    pub template_kwargs: Option<HashMap<String, Value>>,
    pub image_placeholder: Option<String>,
    pub video_placeholder: Option<String>,
    pub audio_placeholder: Option<String>,
}

impl RenderContext {
    pub fn new(tools: Option<Vec<Value>>, template_kwargs: Option<HashMap<String, Value>>) -> Self {
        Self {
            tools,
            template_kwargs,
            image_placeholder: None,
            video_placeholder: None,
            audio_placeholder: None,
        }
    }

    /// Convenience constructor that also captures the image placeholder.
    pub fn with_image_placeholder(
        tools: Option<Vec<Value>>,
        template_kwargs: Option<HashMap<String, Value>>,
        image_placeholder: Option<String>,
    ) -> Self {
        Self {
            tools,
            template_kwargs,
            image_placeholder,
            video_placeholder: None,
            audio_placeholder: None,
        }
    }

    pub fn tools_ref(&self) -> Option<&[Value]> {
        self.tools.as_deref()
    }

    pub fn template_kwargs_ref(&self) -> Option<&HashMap<String, Value>> {
        self.template_kwargs.as_ref()
    }

    pub fn image_placeholder_ref(&self) -> Option<&str> {
        self.image_placeholder.as_deref()
    }

    pub fn with_media_placeholders(
        mut self,
        image: Option<String>,
        video: Option<String>,
        audio: Option<String>,
    ) -> Self {
        self.image_placeholder = image;
        self.video_placeholder = video;
        self.audio_placeholder = audio;
        self
    }

    pub fn placeholder_for_part_type(&self, part_type: &str) -> Option<&str> {
        match part_type {
            "image_url" | "image" | "input_image" => self.image_placeholder.as_deref(),
            "video_url" | "video" => self.video_placeholder.as_deref(),
            "audio_url" | "audio" | "input_audio" => self.audio_placeholder.as_deref(),
            _ => None,
        }
    }
}

/// Hash a slice of messages and the rendering context using Blake3.
pub fn hash_messages_with_context(messages: &[ChatMessage], context: &RenderContext) -> PrefixHash {
    let mut hasher = blake3::Hasher::new();
    hash_render_context_into(&mut hasher, context);
    for msg in messages {
        hash_message_into(&mut hasher, msg);
    }
    *hasher.finalize().as_bytes()
}

pub fn initialize_context_hasher(context: &RenderContext) -> blake3::Hasher {
    let mut hasher = blake3::Hasher::new();
    hash_render_context_into(&mut hasher, context);
    hasher
}

#[inline]
pub fn finalize_hash(hasher: &PrefixHasher) -> PrefixHash {
    *hasher.finalize().as_bytes()
}

fn hash_render_context_into(hasher: &mut blake3::Hasher, context: &RenderContext) {
    hasher.update(b"tito-render-context-v1\x00");

    hasher.update(b"tools\x00");
    match &context.tools {
        Some(tools) => {
            hasher.update(tools.len().to_string().as_bytes());
            hasher.update(b"\x00");
            for tool in tools {
                let canonical = canonicalize_value(tool.clone());
                hasher.update(canonical.as_bytes());
                hasher.update(b"\x00");
            }
        }
        None => {
            hasher.update(b"none\x00");
        }
    }
    for (name, placeholder) in [
        ("image", &context.image_placeholder),
        ("video", &context.video_placeholder),
        ("audio", &context.audio_placeholder),
    ] {
        hasher.update(name.as_bytes());
        hasher.update(b"_placeholder\x00");
        hasher.update(placeholder.as_deref().unwrap_or("none").as_bytes());
        hasher.update(b"\x00");
    }

    hasher.update(b"template_kwargs\x00");
    match &context.template_kwargs {
        Some(kwargs) => {
            let sorted: BTreeMap<_, _> = kwargs.iter().collect();
            for (key, value) in sorted {
                hasher.update(key.as_bytes());
                hasher.update(b"\x00");
                let canonical = canonicalize_value(value.clone());
                hasher.update(canonical.as_bytes());
                hasher.update(b"\x00");
            }
        }
        None => {
            hasher.update(b"none\x00");
        }
    }
    hasher.update(b"\x01");
}

/// Hash a slice of messages using Blake3.
/// Normalizes: content None → "", tool_calls None → [], tool_call.function.arguments JSON sorted keys.
pub fn hash_messages(messages: &[ChatMessage]) -> PrefixHash {
    let mut hasher = blake3::Hasher::new();
    for msg in messages {
        hash_message_into(&mut hasher, msg);
    }
    *hasher.finalize().as_bytes()
}

pub fn hash_message_into(hasher: &mut blake3::Hasher, msg: &ChatMessage) {
    match msg {
        ChatMessage::System { content, .. } => {
            hasher.update(b"system\x00");
            hash_message_content(hasher, Some(content));
            hasher.update(b"\x00"); // reasoning_content (none)
                                    // no tool_calls
        }
        ChatMessage::User { content, .. } => {
            hasher.update(b"user\x00");
            hash_message_content(hasher, Some(content));
            hasher.update(b"\x00");
        }
        ChatMessage::Assistant {
            content,
            tool_calls,
            reasoning_content,
            ..
        } => {
            hasher.update(b"assistant\x00");
            // content may be None for tool-call-only assistant messages
            match content {
                Some(c) => hash_message_content(hasher, Some(c)),
                None => {
                    hasher.update(b"\x00");
                }
            }
            // reasoning_content
            let reasoning = reasoning_content.as_deref().unwrap_or("");
            hasher.update(reasoning.as_bytes());
            hasher.update(b"\x00");
            // tool_calls
            let tool_calls_slice = tool_calls.as_deref().unwrap_or(&[]);
            for tc in tool_calls_slice {
                hasher.update(tc.id.as_bytes());
                hasher.update(b"\x00");
                hasher.update(tc.tool_type.as_bytes());
                hasher.update(b"\x00");
                hasher.update(tc.function.name.as_bytes());
                hasher.update(b"\x00");
                // Canonicalize JSON arguments
                let args = tc.function.arguments.as_deref().unwrap_or("{}");
                let canonical = canonicalize_json(args);
                hasher.update(canonical.as_bytes());
                hasher.update(b"\x00");
            }
        }
        ChatMessage::Tool {
            content,
            tool_call_id,
        } => {
            hasher.update(b"tool\x00");
            hash_message_content(hasher, Some(content));
            hasher.update(b"\x00");
            hasher.update(tool_call_id.as_bytes());
            hasher.update(b"\x00");
        }
        ChatMessage::Function { content, name } => {
            hasher.update(b"function\x00");
            hasher.update(content.as_bytes());
            hasher.update(b"\x00");
            hasher.update(name.as_bytes());
            hasher.update(b"\x00");
        }
        ChatMessage::Developer { content, .. } => {
            hasher.update(b"developer\x00");
            hash_message_content(hasher, Some(content));
            hasher.update(b"\x00");
        }
    }
    hasher.update(b"\x01"); // message separator
}

fn hash_message_content(
    hasher: &mut blake3::Hasher,
    content: Option<&openai_protocol::chat::MessageContent>,
) {
    use openai_protocol::{chat::MessageContent, common::ContentPart};
    match content {
        None => {
            hasher.update(b"\x00");
        }
        Some(MessageContent::Text(s)) => {
            hasher.update(s.as_bytes());
            hasher.update(b"\x00");
        }
        Some(MessageContent::Parts(parts)) => {
            for part in parts {
                match part {
                    ContentPart::Text { text } => {
                        hasher.update(text.as_bytes());
                        hasher.update(b"\x00");
                    }
                    _ => {
                        // non-text parts: hash their JSON representation
                        if let Ok(v) = serde_json::to_string(part) {
                            hasher.update(v.as_bytes());
                            hasher.update(b"\x00");
                        }
                    }
                }
            }
        }
    }
}

/// Render a one-line, byte-level digest of every assistant message in `messages`.
///
/// This is a debugging aid for diagnosing TITO prefix-hash mismatches across turns:
/// the store side (after generation) and the lookup side (`find_prefix`) both log
/// this string, so any per-field divergence — content length/sha, tool-call ids,
/// `reasoning_content`, `name` presence — becomes visible at a glance when grepping
/// a single `session_id`.
pub fn assistants_diagnostic_summary(messages: &[ChatMessage]) -> String {
    use openai_protocol::chat::MessageContent;

    let mut out = String::with_capacity(64);
    out.push('[');
    let mut first = true;
    for (i, msg) in messages.iter().enumerate() {
        let ChatMessage::Assistant {
            content,
            name,
            tool_calls,
            reasoning_content,
        } = msg
        else {
            continue;
        };

        if first {
            first = false;
        } else {
            out.push_str(" | ");
        }

        // Position k = i+1 (the prefix length once this assistant has been hashed).
        out.push_str(&format!(
            "k={} name={}",
            i + 1,
            if name.is_some() { "Some" } else { "None" }
        ));

        match content {
            None => out.push_str(" content=None"),
            Some(MessageContent::Text(s)) => out.push_str(&format!(
                " content=Text({},{})",
                s.len(),
                short_sha(s.as_bytes())
            )),
            Some(MessageContent::Parts(parts)) => {
                out.push_str(&format!(" content=Parts({})", parts.len()))
            }
        }

        match reasoning_content {
            None => out.push_str(" rc=None"),
            Some(s) => out.push_str(&format!(
                " rc=Some({},{})",
                s.len(),
                short_sha(s.as_bytes())
            )),
        }

        match tool_calls {
            None => out.push_str(" tc=None"),
            Some(tcs) => {
                out.push_str(&format!(" tc=[n={}", tcs.len()));
                for (j, tc) in tcs.iter().enumerate() {
                    let args = tc.function.arguments.as_deref().unwrap_or("{}");
                    out.push_str(&format!(
                        " {}:(id={},type={},name={},args=({},{}))",
                        j,
                        tc.id,
                        tc.tool_type,
                        tc.function.name,
                        args.len(),
                        short_sha(args.as_bytes())
                    ));
                }
                out.push(']');
            }
        }
    }
    out.push(']');
    out
}

fn short_sha(bytes: &[u8]) -> String {
    let h = blake3::hash(bytes);
    let prefix = &h.as_bytes()[..4];
    format!(
        "{:02x}{:02x}{:02x}{:02x}",
        prefix[0], prefix[1], prefix[2], prefix[3]
    )
}

fn canonicalize_json(json_str: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(json_str) {
        Ok(v) => serde_json::to_string(&sort_keys(v)).unwrap_or_else(|_| json_str.to_owned()),
        Err(_) => json_str.to_owned(),
    }
}

fn canonicalize_value(value: Value) -> String {
    serde_json::to_string(&sort_keys(value)).unwrap_or_default()
}

fn sort_keys(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let sorted: serde_json::Map<_, _> = map
                .into_iter()
                .map(|(k, v)| (k, sort_keys(v)))
                .collect::<std::collections::BTreeMap<_, _>>()
                .into_iter()
                .collect();
            serde_json::Value::Object(sorted)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.into_iter().map(sort_keys).collect())
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::chat::{ChatMessage, MessageContent};

    use super::*;

    fn user_msg(content: &str) -> ChatMessage {
        ChatMessage::User {
            content: MessageContent::Text(content.to_string()),
            name: None,
        }
    }

    fn assistant_msg(content: &str) -> ChatMessage {
        ChatMessage::Assistant {
            content: Some(MessageContent::Text(content.to_string())),
            name: None,
            tool_calls: None,
            reasoning_content: None,
        }
    }

    #[test]
    fn same_messages_produce_same_hash() {
        let msgs = vec![user_msg("hello"), assistant_msg("hi")];
        assert_eq!(hash_messages(&msgs), hash_messages(&msgs));
    }

    #[test]
    fn different_messages_produce_different_hash() {
        let a = vec![user_msg("hello")];
        let b = vec![user_msg("world")];
        assert_ne!(hash_messages(&a), hash_messages(&b));
    }

    #[test]
    fn tool_call_args_canonicalized() {
        use openai_protocol::common::{FunctionCallResponse, ToolCall};
        let mk_tool_msg = |args: &str| ChatMessage::Assistant {
            content: None,
            name: None,
            reasoning_content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_1".to_string(),
                tool_type: "function".to_string(),
                function: FunctionCallResponse {
                    name: "my_fn".to_string(),
                    arguments: Some(args.to_string()),
                },
            }]),
        };
        let m1 = vec![mk_tool_msg(r#"{"b":1,"a":2}"#)];
        let m2 = vec![mk_tool_msg(r#"{"a":2,"b":1}"#)];
        assert_eq!(hash_messages(&m1), hash_messages(&m2));
    }

    #[test]
    fn different_render_contexts_produce_different_hashes() {
        let msgs = vec![user_msg("hello"), assistant_msg("hi")];
        let ctx_a = RenderContext::new(Some(vec![serde_json::json!({"name":"tool_a"})]), None);
        let ctx_b = RenderContext::new(Some(vec![serde_json::json!({"name":"tool_b"})]), None);
        assert_ne!(
            hash_messages_with_context(&msgs, &ctx_a),
            hash_messages_with_context(&msgs, &ctx_b)
        );
    }
}
