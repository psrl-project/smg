//! Harness-specific canonicalization of assistant tool-call inputs during
//! prefix hashing.
//!
//! ## Problem
//!
//! Agent harnesses normalize assistant `tool_use` inputs as they receive them
//! (schema-parse with defaults, fixed parameter order, dropping a redundant
//! `cd <cwd> && ` prefix on Bash commands, ...).  On resume/replay the client
//! re-sends those *canonical* messages, while the session originally stored
//! hashes over the *raw* model output.  Even after the generic hash-side
//! canonicalization in [`crate::normalizer`] (JSON key sorting and per-line
//! trailing whitespace tolerance), value-level differences such as a redundant
//! `cd /testbed && ` prefix or an added `replace_all` default field make the
//! prefix lookup miss, so SMG forks a new phantom trajectory for a turn that is
//! semantically identical.
//!
//! ## Design
//!
//! The canonicalization performed here is **harness-specific** and value-level:
//! it must be safe to apply symmetrically to the message a client re-sends and
//! to the assistant message recorded from the model's own generation.  It only
//! rewrites structure that is provably inert (`cd <cwd> && ` when `<cwd>` is the
//! harness working directory) or that the harness itself materializes as a
//! default (`replace_all`).  Tool argument content — in particular
//! `old_string`/`new_string` — is never touched.
//!
//! The trait is the extension point for future harnesses: implement
//! [`ToolInputCanonicalizer`] and register it in [`build_tool_canonicalizer`].

use std::sync::Arc;

use serde_json::Value;

/// Canonicalizes one assistant tool-call input *value* before hashing.
pub trait ToolInputCanonicalizer: Send + Sync {
    /// Stable name used for logs and config diagnostics.
    fn name(&self) -> &str;

    /// Return the canonical input for a tool named `tool_name`.
    ///
    /// `input` is the parsed JSON of the tool call's `arguments`.  Callers are
    /// responsible for applying generic hash canonicalization afterwards; this
    /// trait only performs harness-specific value rewrites.
    fn canonicalize(&self, tool_name: &str, input: Value) -> Value;
}

/// Identity canonicalizer: no harness-specific rewrite (default).
#[derive(Debug, Default)]
pub struct NoopToolCanonicalizer;

impl ToolInputCanonicalizer for NoopToolCanonicalizer {
    fn name(&self) -> &str {
        "none"
    }

    fn canonicalize(&self, _tool_name: &str, input: Value) -> Value {
        input
    }
}

/// Claude Code value-level canonicalization (CLI >= 2.1.x).
///
/// Mirrors the parts of Claude Code's `normalizeToolInput` that are safe to
/// apply at hash time:
///
/// - Bash: strip a leading `cd {workdir} && ` from `command`.  Claude Code runs
///   every Bash call from the session working directory and removes that exact
///   redundant prefix when it re-serializes the tool input; a model-generated
///   command that keeps the prefix is semantically identical.
/// - Edit: materialize the `replace_all` default (`false`) when the field is
///   absent.  Claude Code's Edit schema declares `replace_all` optional with
///   default `false`, and its parsed/re-serialized inputs always carry it.
///
/// Argument content is never modified.
#[derive(Debug)]
pub struct ClaudeCodeToolCanonicalizer {
    workdir: String,
}

impl ClaudeCodeToolCanonicalizer {
    pub fn new(workdir: impl Into<String>) -> Self {
        Self {
            workdir: workdir.into(),
        }
    }
}

impl ToolInputCanonicalizer for ClaudeCodeToolCanonicalizer {
    fn name(&self) -> &str {
        "claude_code"
    }

    fn canonicalize(&self, tool_name: &str, input: Value) -> Value {
        let Value::Object(mut map) = input else {
            return input;
        };
        match tool_name.to_ascii_lowercase().as_str() {
            "bash" => {
                if let Some(Value::String(command)) = map.get_mut("command") {
                    let prefix = format!("cd {} && ", self.workdir);
                    if let Some(rest) = command.strip_prefix(&prefix) {
                        *command = rest.to_string();
                    }
                }
            }
            "edit" => {
                if !map.contains_key("replace_all") {
                    map.insert("replace_all".to_string(), Value::Bool(false));
                }
            }
            _ => {}
        }
        Value::Object(map)
    }
}

/// Build the canonicalizer selected by a configuration string.
///
/// `mode` accepts `none` (default, identity) or `claude_code`.  Unknown modes
/// fall back to the identity canonicalizer and are reported to the caller via
/// logging inside the store, never by panicking.
pub fn build_tool_canonicalizer(
    mode: Option<&str>,
    workdir: Option<&str>,
) -> Arc<dyn ToolInputCanonicalizer> {
    let normalized = mode.map(str::trim).unwrap_or("").to_ascii_lowercase();
    match normalized.as_str() {
        "claude_code" => Arc::new(ClaudeCodeToolCanonicalizer::new(
            workdir.unwrap_or("/testbed"),
        )),
        _ => Arc::new(NoopToolCanonicalizer),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn claude_code_strips_redundant_cd_prefix() {
        let c = ClaudeCodeToolCanonicalizer::new("/testbed");
        let input = json!({"command": "cd /testbed && python -m pytest -v", "timeout": 1});
        let out = c.canonicalize("Bash", input);
        assert_eq!(out["command"], "python -m pytest -v");
        assert_eq!(out["timeout"], 1);
    }

    #[test]
    fn claude_code_preserves_other_cd_prefixes() {
        let c = ClaudeCodeToolCanonicalizer::new("/testbed");
        let input = json!({"command": "cd /workspace && make"});
        let out = c.canonicalize("Bash", input);
        assert_eq!(out["command"], "cd /workspace && make");
    }

    #[test]
    fn claude_code_materializes_edit_replace_all_default() {
        let c = ClaudeCodeToolCanonicalizer::new("/testbed");
        let input = json!({
            "file_path": "a.py",
            "old_string": "x",
            "new_string": "y",
        });
        let out = c.canonicalize("Edit", input);
        assert_eq!(out["replace_all"], false);
        assert_eq!(out["old_string"], "x");
        assert_eq!(out["new_string"], "y");
        // Existing value is preserved.
        let input2 = json!({"file_path": "a.py", "replace_all": true});
        assert_eq!(c.canonicalize("Edit", input2)["replace_all"], true);
    }

    #[test]
    fn noop_keeps_input_untouched() {
        let input = json!({"command": "cd /testbed && make", "replace_all": false});
        assert_eq!(NoopToolCanonicalizer.canonicalize("Bash", input.clone()), input);
    }

    #[test]
    fn builder_selects_known_harnesses_only() {
        assert_eq!(build_tool_canonicalizer(Some("none"), None).name(), "none");
        assert_eq!(build_tool_canonicalizer(None, None).name(), "none");
        assert_eq!(
            build_tool_canonicalizer(Some("claude_code"), Some("/repo")).name(),
            "claude_code"
        );
        assert_eq!(
            build_tool_canonicalizer(Some("unknown"), None).name(),
            "none"
        );
    }
}
