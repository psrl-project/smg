use std::sync::Arc;

use base64::Engine as _;
use serde::{ser::SerializeStruct, Serialize};

/// Immutable sparse sampler rows. Cloning a TITO tree shares the entry buffers.
#[derive(Clone, Debug)]
pub struct SamplerDistribution {
    offsets: Arc<[u64]>,
    token_ids: Arc<[u8]>,
    log_probs: Arc<[u8]>,
    complete: Arc<[bool]>,
}

impl SamplerDistribution {
    /// Negative infinity witnesses that the captured head covers the finite support.
    ///
    /// The captured row is expected in descending log-probability order, so every finite entry must
    /// precede the sentinel; a finite entry after a `-inf` breaks the "support is a prefix" contract
    /// the compact trainer transport relies on and is rejected rather than silently kept.
    pub fn from_rows(rows: impl IntoIterator<Item = (Vec<u32>, Vec<f32>)>) -> Option<Self> {
        let mut offsets = vec![0];
        let mut token_ids = Vec::new();
        let mut log_probs = Vec::new();
        let mut complete = Vec::new();
        for (ids, values) in rows {
            if ids.is_empty() || ids.len() != values.len() {
                return None;
            }
            let mut full = false;
            for (id, value) in ids.into_iter().zip(values) {
                if value == f32::NEG_INFINITY {
                    full = true;
                } else if full || !value.is_finite() || value > 0.000001 || id > i32::MAX as u32 {
                    return None;
                } else {
                    token_ids.extend_from_slice(&id.to_le_bytes());
                    log_probs.extend_from_slice(&value.to_le_bytes());
                }
            }
            let end = (token_ids.len() / 4) as u64;
            if offsets.last() == Some(&end) {
                return None;
            }
            offsets.push(end);
            complete.push(full);
        }
        Some(Self {
            offsets: offsets.into(),
            token_ids: token_ids.into(),
            log_probs: log_probs.into(),
            complete: complete.into(),
        })
    }
}

impl Serialize for SamplerDistribution {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("SamplerDistribution", 4)?;
        state.serialize_field("offsets", &*self.offsets)?;
        state.serialize_field(
            "token_ids",
            &base64::engine::general_purpose::STANDARD.encode(&self.token_ids),
        )?;
        state.serialize_field(
            "log_probs",
            &base64::engine::general_purpose::STANDARD.encode(&self.log_probs),
        )?;
        state.serialize_field("complete", &*self.complete)?;
        state.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn support_witness_is_preserved_without_serializing_infinity() {
        let distribution = SamplerDistribution::from_rows([
            (
                vec![2, 4, 9],
                vec![0.25_f32.ln(), 0.75_f32.ln(), f32::NEG_INFINITY],
            ),
            (vec![3], vec![-0.000001]),
        ])
        .expect("valid distribution");
        let value = serde_json::to_value(&distribution).expect("serializable distribution");
        assert_eq!(value["offsets"], serde_json::json!([0, 2, 3]));
        assert_eq!(value["complete"], serde_json::json!([true, false]));
        assert_eq!(distribution.token_ids.len(), 12);
        assert_eq!(distribution.log_probs.len(), 12);
        let cloned = distribution.clone();
        assert!(Arc::ptr_eq(&cloned.token_ids, &distribution.token_ids));
    }

    #[test]
    fn rejects_misaligned_or_invalid_rows() {
        assert!(SamplerDistribution::from_rows([(vec![1], vec![])]).is_none());
        assert!(SamplerDistribution::from_rows([(vec![1], vec![f32::NAN])]).is_none());
        assert!(SamplerDistribution::from_rows([(vec![1], vec![f32::NEG_INFINITY])]).is_none());
    }

    #[test]
    fn rejects_finite_entry_after_support_sentinel() {
        // A finite log-prob after the sentinel breaks the "support is a prefix" contract the compact
        // trainer transport relies on, so the row must be rejected rather than silently kept.
        assert!(
            SamplerDistribution::from_rows([(
                vec![2, 9, 4],
                vec![0.25_f32.ln(), f32::NEG_INFINITY, 0.75_f32.ln()],
            )])
            .is_none()
        );
    }
}
