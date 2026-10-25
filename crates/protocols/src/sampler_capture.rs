use std::{error::Error, fmt};

use serde::{de::Error as _, Deserialize, Deserializer, Serialize};

/// Absolute protocol ceiling for one token's captured sampler head.
///
/// Backends may enforce a lower model-specific limit. This ceiling rejects values that cannot be a
/// deliberate vocabulary-sized capture before they reach allocation or gRPC conversion paths.
pub const MAX_SAMPLER_CAPTURE_WIDTH: u32 = 1 << 20;

/// Validated width of the sampler head captured for each generated token.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, schemars::JsonSchema,
)]
#[serde(transparent)]
#[schemars(transparent)]
pub struct SamplerCaptureWidth(u32);

impl SamplerCaptureWidth {
    pub const fn get(self) -> u32 {
        self.0
    }

    pub const fn is_enabled(self) -> bool {
        self.0 > 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidSamplerCaptureWidth(u32);

impl fmt::Display for InvalidSamplerCaptureWidth {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "sampler_top_logprobs must be at most {MAX_SAMPLER_CAPTURE_WIDTH}, got {}",
            self.0
        )
    }
}

impl Error for InvalidSamplerCaptureWidth {}

impl TryFrom<u32> for SamplerCaptureWidth {
    type Error = InvalidSamplerCaptureWidth;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        if value <= MAX_SAMPLER_CAPTURE_WIDTH {
            Ok(Self(value))
        } else {
            Err(InvalidSamplerCaptureWidth(value))
        }
    }
}

impl From<SamplerCaptureWidth> for u32 {
    fn from(value: SamplerCaptureWidth) -> Self {
        value.0
    }
}

impl From<SamplerCaptureWidth> for usize {
    fn from(value: SamplerCaptureWidth) -> Self {
        value.0 as Self
    }
}

impl From<SamplerCaptureWidth> for i32 {
    fn from(value: SamplerCaptureWidth) -> Self {
        value.0 as Self
    }
}

impl<'de> Deserialize<'de> for SamplerCaptureWidth {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u32::deserialize(deserializer)?;
        Self::try_from(value).map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_rejects_width_above_protocol_ceiling() {
        let valid = serde_json::from_str::<SamplerCaptureWidth>("1048576").unwrap();
        assert_eq!(valid.get(), MAX_SAMPLER_CAPTURE_WIDTH);
        assert!(serde_json::from_str::<SamplerCaptureWidth>("1048577").is_err());
    }
}
