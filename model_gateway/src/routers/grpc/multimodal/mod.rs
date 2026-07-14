//! Multimodal processing integration for gRPC pipeline (chat + messages).
//!
//! Bridges the `llm-multimodal` crate with the gRPC router pipeline, split by
//! processing phase:
//!
//! - [`detect`]: find modalities and extract content parts from chat/messages.
//! - [`config`]: model config-file registry and per-router component bundle.
//! - [`process`]: fetch media → preprocess → expand placeholder tokens →
//!   build the lightweight [`MultimodalIntermediate`].
//! - [`assemble`]: turn the intermediate into backend-specific `MultimodalData`
//!   once the target backend is known (after worker selection).
//! - [`serialize`]: tensor byte/dtype serialization used by assembly.
//! - [`transport`]: SHM-vs-inline transport resolution and `/dev/shm`
//!   namespace verification.

use std::{
    collections::HashSet,
    sync::{Arc, OnceLock},
};

use base64::Engine as _;
use llm_multimodal::{
    AudioClip, EncoderFieldLayouts, FieldLayout, ImageFrame, Modality, ModelSpecificValue,
    PlaceholderRange, PreprocessedEncoderInputs, VideoClip,
};
use ndarray::ArrayD;
use openai_protocol::generate::{PreprocessedMultimodalInputs, SerializedMultimodalTensor};

mod assemble;
mod capability;
mod config;
mod detect;
mod pixel_cache;
mod plan;
mod process;
mod serialize;
mod transport;

pub(crate) use assemble::{
    assemble_multimodal_data, assemble_multimodal_data_after_encode,
    assemble_tokenspeed_for_encode, encode_routing_hashes,
};
pub(crate) use capability::ensure_backend_supports_modalities;
pub(crate) use config::{
    load_preprocessor_config_file, load_video_preprocessor_config, MultimodalComponents,
    MultimodalConfigRegistry, MultimodalModelConfig,
};
pub(crate) use detect::{media_plan_chat, media_plan_generate, media_plan_messages};
pub(crate) use plan::{
    prepare_placeholder_tokens, validate_rendered_media_anchors, PlaceholderTokens,
};
pub(crate) use process::{
    process_multimodal_plan, process_multimodal_plan_preexpanded,
    process_multimodal_plan_python_preprocessed,
};
pub(crate) use transport::{init_mm_transport_defaults, mm_rdma_exporter};

/// Whether verbose multimodal timing logs are enabled via `SMG_LOG_MM_TIMING`.
/// Read from the environment once and cached; the flag is not expected to change
/// at runtime, and this is called on every multimodal request.
fn log_mm_timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SMG_LOG_MM_TIMING")
            .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

/// Output of the multimodal processing pipeline.
pub(crate) struct MultimodalOutput {
    /// Token IDs with placeholder tokens expanded to the correct count per media item.
    pub expanded_token_ids: Vec<u32>,
    /// Lightweight intermediate holding preprocessing results.
    /// Assembled into backend-specific `MultimodalData` in request_building.
    pub intermediate: MultimodalIntermediate,
}

/// Lightweight intermediate from the preparation stage.
///
/// Holds all preprocessing results without serializing tensors to bytes.
/// The assembly stage converts this into a backend-specific `MultimodalData`
/// variant once the target backend is known (after worker selection).
#[derive(Debug, Clone)]
pub(crate) struct MultimodalIntermediate {
    /// Independently preprocessed modality batches sharing one expanded prompt.
    /// A single-modality request is represented by a one-element vector.
    batches: Vec<PrecomputedMultimodalIntermediate>,
}

impl MultimodalIntermediate {
    pub(crate) fn try_new(batches: Vec<PrecomputedMultimodalIntermediate>) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !batches.is_empty(),
            "multimodal intermediate requires at least one batch"
        );
        let mut modalities = HashSet::with_capacity(batches.len());
        for batch in &batches {
            let modality = batch.media.modality();
            anyhow::ensure!(
                modalities.insert(modality),
                "multimodal intermediate contains duplicate {modality} batches"
            );
            anyhow::ensure!(
                batch.media.len() > 0,
                "multimodal intermediate contains an empty {modality} batch"
            );
        }
        Ok(Self { batches })
    }

    pub(crate) fn batches(&self) -> &[PrecomputedMultimodalIntermediate] {
        &self.batches
    }

    pub(crate) fn into_batches(self) -> Vec<PrecomputedMultimodalIntermediate> {
        self.batches
    }
}

/// Raw media for one preprocessed batch.
///
/// Encoding the modality in the enum prevents contradictory states such as an
/// audio batch carrying images or an image batch carrying both images and
/// videos.
#[derive(Debug, Clone)]
pub(crate) enum MediaBatch {
    Images(Vec<Arc<ImageFrame>>),
    Audios(Vec<Arc<AudioClip>>),
    Videos(Vec<Arc<VideoClip>>),
}

impl MediaBatch {
    pub(crate) fn modality(&self) -> Modality {
        match self {
            Self::Images(_) => Modality::Image,
            Self::Audios(_) => Modality::Audio,
            Self::Videos(_) => Modality::Video,
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Images(items) => items.len(),
            Self::Audios(items) => items.len(),
            Self::Videos(items) => items.len(),
        }
    }
}

/// Explicit association between one media item and its expanded prompt span.
#[derive(Debug, Clone)]
pub(crate) struct PromptBinding {
    /// Index of the media/preprocessed item within its modality batch.
    pub item_index: usize,
    /// Position of this media item among all modalities in the rendered prompt.
    pub prompt_ordinal: usize,
    /// Full replacement span, including structural tokens.
    pub structural: PlaceholderRange,
    /// Patch-only spans within `structural`.
    pub patches: Vec<PlaceholderRange>,
}

#[derive(Debug, Clone)]
pub(crate) struct PrecomputedMultimodalIntermediate {
    /// Preprocessed encoder input and model-specific tensors (not yet serialized).
    pub preprocessed: PreprocessedEncoderInputs,
    /// Raw media whose variant determines this batch's modality.
    pub media: MediaBatch,
    /// Exact media-to-prompt associations for this batch.
    pub bindings: Vec<PromptBinding>,
    /// Placeholder token ID from model config for the active modality.
    pub placeholder_token_id: Option<u32>,
    /// Primary encoder input and model-specific side-tensor layouts.
    pub field_layouts: EncoderFieldLayouts,
    /// Tensor keys that should remain on CPU (vLLM `keep_on_cpu` hint).
    pub keep_on_cpu_keys: Vec<String>,
}

/// Replace Rust-produced tensors with caller-produced HF tensors while keeping
/// the fetched raw media (and therefore stable content hashes) authoritative.
pub(crate) fn apply_python_preprocessed_inputs(
    intermediate: MultimodalIntermediate,
    payload: &PreprocessedMultimodalInputs,
) -> anyhow::Result<MultimodalIntermediate> {
    let mut batches = intermediate.into_batches();
    anyhow::ensure!(
        batches.len() == 1,
        "Python-preprocessed /generate supports one image batch"
    );
    let batch = &mut batches[0];
    anyhow::ensure!(
        matches!(batch.media, MediaBatch::Images(_)),
        "Python-preprocessed inputs require images"
    );
    anyhow::ensure!(
        payload.mm_placeholders.len() == batch.media.len(),
        "Python placeholders ({}) do not match image count ({})",
        payload.mm_placeholders.len(),
        batch.media.len()
    );

    let pixels = decode_f32_tensor(&payload.pixel_values)?;
    let mut model_specific = std::collections::HashMap::new();
    for (key, tensor) in &payload.model_specific_tensors {
        let value = match tensor.dtype.as_str() {
            "float32" => ModelSpecificValue::Tensor {
                data: decode_f32_values(tensor)?,
                shape: tensor.shape.clone(),
            },
            "int64" => ModelSpecificValue::IntTensor {
                data: decode_i64_values(tensor)?,
                shape: tensor.shape.clone(),
            },
            other => anyhow::bail!("Unsupported Python tensor dtype {other} for {key}"),
        };
        model_specific.insert(key.clone(), value);
    }
    let item_sizes = batch.preprocessed.item_sizes.clone();
    batch.preprocessed = PreprocessedEncoderInputs {
        encoder_input: pixels,
        feature_token_counts: payload
            .mm_placeholders
            .iter()
            .map(|(_, len)| *len as usize)
            .collect(),
        item_sizes,
        model_specific,
    };
    batch.bindings = payload
        .mm_placeholders
        .iter()
        .enumerate()
        .map(|(item_index, &(offset, length))| {
            let range = PlaceholderRange {
                offset: offset as usize,
                length: length as usize,
            };
            PromptBinding {
                item_index,
                prompt_ordinal: item_index,
                structural: range.clone(),
                patches: vec![range],
            }
        })
        .collect();
    let encoder_input_layout = payload
        .flat_keys
        .get("pixel_values")
        .map(|sizes_key| FieldLayout::flat(sizes_key.clone()))
        .unwrap_or(FieldLayout::Batched);
    let mut model_layouts = std::collections::HashMap::new();
    for key in &payload.batched_keys {
        if key != "pixel_values" {
            model_layouts.insert(key.clone(), FieldLayout::Batched);
        }
    }
    for (key, sizes_key) in &payload.flat_keys {
        if key != "pixel_values" {
            model_layouts.insert(key.clone(), FieldLayout::flat(sizes_key.clone()));
        }
    }
    batch.field_layouts = EncoderFieldLayouts::new(encoder_input_layout, model_layouts);
    batch.keep_on_cpu_keys.clone_from(&payload.keep_on_cpu_keys);
    MultimodalIntermediate::try_new(batches)
}

fn decode_bytes(tensor: &SerializedMultimodalTensor) -> anyhow::Result<Vec<u8>> {
    base64::engine::general_purpose::STANDARD
        .decode(&tensor.data)
        .map_err(|error| anyhow::anyhow!("Invalid base64 tensor: {error}"))
}

fn element_count(shape: &[usize]) -> anyhow::Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, value| acc.checked_mul(*value))
        .ok_or_else(|| anyhow::anyhow!("Tensor shape overflows usize"))
}

fn expected_bytes(shape: &[usize], element_size: usize) -> anyhow::Result<usize> {
    element_count(shape)?
        .checked_mul(element_size)
        .ok_or_else(|| anyhow::anyhow!("Tensor byte length overflows usize"))
}

fn decode_f32_values(tensor: &SerializedMultimodalTensor) -> anyhow::Result<Vec<f32>> {
    anyhow::ensure!(tensor.dtype == "float32", "pixel_values must use float32");
    let bytes = decode_bytes(tensor)?;
    anyhow::ensure!(
        bytes.len() == expected_bytes(&tensor.shape, 4)?,
        "float32 tensor byte length mismatch"
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn decode_i64_values(tensor: &SerializedMultimodalTensor) -> anyhow::Result<Vec<i64>> {
    let bytes = decode_bytes(tensor)?;
    anyhow::ensure!(
        bytes.len() == expected_bytes(&tensor.shape, 8)?,
        "int64 tensor byte length mismatch"
    );
    Ok(bytes
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
        .collect())
}

fn decode_f32_tensor(tensor: &SerializedMultimodalTensor) -> anyhow::Result<ArrayD<f32>> {
    ArrayD::from_shape_vec(tensor.shape.clone(), decode_f32_values(tensor)?)
        .map_err(|error| anyhow::anyhow!("Invalid pixel_values shape: {error}"))
}
