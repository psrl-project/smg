use std::{collections::HashMap, sync::Arc};

use tokio::task::JoinHandle;

use super::{
    error::{MultiModalError, MultiModalResult},
    media::{ImageFetchConfig, MediaConnector, MediaSource, VideoFetchConfig},
    types::{
        ImageDetail, MediaContentPart, Modality, MultiModalData, MultiModalUUIDs, TrackedMedia,
    },
};

type PendingTask = JoinHandle<MultiModalResult<TrackedMedia>>;

/// Classify a raw `*Url` content-part string into the right [`MediaSource`] variant.
///
/// Recognizes, in order: a `data:` URI (inline base64), a `file://` URI (explicit
/// local-file reference), and a bare absolute filesystem path (no scheme at all --
/// what callers that predate the `file://` convention, e.g. PSRL's own gateway, send).
/// Everything else -- including a string that merely fails to parse as a URL for some
/// other reason -- falls back to the pre-existing behavior of treating it as an HTTP(S)
/// URL, so `fetch_http_*`'s own error reporting is unchanged for genuinely malformed input.
///
/// Before this function existed, every one of these three cases other than `data:`
/// fell into the same `_ => MediaSource::Url(url)` arm, including both a `file://` URI
/// and a schemeless absolute path -- `MediaSource::File`'s already-implemented,
/// already-tested fetch path (`MediaConnector::fetch_file`, with its
/// `allowed_local_media_path` sandbox check) was never reachable from here. See
/// `workspace/issue3/README.md` for the full writeup.
fn classify_media_url(url: String) -> MediaSource {
    match url::Url::parse(&url) {
        Ok(parsed) if parsed.scheme() == "data" => MediaSource::DataUrl(url),
        Ok(parsed) if parsed.scheme() == "file" => match parsed.to_file_path() {
            Ok(path) => MediaSource::File(path),
            // Malformed file:// URL (e.g. a host component) -- fall through to the
            // Url path so the existing InvalidUrl/HTTP error reporting still applies,
            // rather than silently mis-reading a URL the caller mis-constructed.
            Err(()) => MediaSource::Url(url),
        },
        _ if url.starts_with('/') => MediaSource::File(std::path::PathBuf::from(url)),
        _ => MediaSource::Url(url),
    }
}

#[derive(Debug)]
pub struct TrackerOutput {
    pub data: MultiModalData,
    pub uuids: MultiModalUUIDs,
}

pub struct AsyncMultiModalTracker {
    media_connector: Arc<MediaConnector>,
    pending: HashMap<Modality, Vec<PendingTask>>,
    uuids: MultiModalUUIDs,
}

impl AsyncMultiModalTracker {
    pub fn new(media_connector: Arc<MediaConnector>) -> Self {
        Self {
            media_connector,
            pending: HashMap::new(),
            uuids: HashMap::new(),
        }
    }

    pub fn push_part(&mut self, part: MediaContentPart) -> MultiModalResult<()> {
        match part {
            MediaContentPart::Text { .. } => {}
            MediaContentPart::ImageUrl { url, detail, uuid } => {
                let source = classify_media_url(url);
                self.enqueue_image(source, detail.unwrap_or_default(), uuid);
            }
            MediaContentPart::ImageData {
                data,
                mime_type: _,
                uuid,
                detail,
            } => {
                self.enqueue_image(
                    MediaSource::InlineBytes(data),
                    detail.unwrap_or_default(),
                    uuid,
                );
            }
            MediaContentPart::ImageEmbeds { .. } => {
                return Err(MultiModalError::UnsupportedContent("image_embeds"));
            }
            MediaContentPart::AudioUrl { url, uuid } => {
                let source = classify_media_url(url);
                self.enqueue_audio(source, uuid);
            }
            MediaContentPart::AudioData {
                data,
                mime_type: _,
                uuid,
            } => {
                self.enqueue_audio(MediaSource::InlineBytes(data), uuid);
            }
            MediaContentPart::VideoUrl { url, uuid } => {
                let source = classify_media_url(url);
                self.enqueue_video(source, uuid);
            }
            MediaContentPart::VideoData {
                data,
                mime_type: _,
                uuid,
            } => {
                self.enqueue_video(MediaSource::InlineBytes(data), uuid);
            }
        }
        Ok(())
    }

    pub async fn finalize(mut self) -> MultiModalResult<TrackerOutput> {
        let mut data = MultiModalData::new();
        for (modality, tasks) in self.pending.drain() {
            let mut items = Vec::with_capacity(tasks.len());
            for task in tasks {
                let media = task.await??;
                items.push(media);
            }
            data.insert(modality, items);
        }

        Ok(TrackerOutput {
            data,
            uuids: self.uuids,
        })
    }

    fn enqueue_image(&mut self, source: MediaSource, detail: ImageDetail, uuid: Option<String>) {
        let modality = Modality::Image;
        self.uuids.entry(modality).or_default().push(uuid);

        let connector = Arc::clone(&self.media_connector);
        #[expect(
            clippy::disallowed_methods,
            reason = "spawn handle is stored in self.pending and awaited in finalize(); fire-and-forget is intentional for concurrent media fetching"
        )]
        let handle = tokio::spawn(async move {
            let frame = connector
                .fetch_image(source, ImageFetchConfig { detail })
                .await?;
            Ok(TrackedMedia::Image(frame))
        });

        self.pending.entry(modality).or_default().push(handle);
    }

    fn enqueue_video(&mut self, source: MediaSource, uuid: Option<String>) {
        let modality = Modality::Video;
        self.uuids.entry(modality).or_default().push(uuid);

        let connector = Arc::clone(&self.media_connector);
        #[expect(
            clippy::disallowed_methods,
            reason = "spawn handle is stored in self.pending and awaited in finalize(); fire-and-forget is intentional for concurrent media fetching"
        )]
        let handle = tokio::spawn(async move {
            let clip = connector
                .fetch_video(source, VideoFetchConfig::default())
                .await?;
            Ok(TrackedMedia::Video(clip))
        });

        self.pending.entry(modality).or_default().push(handle);
    }

    fn enqueue_audio(&mut self, source: MediaSource, uuid: Option<String>) {
        let modality = Modality::Audio;
        self.uuids.entry(modality).or_default().push(uuid);

        let connector = Arc::clone(&self.media_connector);
        #[expect(
            clippy::disallowed_methods,
            reason = "spawn handle is stored in self.pending and awaited in finalize(); fire-and-forget is intentional for concurrent media fetching"
        )]
        let handle = tokio::spawn(async move {
            let clip = connector.fetch_audio(source).await?;
            Ok(TrackedMedia::Audio(clip))
        });

        self.pending.entry(modality).or_default().push(handle);
    }
}
