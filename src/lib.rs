pub mod banned;
pub mod classifier;
pub mod client;
pub mod guards;
pub mod langid;
pub mod models;
pub mod nsfw;
pub mod text;

pub use classifier::Classifier;
pub use client::BitmagnetClient;
pub use langid::LanguageDetector;
pub use models::{ClassificationVerdict, TorrentFile, TorrentHint, TorrentPayload, VerdictAction};
pub use nsfw::NsfwModel;
