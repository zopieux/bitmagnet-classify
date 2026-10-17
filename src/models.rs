use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum VerdictAction {
    #[default]
    Keep,
    Delete,
    Unmatched,
}

impl std::fmt::Display for VerdictAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Keep => write!(f, "keep"),
            Self::Delete => write!(f, "delete"),
            Self::Unmatched => write!(f, "unmatched"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TorrentFile {
    pub path: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub extension: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct TorrentHint {
    #[serde(default)]
    pub content_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TorrentPayload {
    pub info_hash: String,
    pub name: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub extension: Option<String>,
    #[serde(default)]
    pub files: Vec<TorrentFile>,
    #[serde(default)]
    pub hint: Option<TorrentHint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassificationVerdict {
    #[serde(default)]
    pub action: VerdictAction,
    #[serde(default)]
    pub content_type: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub release_group: Option<String>,
}

impl Default for ClassificationVerdict {
    fn default() -> Self {
        Self {
            action: VerdictAction::Keep,
            content_type: None,
            tags: Vec::new(),
            release_group: None,
        }
    }
}
