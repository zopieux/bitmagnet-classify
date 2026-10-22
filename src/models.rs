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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ApiTorrent {
    #[serde(default)]
    pub info_hash: String,
    #[serde(default)]
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

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiClassificationItem {
    pub info_hash: String,
    #[serde(default)]
    pub content_type: Option<String>,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub torrent: Option<ApiTorrent>,
}

impl ApiClassificationItem {
    pub fn to_payload(&self) -> TorrentPayload {
        if let Some(ref t) = self.torrent {
            let extension = t.extension.clone().or_else(|| {
                t.name.rsplit_once('.').map(|(_, ext)| ext.to_lowercase())
            });
            let hint = t.hint.clone().or_else(|| {
                self.content_type.as_ref().map(|ct| TorrentHint {
                    content_type: Some(ct.clone()),
                })
            });

            TorrentPayload {
                info_hash: self.info_hash.clone(),
                name: t.name.clone(),
                size: if t.size > 0 { t.size } else { self.size },
                extension,
                files: t.files.clone(),
                hint,
            }
        } else {
            TorrentPayload {
                info_hash: self.info_hash.clone(),
                name: String::new(),
                size: self.size,
                extension: None,
                files: Vec::new(),
                hint: self.content_type.as_ref().map(|ct| TorrentHint {
                    content_type: Some(ct.clone()),
                }),
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiGetBatchResponse {
    #[serde(default)]
    pub items: Vec<ApiClassificationItem>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PatchClassificationItem {
    pub info_hash: String,
    pub action: String, // "do_nothing" or "delete"
    pub tags: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiPatchBatchResponse {
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub updated: usize,
}

pub fn is_valid_tag_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    let mut prev_hyphen = true;
    for ch in name.chars() {
        if ch.is_ascii_lowercase() || ch.is_ascii_digit() {
            prev_hyphen = false;
        } else if ch == '-' {
            if prev_hyphen {
                return false;
            }
            prev_hyphen = true;
        } else {
            return false;
        }
    }
    !prev_hyphen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_tag_names() {
        assert!(is_valid_tag_name("ok"));
        assert!(is_valid_tag_name("lang-en"));
        assert!(is_valid_tag_name("nsfw-keyword"));
        assert!(!is_valid_tag_name(""));
        assert!(!is_valid_tag_name("-ok"));
        assert!(!is_valid_tag_name("ok-"));
        assert!(!is_valid_tag_name("OK"));
        assert!(!is_valid_tag_name("ok_tag"));
    }

    #[test]
    fn test_patch_item_serialization() {
        let item = PatchClassificationItem {
            info_hash: "0fc1f47aec8d06c11c54394a1e2e8d128d00b228".to_string(),
            action: "do_nothing".to_string(),
            tags: vec!["ok".to_string(), "lang-en".to_string()],
            content_type: Some("movie".to_string()),
        };

        let json_val = serde_json::to_value(&item).expect("Serialize item");
        assert_eq!(
            json_val,
            serde_json::json!({
                "infoHash": "0fc1f47aec8d06c11c54394a1e2e8d128d00b228",
                "action": "do_nothing",
                "tags": ["ok", "lang-en"],
                "contentType": "movie"
            })
        );
    }

    #[test]
    fn test_api_item_deserialization() {
        let raw = r#"{
            "id": "123",
            "infoHash": "0fc1f47aec8d06c11c54394a1e2e8d128d00b228",
            "contentType": "movie",
            "torrent": {
                "name": "Sample.Movie.1080p.mkv",
                "files": [
                    { "path": "Sample.Movie.1080p.mkv", "size": 1000 }
                ]
            }
        }"#;

        let item: ApiClassificationItem = serde_json::from_str(raw).expect("Deserialize item");
        assert_eq!(item.info_hash, "0fc1f47aec8d06c11c54394a1e2e8d128d00b228");
        let payload = item.to_payload();
        assert_eq!(payload.name, "Sample.Movie.1080p.mkv");
        assert_eq!(payload.extension, Some("mkv".to_string()));
        assert_eq!(payload.files.len(), 1);
    }
}
