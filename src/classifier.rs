use anyhow::Result;
use std::collections::HashSet;

use crate::banned;
use crate::guards;
use crate::langid::LanguageDetector;
use crate::models::{ClassificationVerdict, TorrentFile, TorrentPayload, VerdictAction};
use crate::nsfw::{self, NsfwModel};

pub const BORDERLINE_LOW: f32 = 0.35;
pub const BORDERLINE_HIGH: f32 = 0.70;

pub const SAFE_DOC_AUDIO_EXTENSIONS: &[&str] = &[
    "epub", "pdf", "mobi", "azw3", "djvu", "cbz", "cbr", "flac", "alac", "ape", "wav", "dsf", "dff",
];

pub const SAFE_AUX_EXTENSIONS: &[&str] = &[
    "txt", "nfo", "cue", "log", "jpg", "jpeg", "png", "sfv", "md5",
];

pub const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "mkv", "avi", "wmv", "mov", "flv", "m4v", "mpg", "mpeg", "webm", "ts",
];

pub const NSFW_PATH_IGNORED_EXTENSIONS: &[&str] = &[
    "txt", "nfo", "cue", "log", "sfv", "md5", "srt", "sub", "ass", "ssa", "url", "html", "htm",
    "epub", "pdf", "mobi", "azw3", "djvu", "cbz", "cbr", "flac", "alac", "ape", "wav", "dsf",
    "dff", "mp3", "m4a", "aac", "ogg", "opus", "wma",
];

pub const BANNED_PATH_IGNORED_EXTENSIONS: &[&str] = &[
    "txt", "nfo", "url", "html", "htm", "mht", "mhtml", "chm", "ini", "sfv", "md5", "cue", "log",
    "apk",
];

fn file_extension(file: &TorrentFile) -> Option<&str> {
    file.extension
        .as_deref()
        .or_else(|| file.path.rsplit_once('.').map(|(_, ext)| ext))
}

fn is_nsfw_path_candidate(file: &TorrentFile) -> bool {
    match file_extension(file) {
        Some(ext) => {
            let ext = ext.to_lowercase();
            !NSFW_PATH_IGNORED_EXTENSIONS.contains(&ext.as_str())
        }
        None => true,
    }
}

fn is_banned_path_candidate(file: &TorrentFile) -> bool {
    match file_extension(file) {
        Some(ext) => {
            let ext = ext.to_lowercase();
            !BANNED_PATH_IGNORED_EXTENSIONS.contains(&ext.as_str())
        }
        None => true,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum EarlyLangAction {
    Delete,
    #[default]
    SkipsNsfw,
    PostNsfw,
}

pub struct Classifier {
    pub lang_detector: LanguageDetector,
    pub nsfw_model: Option<NsfwModel>,
    pub default_early_langs: HashSet<String>,
    pub early_lang_action: EarlyLangAction,
}

impl Classifier {
    pub fn new(
        lang_detector: LanguageDetector,
        nsfw_model: Option<NsfwModel>,
        default_early_langs: HashSet<String>,
        early_lang_action: EarlyLangAction,
    ) -> Self {
        Self {
            lang_detector,
            nsfw_model,
            default_early_langs,
            early_lang_action,
        }
    }

    fn early_lang_verdict(&self, tags: &[String]) -> Option<ClassificationVerdict> {
        match self.early_lang_action {
            EarlyLangAction::Delete => Some(ClassificationVerdict {
                action: VerdictAction::Delete,
                ..Default::default()
            }),
            EarlyLangAction::SkipsNsfw => Some(ClassificationVerdict {
                action: VerdictAction::Keep,
                tags: tags.to_vec(),
                ..Default::default()
            }),
            EarlyLangAction::PostNsfw => None,
        }
    }

    pub fn classify(
        &self,
        payload: &TorrentPayload,
        early_lang_override: Option<&[String]>,
        classified_tag: Option<&str>,
    ) -> Result<ClassificationVerdict> {
        if banned::is_banned(&payload.name)
            || payload
                .files
                .iter()
                .any(|f| is_banned_path_candidate(f) && banned::is_banned(&f.path))
        {
            return Ok(ClassificationVerdict {
                action: VerdictAction::Delete,
                ..Default::default()
            });
        }

        let mut tags: Vec<String> = Vec::new();
        if let Some(tag) = classified_tag.filter(|s| !s.is_empty()) {
            tags.push(tag.to_string());
        }

        let skip_langs: HashSet<&str> = if let Some(override_langs) = early_lang_override {
            override_langs.iter().map(|s| s.as_str()).collect()
        } else {
            self.default_early_langs
                .iter()
                .map(|s| s.as_str())
                .collect()
        };

        // 1. Language detection on title
        let detected_lang = self.lang_detector.detect(&payload.name);
        if detected_lang != "other" {
            tags.push(format!("lang-{detected_lang}"));
            if skip_langs.contains(detected_lang.as_str())
                && let Some(verdict) = self.early_lang_verdict(&tags)
            {
                return Ok(verdict);
            }
        }

        // 2. Check file paths for distinct languages
        if !payload.files.is_empty() {
            let mut file_langs: HashSet<String> = HashSet::new();
            for f in &payload.files {
                let fl = self.lang_detector.detect(&f.path);
                if fl != "other" && fl != detected_lang {
                    file_langs.insert(fl);
                }
            }

            let mut sorted_file_langs: Vec<String> = file_langs.into_iter().collect();
            sorted_file_langs.sort();

            for fl in sorted_file_langs {
                tags.push(format!("lang-{fl}"));
                if skip_langs.contains(fl.as_str())
                    && let Some(verdict) = self.early_lang_verdict(&tags)
                {
                    return Ok(verdict);
                }
            }
        }

        // 3. Fast keyword matching for adult terms
        if !nsfw::is_mainstream_allowed(&payload.name) {
            if nsfw::is_keyword_nsfw(&payload.name) {
                tags.push("nsfw-keyword".to_string());
                return Ok(ClassificationVerdict {
                    action: VerdictAction::Keep,
                    tags,
                    content_type: Some("xxx".to_string()),
                    ..Default::default()
                });
            }

            // Check sample of file paths for adult keywords (up to 10)
            for f in payload.files.iter().take(10) {
                if !is_nsfw_path_candidate(f) {
                    continue;
                }
                if nsfw::is_keyword_nsfw(&f.path) {
                    tags.push("nsfw-keyword".to_string());
                    return Ok(ClassificationVerdict {
                        action: VerdictAction::Keep,
                        tags,
                        content_type: Some("xxx".to_string()),
                        ..Default::default()
                    });
                }
            }
        }

        let mut file_extensions: HashSet<String> = HashSet::new();
        for f in &payload.files {
            if let Some(ext) = file_extension(f) {
                file_extensions.insert(ext.to_lowercase());
            }
        }
        if let Some(ext) = &payload.extension {
            file_extensions.insert(ext.to_lowercase());
        }

        let has_nsfw_paths = payload
            .files
            .iter()
            .any(|f| is_nsfw_path_candidate(f) && nsfw::is_keyword_nsfw(&f.path));

        if !has_nsfw_paths && guards::is_clean_artifact(&payload.name, &file_extensions) {
            return Ok(ClassificationVerdict {
                action: VerdictAction::Keep,
                tags,
                ..Default::default()
            });
        }

        // 4. Early exit for non-video safe extensions (books, lossless audio, documents)
        if !file_extensions.is_empty() {
            let has_video = file_extensions
                .iter()
                .any(|ext| VIDEO_EXTENSIONS.contains(&ext.as_str()));

            let only_safe = file_extensions.iter().all(|ext| {
                SAFE_DOC_AUDIO_EXTENSIONS.contains(&ext.as_str())
                    || SAFE_AUX_EXTENSIONS.contains(&ext.as_str())
            });

            if only_safe && !has_video {
                return Ok(ClassificationVerdict {
                    action: VerdictAction::Keep,
                    tags,
                    ..Default::default()
                });
            }
        }

        // 5. Fallback to ML sequence classifier if model is loaded
        let mut is_nsfw_final = false;
        if let Some(ref model) = self.nsfw_model {
            let (mut is_nsfw, prob) =
                model.is_model_nsfw(&payload.name, nsfw::DEFAULT_THRESHOLD)?;

            // Borderline ambiguity check: defer to file paths to confirm or refute
            if (BORDERLINE_LOW..=BORDERLINE_HIGH).contains(&prob) && !payload.files.is_empty() {
                let mut file_scores: Vec<f32> = Vec::new();
                for f in payload.files.iter().take(3) {
                    if file_extension(f).is_some_and(|ext| {
                        NSFW_PATH_IGNORED_EXTENSIONS.contains(&ext.to_lowercase().as_str())
                    }) {
                        continue;
                    }
                    let (_, f_prob) = model.is_model_nsfw(&f.path, nsfw::DEFAULT_THRESHOLD)?;
                    file_scores.push(f_prob);
                }

                if !file_scores.is_empty() {
                    let max_file_score = file_scores
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    if max_file_score >= 0.70 {
                        tags.push("nsfw-model".to_string());
                        return Ok(ClassificationVerdict {
                            action: VerdictAction::Keep,
                            tags,
                            content_type: Some("xxx".to_string()),
                            ..Default::default()
                        });
                    } else if max_file_score < 0.35 {
                        is_nsfw = false;
                    }
                }
            }

            if is_nsfw {
                tags.push("nsfw-model".to_string());
                is_nsfw_final = true;
            }
        }

        Ok(ClassificationVerdict {
            action: VerdictAction::Keep,
            tags,
            content_type: if is_nsfw_final {
                Some("xxx".to_string())
            } else {
                None
            },
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn torrent_file(path: &str, extension: Option<&str>) -> TorrentFile {
        TorrentFile {
            path: path.to_string(),
            size: 0,
            extension: extension.map(str::to_string),
        }
    }

    #[test]
    fn infers_file_extensions_from_paths() {
        assert_eq!(
            file_extension(&torrent_file("music/track.mp3", None)),
            Some("mp3")
        );
        assert_eq!(
            file_extension(&torrent_file("movie.mkv", None)),
            Some("mkv")
        );
        assert_eq!(
            file_extension(&torrent_file("movie.mkv", Some("mkv"))),
            Some("mkv")
        );
    }

    #[test]
    fn ignores_spam_paths_for_banned_matching() {
        assert!(!is_banned_path_candidate(&torrent_file(
            "全国各地援交妹.mht",
            None
        )));
        assert!(is_banned_path_candidate(&torrent_file(
            "V/赠品/Loli/38.jpg",
            None
        )));
    }

    #[test]
    fn ignores_audio_paths_for_nsfw_keywords() {
        assert!(!is_nsfw_path_candidate(&torrent_file(
            "01 - Fucking Noob Slug.mp3",
            None
        )));
        assert!(is_nsfw_path_candidate(&torrent_file(
            "movie/scene.mp4",
            None
        )));
    }
}
