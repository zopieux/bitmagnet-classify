use std::collections::HashSet;
use anyhow::Result;

use crate::langid::LanguageDetector;
use crate::models::{ClassificationVerdict, TorrentPayload, VerdictAction};
use crate::nsfw::{is_keyword_nsfw, is_mainstream_allowed, NsfwModel, DEFAULT_THRESHOLD};

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

pub struct Classifier {
    pub lang_detector: LanguageDetector,
    pub nsfw_model: Option<NsfwModel>,
    pub default_early_langs: HashSet<String>,
}

impl Classifier {
    pub fn new(
        lang_detector: LanguageDetector,
        nsfw_model: Option<NsfwModel>,
        default_early_langs: HashSet<String>,
    ) -> Self {
        Self {
            lang_detector,
            nsfw_model,
            default_early_langs,
        }
    }

    pub fn classify(
        &self,
        payload: &TorrentPayload,
        early_lang_override: Option<&[String]>,
    ) -> Result<ClassificationVerdict> {
        let mut tags: Vec<String> = vec!["external".to_string()];

        let skip_langs: HashSet<&str> = if let Some(override_langs) = early_lang_override {
            override_langs.iter().map(|s| s.as_str()).collect()
        } else {
            self.default_early_langs.iter().map(|s| s.as_str()).collect()
        };

        // 1. Language detection on title
        let detected_lang = self.lang_detector.detect(&payload.name);
        if detected_lang != "other" {
            tags.push(format!("lang-{detected_lang}"));
            if skip_langs.contains(detected_lang.as_str()) {
                return Ok(ClassificationVerdict {
                    action: VerdictAction::Keep,
                    tags,
                    ..Default::default()
                });
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
                if skip_langs.contains(fl.as_str()) {
                    return Ok(ClassificationVerdict {
                        action: VerdictAction::Keep,
                        tags,
                        ..Default::default()
                    });
                }
            }
        }

        // 3. Fast keyword matching for adult terms
        if !is_mainstream_allowed(&payload.name) {
            if is_keyword_nsfw(&payload.name) {
                tags.push("nsfw-keyword".to_string());
                return Ok(ClassificationVerdict {
                    action: VerdictAction::Keep,
                    tags,
                    ..Default::default()
                });
            }

            // Check sample of file paths for adult keywords (up to 10)
            for f in payload.files.iter().take(10) {
                if is_keyword_nsfw(&f.path) {
                    tags.push("nsfw-keyword".to_string());
                    return Ok(ClassificationVerdict {
                        action: VerdictAction::Keep,
                        tags,
                        ..Default::default()
                    });
                }
            }
        }

        // 4. Early exit for non-video safe extensions (books, lossless audio, documents)
        let mut file_extensions: HashSet<String> = HashSet::new();
        for f in &payload.files {
            if let Some(ext) = &f.extension {
                file_extensions.insert(ext.to_lowercase());
            }
        }
        if let Some(ext) = &payload.extension {
            file_extensions.insert(ext.to_lowercase());
        }

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
        if let Some(ref model) = self.nsfw_model {
            let (mut is_nsfw, prob) = model.is_model_nsfw(&payload.name, DEFAULT_THRESHOLD)?;

            // Borderline ambiguity check: defer to file paths to confirm or refute
            if (BORDERLINE_LOW..=BORDERLINE_HIGH).contains(&prob) && !payload.files.is_empty() {
                let mut file_scores: Vec<f32> = Vec::new();
                for f in payload.files.iter().take(3) {
                    if let Some(ext) = &f.extension {
                        if SAFE_AUX_EXTENSIONS.contains(&ext.to_lowercase().as_str()) {
                            continue;
                        }
                    }
                    let (_, f_prob) = model.is_model_nsfw(&f.path, DEFAULT_THRESHOLD)?;
                    file_scores.push(f_prob);
                }

                if !file_scores.is_empty() {
                    let max_file_score = file_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    if max_file_score >= 0.70 {
                        tags.push("nsfw-model".to_string());
                        return Ok(ClassificationVerdict {
                            action: VerdictAction::Keep,
                            tags,
                            ..Default::default()
                        });
                    } else if max_file_score < 0.35 {
                        is_nsfw = false;
                    }
                }
            }

            if is_nsfw {
                tags.push("nsfw-model".to_string());
            }
        }

        Ok(ClassificationVerdict {
            action: VerdictAction::Keep,
            tags,
            ..Default::default()
        })
    }
}
