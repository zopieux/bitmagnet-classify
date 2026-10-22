use std::path::Path;
use std::sync::LazyLock;
use regex::Regex;
use tracing::info;

use crate::text::sanitize_title_for_model;

pub const DEFAULT_THRESHOLD: f32 = 0.65;

#[inline]
pub fn is_kana(c: u32) -> bool {
    (0x3040..=0x30FF).contains(&c) || (0x31F0..=0x31FF).contains(&c) || (0xFF66..=0xFF9F).contains(&c)
}

#[inline]
pub fn is_han(c: u32) -> bool {
    (0x4E00..=0x9FFF).contains(&c)
        || (0x3400..=0x4DBF).contains(&c)
        || (0x20000..=0x2A6DF).contains(&c)
        || (0xF900..=0xFAFF).contains(&c)
}

#[inline]
pub fn is_hangul(c: u32) -> bool {
    (0xAC00..=0xD7AF).contains(&c)
        || (0x1100..=0x11FF).contains(&c)
        || (0x3130..=0x318F).contains(&c)
        || (0xA960..=0xA97F).contains(&c)
        || (0xD7B0..=0xD7FF).contains(&c)
}

#[inline]
pub fn is_cyrillic(c: u32) -> bool {
    (0x0400..=0x052F).contains(&c)
}

static DIGITS_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d+").expect("Valid digits regex"));

pub enum LangBackend {
    FastText(Box<fasttext_pure_rs::FastText>),
    WhatLang,
}

pub struct LanguageDetector {
    backend: LangBackend,
    threshold: f32,
}

impl LanguageDetector {
    pub fn load_fasttext(path: &Path, threshold: f32) -> anyhow::Result<Self> {
        if !path.is_file() {
            anyhow::bail!("FastText model file not found at {:?}", path);
        }
        let model = fasttext_pure_rs::FastText::load(path)
            .map_err(|e| anyhow::anyhow!("Failed to load fastText model from {:?}: {e}", path))?;
        info!("Loaded pure-Rust fastText language model from {:?}", path);
        Ok(Self {
            backend: LangBackend::FastText(Box::new(model)),
            threshold,
        })
    }

    pub fn with_whatlang(threshold: f32) -> Self {
        info!("Using model-free whatlang language detector");
        Self {
            backend: LangBackend::WhatLang,
            threshold,
        }
    }

    pub fn detect(&self, text: &str) -> String {
        let mut kana = 0;
        let mut han = 0;
        let mut hangul = 0;
        let mut letters = 0;

        for ch in text.chars() {
            if !ch.is_alphabetic() {
                continue;
            }
            letters += 1;
            let c = ch as u32;
            if is_kana(c) {
                kana += 1;
            } else if is_hangul(c) {
                hangul += 1;
            } else if is_han(c) {
                han += 1;
            }
        }

        if letters == 0 {
            return "other".to_string();
        }
        if kana > 0 {
            return "ja".to_string();
        }
        if hangul > 0 {
            return "ko".to_string();
        }
        if han > 0 {
            return "zh".to_string();
        }

        let sanitized = sanitize_title_for_model(text);
        let sanitized = DIGITS_RE.replace_all(&sanitized, " ");
        let sanitized = sanitized.split_whitespace().collect::<Vec<_>>().join(" ");

        if sanitized.is_empty() || !sanitized.chars().any(|c| c.is_alphabetic()) {
            return "other".to_string();
        }

        match &self.backend {
            LangBackend::FastText(model) => {
                if let Ok(predictions) = model.predict(&sanitized, 1, 0.0)
                    && let Some(pred) = predictions.first().filter(|p| p.probability >= self.threshold)
                {
                    return pred.label.trim_start_matches("__label__").to_string();
                }
            }
            LangBackend::WhatLang => {
                if let Some(info) = whatlang::detect(&sanitized).filter(|i| i.confidence() >= (self.threshold as f64)) {
                    return info.lang().code().to_string();
                }
            }
        }

        "other".to_string()
    }
}
