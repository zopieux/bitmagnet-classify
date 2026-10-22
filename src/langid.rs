use regex::Regex;
use std::path::Path;
use std::sync::LazyLock;
use tracing::info;

use crate::text;

pub const DEFAULT_THRESHOLD: f32 = 0.65;

const MIN_DETECT_LETTERS: usize = 8;
const SHORT_TEXT_WORDS: usize = 4;
const SHORT_TEXT_THRESHOLD: f32 = 0.5;

#[inline]
pub fn is_kana(c: u32) -> bool {
    (0x3040..=0x30FF).contains(&c)
        || (0x31F0..=0x31FF).contains(&c)
        || (0xFF66..=0xFF9F).contains(&c)
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

#[inline]
fn is_hebrew(c: u32) -> bool {
    (0x0590..=0x05FF).contains(&c)
}

#[inline]
fn is_arabic(c: u32) -> bool {
    (0x0600..=0x06FF).contains(&c) || (0x0750..=0x077F).contains(&c)
}

#[inline]
fn is_greek(c: u32) -> bool {
    (0x0370..=0x03FF).contains(&c) || (0x1F00..=0x1FFF).contains(&c)
}

#[inline]
fn is_devanagari(c: u32) -> bool {
    (0x0900..=0x097F).contains(&c)
}

#[inline]
fn is_thai(c: u32) -> bool {
    (0x0E00..=0x0E7F).contains(&c)
}

fn is_hex_like(text: &str) -> bool {
    let mut hex_chars = 0;
    for ch in text.chars() {
        if ch.is_ascii_hexdigit() {
            hex_chars += 1;
        } else if !ch.is_whitespace() && !matches!(ch, '-' | '_' | '.') {
            return false;
        }
    }
    hex_chars >= 16
}

fn iso639_1(code: &str) -> Option<&'static str> {
    Some(match code {
        "eng" => "en",
        "zho" => "zh",
        "jpn" => "ja",
        "kor" => "ko",
        "rus" => "ru",
        "ukr" => "uk",
        "bel" => "be",
        "bul" => "bg",
        "srp" => "sr",
        "mkd" => "mk",
        "spa" => "es",
        "fra" => "fr",
        "deu" => "de",
        "ita" => "it",
        "por" => "pt",
        "nld" => "nl",
        "pol" => "pl",
        "ces" => "cs",
        "slk" => "sk",
        "hun" => "hu",
        "ron" => "ro",
        "swe" => "sv",
        "nor" => "no",
        "dan" => "da",
        "fin" => "fi",
        "tur" => "tr",
        "ara" => "ar",
        "heb" => "he",
        "ell" => "el",
        "hin" => "hi",
        "tha" => "th",
        "vie" => "vi",
        "ind" => "id",
        "msa" => "ms",
        "cat" => "ca",
        "eus" => "eu",
        "glg" => "gl",
        "lat" => "la",
        "epo" => "eo",
        _ => return None,
    })
}

fn script_matches(lang: &str, text: &str) -> bool {
    let has = |predicate: fn(u32) -> bool| text.chars().any(|ch| predicate(ch as u32));
    match lang {
        "ru" | "uk" | "be" | "bg" | "mk" | "kk" | "ky" | "tg" | "rus" | "ukr" | "bel" | "bul"
        | "mkd" | "kaz" | "kir" | "tgk" => has(is_cyrillic),
        "he" | "yi" | "heb" | "yid" => has(is_hebrew),
        "ar" | "fa" | "ur" | "ps" | "ara" | "fas" | "urd" | "pus" => has(is_arabic),
        "el" | "ell" => has(is_greek),
        "hi" | "mr" | "ne" | "hin" | "mar" | "nep" => has(is_devanagari),
        "th" | "tha" => has(is_thai),
        _ => true,
    }
}

static DIGITS_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\d+").expect("Valid digits regex"));

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

        if letters == 0 || is_hex_like(text) {
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

        let sanitized = text::sanitize_title_for_model(text);
        let sanitized = DIGITS_RE.replace_all(&sanitized, " ");
        let sanitized = sanitized.split_whitespace().collect::<Vec<_>>().join(" ");

        let letter_count = sanitized.chars().filter(|c| c.is_alphabetic()).count();
        if letter_count < MIN_DETECT_LETTERS {
            return "other".to_string();
        }

        let word_count = sanitized.split_whitespace().count();
        let threshold = if word_count <= SHORT_TEXT_WORDS {
            self.threshold.min(SHORT_TEXT_THRESHOLD)
        } else {
            self.threshold
        };

        match &self.backend {
            LangBackend::FastText(model) => {
                if let Ok(predictions) = model.predict(&sanitized, 1, 0.0)
                    && let Some(pred) = predictions.first().filter(|p| p.probability >= threshold)
                {
                    let lang = pred.label.trim_start_matches("__label__");
                    return if script_matches(lang, &sanitized) {
                        lang.to_string()
                    } else {
                        "other".to_string()
                    };
                }

                if let Some(info) =
                    whatlang::detect(&sanitized).filter(|i| i.confidence() >= (threshold as f64))
                    && let Some(lang) = iso639_1(info.lang().code())
                    && script_matches(lang, &sanitized)
                {
                    return lang.to_string();
                }
            }
            LangBackend::WhatLang => {
                if let Some(info) =
                    whatlang::detect(&sanitized).filter(|i| i.confidence() >= (threshold as f64))
                {
                    let lang = info.lang().code();
                    if script_matches(lang, &sanitized) {
                        return lang.to_string();
                    }
                }
            }
        }

        "other".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_hex_like_names() {
        assert!(is_hex_like(
            "7508529af4bd8bb3eeb09743fdfd99c6142283e1ece50e18d153a1676fb3555e"
        ));
        assert!(!is_hex_like("The Big Short"));
    }

    #[test]
    fn rejects_predictions_with_mismatched_scripts() {
        assert!(!script_matches("he", "Ichigo Mashimaro"));
        assert!(script_matches("ru", "Кошка и самурай"));
        assert!(script_matches("en", "The Big Short"));
    }
}
