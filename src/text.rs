use std::sync::LazyLock;
use regex::Regex;

static CLEAN_EXT_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\.(?:mp4|mkv|avi|wmv|mov|flv|iso|zip|rar|tar|gz|7z|epub|pdf|cbz|cbr|txt|nfo|sfv)$")
        .expect("Valid ext regex")
});

static CLEAN_BRACKETS_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\[[^\]]*\]|\{[^\}]*\}").expect("Valid brackets regex")
});

static CLEAN_PARENS_PATTERN: LazyLock<fancy_regex::Regex> = LazyLock::new(|| {
    fancy_regex::Regex::new(r"\((?!\b(?:19\d\d|20[0-2]\d)\b)[^\)]*\)")
        .expect("Valid parens regex")
});

static CLEAN_SCENE_TAGS_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:1080[pi]|720[pi]|2160[pi]|4k|uhd|bluray|blu-ray|bdrip|brrip|dvdrip|web-?dl|webrip|hdtv|h264|x264|h265|x265|hevc|avc|10bit|aac|ac3|dts(?:-hd)?|flac|mp3|repack|proper|unrated|remux)\b")
        .expect("Valid scene tags regex")
});

pub fn sanitize_title_for_model(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let cleaned = CLEAN_EXT_PATTERN.replace(text, "");
    let cleaned = CLEAN_BRACKETS_PATTERN.replace_all(&cleaned, " ");
    let cleaned = CLEAN_PARENS_PATTERN.replace_all(&cleaned, " ");
    let cleaned = CLEAN_SCENE_TAGS_PATTERN.replace_all(&cleaned, " ");
    let cleaned = cleaned.replace('.', " ").replace('_', " ").replace('-', " ");
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize() {
        assert_eq!(
            sanitize_title_for_model("Wreck-It Ralph 2012 Rus BDRip-AVC 402p.mkv"),
            "Wreck It Ralph 2012 Rus 402p"
        );
        assert_eq!(
            sanitize_title_for_model("Circo.Ambulante (1995) [1080p].avi"),
            "Circo Ambulante (1995)"
        );
        assert_eq!(
            sanitize_title_for_model("Circo.Ambulante (Remastered).avi"),
            "Circo Ambulante"
        );
    }
}
