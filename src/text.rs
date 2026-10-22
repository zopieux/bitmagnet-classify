use regex::Regex;
use std::sync::LazyLock;

static CLEAN_EXT_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\.(?:mp4|mkv|avi|wmv|mov|flv|iso|zip|rar|tar|gz|7z|epub|pdf|cbz|cbr|txt|nfo|sfv)$",
    )
    .expect("Valid ext regex")
});

static BRACKET_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\[([^\]]*)\]|\{([^\}]*)\}|\(([^\)]*)\)").expect("Valid brackets regex")
});

static CLEAN_DELIMITER_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[\[\]{}()]").expect("Valid delimiter regex"));

static CLEAN_SCENE_TAGS_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:\d{3,4}[pi]|uhd|4k|bluray|blu-ray|bdrip|brrip|dvdrip|dvd|web-?dl|webrip|web|hdtv|hdrip|remux|h\.?264|x\.?264|h\.?265|x\.?265|hevc|avc|av1|10bit|8bit|xvid|divx|aac|ac3|eac3|ddp?5\.1|dts(?:-hd)?|truehd|flac|mp3|opus|repack|proper|unrated|remastered|extended|director'?s cut|limited|internal|readnfo|rerip)\b")
        .expect("Valid scene tags regex")
});

fn is_year(text: &str) -> bool {
    text.len() == 4
        && (text.starts_with("19") || text.starts_with("20"))
        && text.bytes().all(|b| b.is_ascii_digit())
}

pub fn sanitize_title_for_model(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let cleaned = CLEAN_EXT_PATTERN.replace(text, " ");
    let cleaned = BRACKET_PATTERN.replace_all(&cleaned, |caps: &regex::Captures| {
        let content = caps
            .get(1)
            .or_else(|| caps.get(2))
            .or_else(|| caps.get(3))
            .map(|m| m.as_str().trim())
            .unwrap_or("");
        let keep = is_year(content) || crate::nsfw::is_keyword_nsfw(content);
        if keep {
            format!(" {content} ")
        } else {
            " ".to_string()
        }
    });
    let cleaned = CLEAN_DELIMITER_PATTERN.replace_all(&cleaned, " ");
    let cleaned = CLEAN_SCENE_TAGS_PATTERN.replace_all(&cleaned, " ");
    let cleaned = cleaned.replace(['.', '_', '-'], " ");
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize() {
        assert_eq!(
            sanitize_title_for_model("Wreck-It Ralph 2012 Rus BDRip-AVC 402p.mkv"),
            "Wreck It Ralph 2012 Rus"
        );
        assert_eq!(
            sanitize_title_for_model("Circo.Ambulante (1995) [1080p].avi"),
            "Circo Ambulante 1995"
        );
        assert_eq!(
            sanitize_title_for_model("Circo.Ambulante (Remastered).avi"),
            "Circo Ambulante"
        );
    }

    #[test]
    fn preserves_semantic_brackets_and_parens() {
        assert_eq!(
            sanitize_title_for_model("[Tushy] Kelly Collins (First Shemale).mp4"),
            "Tushy Kelly Collins First Shemale"
        );
        assert_eq!(
            sanitize_title_for_model("[BelAmi] Pillow Talk 3"),
            "BelAmi Pillow Talk 3"
        );
        assert_eq!(
            sanitize_title_for_model("Cock in the Sun (Cock in the Sun)"),
            "Cock in the Sun Cock in the Sun"
        );
    }

    #[test]
    fn strips_release_group_noise() {
        assert_eq!(
            sanitize_title_for_model("[HorribleSubs] Kill la Kill - 02 [720p]"),
            "Kill la Kill 02"
        );
        assert_eq!(
            sanitize_title_for_model("[YTS.MX] The Matrix 1999 1080p BluRay"),
            "The Matrix 1999"
        );
        assert_eq!(
            sanitize_title_for_model("[Erai-raws] Show - 01 [1080p]"),
            "Show 01"
        );
        assert_eq!(
            sanitize_title_for_model(
                "[JacobSwaggedUp] Danna ga Nani wo Itteiru ka Wakaranai Ken [Season 2] (BD 1280x720)"
            ),
            "Danna ga Nani wo Itteiru ka Wakaranai Ken"
        );
        assert_eq!(
            sanitize_title_for_model("[Scinna] Frieren [AI Generated].zip"),
            "Frieren"
        );
    }

    #[test]
    fn preserves_adult_bracket_content() {
        assert_eq!(
            sanitize_title_for_model("[Blacked] Avery Cristy [1080p]"),
            "Blacked Avery Cristy"
        );
        assert_eq!(sanitize_title_for_model("[ABP-449].avi"), "ABP 449");
        assert_eq!(
            sanitize_title_for_model("[EvilAngel] Scene (First Shemale)"),
            "EvilAngel Scene First Shemale"
        );
    }
}
