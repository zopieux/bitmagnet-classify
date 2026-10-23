use std::collections::HashSet;

use crate::nsfw;

const ASSET_EXTENSIONS: &[&str] = &[
    "3dm", "ai", "blend", "c4d", "cdi", "dxf", "dwg", "eps", "fbx", "indd", "max", "obj", "psd",
    "skp", "step", "stl", "stp",
];

const SOFTWARE_EXTENSIONS: &[&str] = &["apk", "appimage", "deb", "dmg", "exe", "msi", "pkg", "rpm"];

const CONTAINER_EXTENSIONS: &[&str] = &["7z", "bin", "gz", "img", "iso", "rar", "tar", "zip"];

const VIDEO_EXTENSIONS: &[&str] = &[
    "avi", "flv", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "webm", "wmv",
];

const SOFTWARE_MARKERS: &[&str] = &[
    "3d model",
    "adobe",
    "after effects",
    "android",
    "autocad",
    "bios",
    "blender",
    "borisfx",
    "cad",
    "cinema 4d",
    "crumplepop",
    "driver",
    "firmware",
    "fitgirl",
    "godot",
    "illustrator",
    "installer",
    "kali",
    "linux",
    "macos",
    "maya",
    "microsoft",
    "nethunter",
    "office",
    "photoshop",
    "plugin",
    "portable",
    "premiere",
    "preset",
    "repack",
    "revit",
    "sample pack",
    "setup",
    "sketchup",
    "steam",
    "substance",
    "texture pack",
    "torrentmac",
    "ubuntu",
    "unity",
    "unreal",
    "virtualdj",
    "vst",
    "vsti",
    "win64",
    "windows",
    "x64",
    "zbrush",
];

const MEDIA_MARKERS: &[&str] = &[
    "album",
    "discography",
    "flac",
    "lossless",
    "music",
    "op single",
    "ost",
    "soundtrack",
];

fn has_any_extension(extensions: &HashSet<String>, candidates: &[&str]) -> bool {
    candidates.iter().any(|ext| extensions.contains(*ext))
}

fn has_any_marker(title: &str, markers: &[&str]) -> bool {
    let title = title.to_lowercase();
    markers.iter().any(|marker| title.contains(marker))
}

fn is_opaque_identifier(title: &str) -> bool {
    let stem = title.rsplit_once('.').map_or(title, |(stem, _)| stem);
    let mut tokens = stem
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty());

    let Some(token) = tokens.next() else {
        return false;
    };

    tokens.next().is_none() && token.len() >= 32 && token.chars().all(|c| c.is_ascii_hexdigit())
}

pub fn is_clean_artifact(title: &str, extensions: &HashSet<String>) -> bool {
    if title.trim().is_empty() || nsfw::is_keyword_nsfw(title) {
        return false;
    }

    if is_opaque_identifier(title) && !has_any_extension(extensions, VIDEO_EXTENSIONS) {
        return true;
    }

    if has_any_extension(extensions, ASSET_EXTENSIONS) {
        return true;
    }

    if has_any_extension(extensions, SOFTWARE_EXTENSIONS) && has_any_marker(title, SOFTWARE_MARKERS)
    {
        return true;
    }

    if has_any_extension(extensions, CONTAINER_EXTENSIONS)
        && (has_any_marker(title, SOFTWARE_MARKERS) || has_any_marker(title, MEDIA_MARKERS))
    {
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extensions(values: &[&str]) -> HashSet<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn guards_known_non_adult_artifacts() {
        assert!(is_clean_artifact(
            "virtualdj823205.iso",
            &extensions(&["iso"])
        ));
        assert!(is_clean_artifact(
            "dog tag cut radius.stl",
            &extensions(&["stl"])
        ));
        assert!(is_clean_artifact(
            "kali-nethunter-2026.2-beyondx-los-sixteen-full-zip",
            &extensions(&["zip"])
        ));
        assert!(is_clean_artifact(
            "BorisFX_CrumplePop_2025.0.8_Torrentmac.net.dmg",
            &extensions(&["dmg"])
        ));
        assert!(is_clean_artifact(
            "[Nipponsei] Occultic;Nine OP Single - Seisuu 3 no Nijou [Itou Kanako].zip",
            &extensions(&["zip"])
        ));
        assert!(is_clean_artifact("vigil82nd.cdi", &extensions(&["cdi"])));
    }

    #[test]
    fn guards_opaque_hashes() {
        assert!(is_clean_artifact(
            "5c8f72677a04f5535dc5aa03bb4d4bde2e644bf31810734452d1569e12e643fa38ca0ad13d2747700e6d7701d52536ca6a313f92a342f121b5bf8072e9e2612593239b",
            &HashSet::new()
        ));
        assert!(!is_clean_artifact(
            "5c8f72677a04f5535dc5aa03bb4d4bde2e644bf31810734452d1569e12e643fa",
            &extensions(&["mkv"])
        ));
    }

    #[test]
    fn does_not_guard_semantic_titles() {
        assert!(!is_clean_artifact(
            "Naked - Mike Leigh",
            &extensions(&["mkv"])
        ));
        assert!(!is_clean_artifact(
            "Enough.2002.HDTVRip.avi",
            &extensions(&["avi"])
        ));
        assert!(!is_clean_artifact(
            "Some Movie 2024.iso",
            &extensions(&["iso"])
        ));
        assert!(!is_clean_artifact(
            "Holiday Photos.zip",
            &extensions(&["zip"])
        ));
    }

    #[test]
    fn adult_keywords_take_precedence_over_guards() {
        assert!(!is_clean_artifact(
            "Brazzers Collection.iso",
            &extensions(&["iso"])
        ));
        assert!(!is_clean_artifact("ABP-449.stl", &extensions(&["stl"])));
    }
}
