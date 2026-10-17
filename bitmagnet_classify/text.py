"""Text preprocessing and sanitization utilities for titles and filenames."""

import re

# File extensions commonly found in torrent titles or paths
_CLEAN_EXT_PATTERN = re.compile(
    r"\.(?:mp4|mkv|avi|wmv|mov|flv|iso|zip|rar|tar|gz|7z|epub|pdf|cbz|cbr|txt|nfo|sfv)$",
    re.IGNORECASE,
)

# Technical release tags, resolution, codecs, audio formats
_CLEAN_SCENE_TAGS_PATTERN = re.compile(
    r"(?i)\b(?:1080[pi]|720[pi]|2160[pi]|4k|uhd|bluray|blu-ray|bdrip|brrip|dvdrip|web-?dl|webrip|hdtv|h264|x264|h265|x265|hevc|avc|10bit|aac|ac3|dts(?:-hd)?|flac|mp3|repack|proper|unrated|remux)\b"
)


def sanitize_title_for_model(text: str) -> str:
    """Clean torrent title of bracketed groups, technical codecs, and file extensions.

    Preserves 4-digit release years (e.g. (1995), (2021)) since dates are a strong
    signal of legitimate cinema / TV releases.
    """
    if not text:
        return ""
    cleaned = _CLEAN_EXT_PATTERN.sub("", text)
    cleaned = re.sub(r"\[[^\]]*\]|\{[^\}]*\}", " ", cleaned)
    # Strip parentheses UNLESS they contain a 4-digit release year
    cleaned = re.sub(r"\((?!\b(?:19\d\d|20[0-2]\d)\b)[^\)]*\)", " ", cleaned)
    cleaned = _CLEAN_SCENE_TAGS_PATTERN.sub(" ", cleaned)
    cleaned = cleaned.replace(".", " ").replace("_", " ").replace("-", " ")
    return " ".join(cleaned.split())
