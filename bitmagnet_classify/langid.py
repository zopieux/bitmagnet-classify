"""Language detection using Unicode script rules and fastText fallback.

Outputs ISO 639 language codes (e.g. 'en', 'fr', 'ja', 'ko', 'zh') or 'other'.
"""

import os
import re
from functools import lru_cache

import fasttext

from bitmagnet_classify.text import sanitize_title_for_model

MODEL_PATH_DEFAULT = "lid.176.ftz"
THRESHOLD = 0.65
MIN_SCRIPT_CHARS = 3


def _is_kana(c: int) -> bool:
    return 0x3040 <= c <= 0x30FF or 0x31F0 <= c <= 0x31FF or 0xFF66 <= c <= 0xFF9F


def _is_han(c: int) -> bool:
    return (
        0x4E00 <= c <= 0x9FFF
        or 0x3400 <= c <= 0x4DBF
        or 0x20000 <= c <= 0x2A6DF
        or 0xF900 <= c <= 0xFAFF
    )


def _is_hangul(c: int) -> bool:
    # Hangul Syllables, Hangul Jamo, Hangul Compatibility Jamo, Hangul Jamo Extended-A/B
    return (
        0xAC00 <= c <= 0xD7AF
        or 0x1100 <= c <= 0x11FF
        or 0x3130 <= c <= 0x318F
        or 0xA960 <= c <= 0xA97F
        or 0xD7B0 <= c <= 0xD7FF
    )


def _is_cyrillic(c: int) -> bool:
    return 0x0400 <= c <= 0x052F


@lru_cache(maxsize=1)
def get_model(model_path: str | None = None):
    path = model_path or os.environ.get("FASTTEXT_MODEL_PATH") or MODEL_PATH_DEFAULT
    return fasttext.load_model(path)


def _ft(text: str, model_path: str | None = None) -> tuple[str, float]:
    # Low-level call avoids the NumPy 2 `copy=False` failure in predict().
    sanitized = sanitize_title_for_model(text)
    # Strip numbers to prevent arbitrary digit sequences / IDs from hallucinating language codes
    sanitized = re.sub(r"\d+", " ", sanitized)
    sanitized = " ".join(sanitized.split())
    if not sanitized or not any(c.isalpha() for c in sanitized):
        return "other", 0.0
    model = get_model(model_path)
    res = model.f.predict(sanitized, 1, 0.0, "strict")
    if not res:
        return "other", 0.0
    ((prob, label),) = res
    return label.removeprefix("__label__"), prob


def detect(text: str, model_path: str | None = None) -> str:
    kana = han = hangul = cyr = letters = 0
    for ch in text:
        if not ch.isalpha():
            continue
        letters += 1
        c = ord(ch)
        if _is_kana(c):
            kana += 1
        elif _is_hangul(c):
            hangul += 1
        elif _is_han(c):
            han += 1
        elif _is_cyrillic(c):
            cyr += 1

    if letters == 0:
        return "other"
    if kana:
        return "ja"
    if hangul:
        return "ko"
    if han:
        return "zh"

    lang, conf = _ft(text, model_path=model_path)
    return lang if conf >= THRESHOLD else "other"
