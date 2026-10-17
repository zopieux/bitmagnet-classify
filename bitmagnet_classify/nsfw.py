"""NSFW and mature content classification: fast keyword filter + ML model fallback."""

import os
import re
from functools import lru_cache
from pathlib import Path

# Silence HuggingFace transformers advisory warning about PyTorch not being installed
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from bitmagnet_classify.text import sanitize_title_for_model

_MODEL_DIR_DEFAULT = Path(__file__).resolve().parent / "models" / "e5_nsfw"
DEFAULT_THRESHOLD = 0.70

# Mainstream titles/franchises and non-adult terms that contain words like 'sex', 'hardcore', 'bang'
MAINSTREAM_ALLOWLIST = [
    r"sex\s+(?:and\s+the\s+city|education|life|lives\s+of\s+college\s+girls|pistols)",
    r"masters\s+of\s+sex",
    r"hardcore\s+history",
    r"(?:the\s+)?big\s+bang\s+theory",
    r"kiss\s+kiss\s+bang\s+bang",
    r"bang\s+bang\s+baby",
    r"chitty\s+chitty\s+bang\s+bang",
]

_MAINSTREAM_PATTERN = re.compile(
    r"(?i)(?:^|[\W_])(?:" + "|".join(MAINSTREAM_ALLOWLIST) + r")(?:$|[\W_])"
)

# Comprehensive adult and mature content keywords
KEYWORDS_XXX = [
    r"anal",
    r"ass(?:es)?",
    r"bangbros",
    r"bdsm",
    r"blowjobs?",
    r"bondage",
    r"boob\w*",
    r"brazzers",
    r"caribbeancom",
    r"cocks?",
    r"creampie",
    r"crossdress\w*",
    r"cuck\w*",
    r"cum\w*",
    r"deepthroat",
    r"dicks?",
    r"digitaldesire",
    r"dildo",
    r"erocomic",
    r"erotic\w*",
    r"evilangel",
    r"fansly\w*",
    r"fc2",
    r"femboy",
    r"fetish",
    r"footjob",
    r"fuck\w*",
    r"gangbang",
    r"gloryhole\w*",
    r"handjob",
    r"hardcore",
    r"hentai",
    r"heyzo",
    r"incest",
    r"interracial",
    r"jav",
    r"jizz\w*",
    r"just[\s_-]?for[\s_-]?fans?",
    r"kink\w*",
    r"ladyboy",
    r"legalporno",
    r"lesbian[sx]?",
    r"lewd\w*",
    r"masturbat\w*",
    r"milf\w*",
    r"naughtyamerica",
    r"netorare",
    r"nubile\w*",
    r"only[\s_-]?fans?",
    r"orgasm\w*",
    r"orgy",
    r"panties",
    r"panty",
    r"pegging",
    r"playboy",
    r"playmate",
    r"porn\w*",
    r"pov",
    r"pthc",
    r"ptsc",
    r"pussy",
    r"realitykings",
    r"seduc\w*",
    r"sex\w*",
    r"shemale",
    r"sissy",
    r"slut\w*",
    r"squirt\w*",
    r"step[\s_-]?(?:daughter|mom|sister|son|dad|father|mother)",
    r"strapon",
    r"stripper",
    r"submissive",
    r"succubus",
    r"swinger",
    r"taboo",
    r"teamskeet",
    r"threesome",
    r"tits?",
    r"tonguejob",
    r"topless",
    r"transsexual",
    r"twistys",
    r"vibrator",
    r"wank\w*",
    r"xxx",
]

_TOKEN_PATTERN = re.compile(
    r"(?i)(?:^|[\W_])(" + "|".join(KEYWORDS_XXX) + r")(?:$|[\W_])"
)
_WILDCARD_PATTERN = re.compile(r"(?i)(?:fuck|gloryhole|porn|wank|xxx)")


def is_mainstream_allowed(text: str) -> bool:
    """Check if title matches known mainstream media containing adult-like words."""
    if not text:
        return False
    # Normalize separators like dots, underscores, dashes to spaces for matching
    normalized = re.sub(r"[\W_]+", " ", text)
    return bool(_MAINSTREAM_PATTERN.search(normalized))


def is_keyword_nsfw(text: str) -> bool:
    """Fast regex check matching bitmagnet's core keywords.xxx."""
    if not text or is_mainstream_allowed(text):
        return False
    return bool(_TOKEN_PATTERN.search(text) or _WILDCARD_PATTERN.search(text))


@lru_cache(maxsize=1)
def _get_onnx_artifacts(model_dir: str | Path | None = None):
    """Load quantized ONNX backbone, classification head weights, and tokenizer."""
    path = Path(model_dir or os.environ.get("NSFW_ONNX_PATH") or _MODEL_DIR_DEFAULT)
    onnx_file = path / "backbone_int8.onnx"
    weights_file = path / "head_weights.npz"

    if not onnx_file.is_file() or not weights_file.is_file():
        raise FileNotFoundError(f"Missing ONNX classifier artifacts in {path}")

    # Limit threads for efficiency on embedded / low-power CPUs (e.g. Celeron J1900)
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = min(4, os.cpu_count() or 1)
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(onnx_file), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    weights = np.load(weights_file)

    return (
        session,
        tokenizer,
        weights["W1"],
        weights["b1"],
        weights["W2"],
        weights["b2"],
    )


def is_model_nsfw(
    text: str, model_path: str | None = None, threshold: float = DEFAULT_THRESHOLD
) -> tuple[bool, float]:
    """Check text using quantized ONNX multilingual-e5 sequence classifier."""
    if not text or not text.strip() or is_mainstream_allowed(text):
        return False, 0.0

    sanitized = sanitize_title_for_model(text)
    if not sanitized:
        return False, 0.0

    session, tokenizer, W1, b1, W2, b2 = _get_onnx_artifacts(model_path)
    enc = tokenizer(
        [f"query: {sanitized}"],
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=96,
    )
    ort_in = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    last_hidden = session.run(None, ort_in)[0]
    # Fast NumPy mean pooling + L2 normalization (standard for E5 representations)
    mask = enc["attention_mask"][:, :, np.newaxis]
    pooled = (last_hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
    pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)

    # 2-layer MLP head + sigmoid
    h = np.maximum(0, pooled @ W1.T + b1)
    logits = (h @ W2.T + b2).squeeze(-1)
    prob = float(1.0 / (1.0 + np.exp(-logits))[0])

    return prob >= threshold, prob


def check_nsfw(
    text: str, model_path: str | None = None, threshold: float = DEFAULT_THRESHOLD
) -> tuple[bool, str]:
    """Three-step NSFW check:

    1. Exemption check for mainstream titles (e.g. Sex and the City).
    2. Fast regex keyword filter (free CPU cost).
    3. Fallback to quantized ONNX multilingual-e5 sequence classifier.

    Returns:
        (is_nsfw, reason) where reason is "keyword", "model", or ""
    """
    if is_mainstream_allowed(text):
        return False, ""

    if is_keyword_nsfw(text):
        return True, "keyword"

    nsfw, _ = is_model_nsfw(text, model_path=model_path, threshold=threshold)
    if nsfw:
        return True, "model"

    return False, ""
