import asyncio
import os
import threading

from fastapi import FastAPI, Query

from bitmagnet_classify.langid import detect
from bitmagnet_classify.models import (
    ClassificationVerdict,
    TorrentPayload,
    VerdictAction,
)
from bitmagnet_classify.nsfw import (
    is_keyword_nsfw,
    is_mainstream_allowed,
    is_model_nsfw,
)

app = FastAPI(title="bitmagnet classifier")
_classify_lock = threading.Lock()

EARLY_LANGUAGES: set[str] = set()

# Clean media extensions that hint toward non-adult documents or audio
# (e.g. books, comics, lossless music, software tools)
SAFE_DOC_AUDIO_EXTENSIONS = {
    "epub",
    "pdf",
    "mobi",
    "azw3",
    "djvu",
    "cbz",
    "cbr",
    "flac",
    "alac",
    "ape",
    "wav",
    "dsf",
    "dff",
}
SAFE_AUX_EXTENSIONS = {"txt", "nfo", "cue", "log", "jpg", "jpeg", "png", "sfv", "md5"}
VIDEO_EXTENSIONS = {
    "mp4",
    "mkv",
    "avi",
    "wmv",
    "mov",
    "flv",
    "m4v",
    "mpg",
    "mpeg",
    "webm",
    "ts",
}

# Borderline model confidence band where file paths are checked before making a final call
BORDERLINE_LOW = 0.35
BORDERLINE_HIGH = 0.70


def _classify_torrent_sync(
    payload: TorrentPayload,
    early_lang: list[str] | None = None,
) -> ClassificationVerdict:
    """Multi-tier torrent classifier.

    Design goals:
    - Zero/minimal model invocations on clean or obvious media.
    - Leverage file paths & extensions to disambiguate borderline / short torrent names.
    - Informative tagging for languages and NSFW signals without forcing actions like delete.
    - Optional early-return for configured language codes (skips downstream adult/ML checks).
    """
    tags: list[str] = ["external"]
    skip_langs = set(early_lang) if early_lang is not None else EARLY_LANGUAGES

    # Language detection on title
    detected_lang = detect(payload.name)
    if detected_lang != "other":
        tags.append(f"lang-{detected_lang}")
        if detected_lang in skip_langs:
            return ClassificationVerdict(
                action=VerdictAction.KEEP,
                tags=tags,
            )

    # Check file paths for distinct languages
    if payload.files:
        file_langs = {detect(f.path) for f in payload.files} - {"other", detected_lang}
        for fl in sorted(file_langs):
            tags.append(f"lang-{fl}")
            if fl in skip_langs:
                return ClassificationVerdict(
                    action=VerdictAction.KEEP,
                    tags=tags,
                )

    # Fast keyword matching for adult terms
    if not is_mainstream_allowed(payload.name):
        if is_keyword_nsfw(payload.name):
            tags.append("nsfw-keyword")
            return ClassificationVerdict(
                action=VerdictAction.KEEP,
                tags=tags,
            )

        # Check sample of file paths for adult keywords
        if payload.files:
            for f in payload.files[:10]:
                if is_keyword_nsfw(f.path):
                    tags.append("nsfw-keyword")
                    return ClassificationVerdict(
                        action=VerdictAction.KEEP,
                        tags=tags,
                    )

    # Early exit for non-video safe extensions (books, lossless audio, documents)
    file_extensions = {f.extension.lower() for f in payload.files if f.extension}
    if payload.extension:
        file_extensions.add(payload.extension.lower())

    if file_extensions:
        has_video = bool(file_extensions & VIDEO_EXTENSIONS)
        only_safe = file_extensions.issubset(
            SAFE_DOC_AUDIO_EXTENSIONS | SAFE_AUX_EXTENSIONS
        )

        if only_safe and not has_video:
            return ClassificationVerdict(
                action=VerdictAction.KEEP,
                tags=tags,
            )

    # Fallback to ML sequence classifier
    is_nsfw, prob = is_model_nsfw(payload.name)

    # Borderline ambiguity check: defer to file paths to confirm or refute
    if BORDERLINE_LOW <= prob <= BORDERLINE_HIGH and payload.files:
        file_scores = []
        for f in payload.files[:3]:
            # Skip checking auxiliary files like NFO or sample images
            if f.extension and f.extension.lower() in SAFE_AUX_EXTENSIONS:
                continue
            _, f_prob = is_model_nsfw(f.path)
            file_scores.append(f_prob)

        if file_scores:
            max_file_score = max(file_scores)
            if max_file_score >= 0.70:
                tags.append("nsfw-model")
                return ClassificationVerdict(
                    action=VerdictAction.KEEP,
                    tags=tags,
                )
            elif max_file_score < 0.35:
                is_nsfw = False

    if is_nsfw:
        tags.append("nsfw-model")

    return ClassificationVerdict(
        action=VerdictAction.KEEP,
        tags=tags,
    )


def classify_torrent(
    payload: TorrentPayload,
    early_lang: list[str] | None = None,
) -> ClassificationVerdict:
    """Thread-safe classification with strict serialization to prevent CPU exhaustion on low-spec hardware."""
    with _classify_lock:
        return _classify_torrent_sync(payload, early_lang=early_lang)


@app.post("/classify", response_model=ClassificationVerdict)
async def classify_endpoint(
    payload: TorrentPayload,
    early_lang: list[str] | None = Query(None),
) -> ClassificationVerdict:
    return await asyncio.to_thread(classify_torrent, payload, early_lang=early_lang)


def _get_systemd_listen_fd() -> int | None:
    """Check for systemd socket activation environment variables (SD_LISTEN_FDS_START = 3)."""
    import os

    listen_pid = os.environ.get("LISTEN_PID")
    listen_fds = os.environ.get("LISTEN_FDS")
    if listen_pid and listen_fds:
        try:
            if int(listen_pid) == os.getpid() and int(listen_fds) >= 1:
                return 3
        except ValueError:
            pass
    return None


def main(argv: list[str] | None = None) -> None:
    import argparse
    import os
    import uvicorn

    parser = argparse.ArgumentParser(
        description="bitmagnet classification service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", "0.0.0.0"),
        help="Bind socket to this host (ignored if systemd socket or --fd/--uds is used)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8080")),
        help="Bind socket to this port (ignored if systemd socket or --fd/--uds is used)",
    )
    parser.add_argument(
        "--fd",
        type=int,
        default=None,
        help="Bind to an already open file descriptor (e.g. passed from supervisor)",
    )
    parser.add_argument(
        "--uds",
        default=None,
        help="Bind to a UNIX domain socket path",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=os.environ.get("RELOAD", "false").lower() in ("1", "true", "yes"),
        help="Enable auto-reload on code change (recommended for dev mode)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("WEB_CONCURRENCY", "1")),
        help="Number of worker processes",
    )
    parser.add_argument(
        "--early-lang",
        nargs="*",
        default=[c for c in os.environ.get("EARLY_LANG", "").replace(",", " ").split() if c],
        help="Language codes (e.g. ja ko zh) that trigger immediate early-return skipping further checks",
    )

    args = parser.parse_args(argv)

    global EARLY_LANGUAGES
    EARLY_LANGUAGES = {
        code.strip().lower()
        for item in args.early_lang
        for part in item.replace(",", " ").split()
        if (code := part.strip())
    }

    systemd_fd = _get_systemd_listen_fd()
    fd = args.fd if args.fd is not None else systemd_fd

    run_kwargs: dict[str, object] = {
        "reload": args.reload,
        "workers": args.workers,
    }

    if fd is not None:
        run_kwargs["fd"] = fd
    elif args.uds:
        run_kwargs["uds"] = args.uds
    else:
        run_kwargs["host"] = args.host
        run_kwargs["port"] = args.port

    uvicorn.run("bitmagnet_classify.server:app", **run_kwargs)


if __name__ == "__main__":
    main()
