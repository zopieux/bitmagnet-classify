from dataclasses import dataclass, field
from enum import StrEnum


class VerdictAction(StrEnum):
    KEEP = "keep"
    DELETE = "delete"
    UNMATCHED = "unmatched"


@dataclass
class TorrentFile:
    path: str
    size: int
    extension: str | None = None


@dataclass
class TorrentHint:
    contentType: str | None = None


@dataclass
class TorrentPayload:
    infoHash: str
    name: str
    size: int
    extension: str | None = None
    files: list[TorrentFile] = field(default_factory=list)
    hint: TorrentHint | None = None


@dataclass
class ClassificationVerdict:
    action: VerdictAction = VerdictAction.KEEP
    contentType: str | None = None
    tags: list[str] = field(default_factory=list)
    releaseGroup: str | None = None
