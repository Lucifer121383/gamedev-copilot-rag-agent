from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


VALID_DOMAINS = {"game", "enterprise"}


@dataclass(slots=True)
class LoadedSection:
    text: str
    source: str
    domain: str
    doc_type: str
    page: int | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    text: str
    source: str
    domain: str
    doc_type: str
    page: int | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Chunk":
        return cls(**value)


@dataclass(slots=True)
class SearchHit:
    chunk: Chunk
    score: float
    char_score: float = 0.0
    word_score: float = 0.0
    semantic_score: float = 0.0
    metadata_score: float = 0.0


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = ""


@dataclass(slots=True)
class BugDiagnosis:
    category: str
    severity: str
    platform: str
    module: str
    version: str | None
    error_code: str | None
    possible_causes: list[str]
    reproduction_steps: list[str]
    regression_tests: list[str]
    confidence: float
    needs_human_review: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

