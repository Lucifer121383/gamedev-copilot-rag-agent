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
    bm25_score: float = 0.0
    dense_score: float = 0.0
    exact_score: float = 0.0
    fusion_score: float = 0.0
    rerank_score: float = 0.0
    char_score: float = 0.0
    word_score: float = 0.0
    semantic_score: float = 0.0
    metadata_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "bm25_score": self.bm25_score,
            "dense_score": self.dense_score,
            "exact_score": self.exact_score,
            "fusion_score": self.fusion_score,
            "rerank_score": self.rerank_score,
            "char_score": self.char_score,
            "word_score": self.word_score,
            "semantic_score": self.semantic_score,
            "metadata_score": self.metadata_score,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SearchHit":
        raw = dict(value)
        raw["chunk"] = Chunk.from_dict(raw["chunk"])
        return cls(**raw)


@dataclass(slots=True)
class EvidenceAssessment:
    sufficient: bool
    confidence: float
    reason: str
    source_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ToolCall":
        return cls(**value)


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

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BugDiagnosis":
        return cls(**value)
