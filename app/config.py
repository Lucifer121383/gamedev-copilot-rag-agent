from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


@dataclass(slots=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("RAG_DATA_DIR", PROJECT_ROOT / "data"))
    )
    storage_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("RAG_STORAGE_DIR", PROJECT_ROOT / "storage")
        )
    )
    embedding_backend: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_BACKEND", "hybrid_tfidf"
        ).strip().lower()
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
        ).strip()
    )
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 420))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 80))
    min_retrieval_score: float = field(
        default_factory=lambda: _env_float("MIN_RETRIEVAL_SCORE", 0.08)
    )
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
    )
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY", "").strip()
    )
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "").strip())
    llm_temperature: float = field(
        default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.1)
    )
    llm_timeout_seconds: int = field(
        default_factory=lambda: _env_int("LLM_TIMEOUT_SECONDS", 45)
    )
    llm_max_tokens: int = field(
        default_factory=lambda: _env_int("LLM_MAX_TOKENS", 900)
    )
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 15))
    max_agent_steps: int = field(
        default_factory=lambda: _env_int("MAX_AGENT_STEPS", 10)
    )

    @property
    def database_path(self) -> Path:
        return self.storage_dir / "gamedev_copilot.db"

    def domain_dir(self, domain: str) -> Path:
        return self.data_dir / domain

    def upload_dir(self, domain: str) -> Path:
        return self.data_dir / "uploads" / domain

    @property
    def llm_enabled(self) -> bool:
        placeholders = ("请把", "你的", "replace", "your_")
        key_is_real = bool(self.llm_api_key) and not any(
            marker in self.llm_api_key.lower() for marker in placeholders
        )
        return bool(self.llm_base_url and self.llm_model and key_is_real)

    def ensure_directories(self) -> None:
        for domain in ("game", "enterprise"):
            self.domain_dir(domain).mkdir(parents=True, exist_ok=True)
            self.upload_dir(domain).mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        if self.chunk_size < 100:
            raise ValueError("CHUNK_SIZE 不能小于 100")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP 必须大于等于0且小于CHUNK_SIZE")
        if self.embedding_backend not in {"hybrid_tfidf", "sentence_transformers"}:
            raise ValueError(
                "EMBEDDING_BACKEND 只支持 hybrid_tfidf 或 sentence_transformers"
            )
        if not 0 <= self.min_retrieval_score <= 1:
            raise ValueError("MIN_RETRIEVAL_SCORE 必须在0到1之间")
        if self.max_agent_steps < 4:
            raise ValueError("MAX_AGENT_STEPS 不能小于4")

