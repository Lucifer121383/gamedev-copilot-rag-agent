from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if not value:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(slots=True)
class Settings:
    """IncidentCopilot运行配置。

    默认配置完全离线可运行：BM25 + Hashing向量 + FAISS + 轻量重排。
    面试演示或生产实验可以切换到BGE Embedding和CrossEncoder Reranker。
    """

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
        default_factory=lambda: os.getenv("EMBEDDING_BACKEND", "hashing").strip().lower()
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
        ).strip()
    )
    vector_backend: str = field(
        default_factory=lambda: os.getenv("VECTOR_BACKEND", "faiss").strip().lower()
    )
    reranker_backend: str = field(
        default_factory=lambda: os.getenv("RERANKER_BACKEND", "lightweight").strip().lower()
    )
    reranker_model: str = field(
        default_factory=lambda: os.getenv(
            "RERANKER_MODEL", "BAAI/bge-reranker-base"
        ).strip()
    )
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 420))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 80))
    initial_recall_k: int = field(
        default_factory=lambda: _env_int("INITIAL_RECALL_K", 24)
    )
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))
    min_retrieval_score: float = field(
        default_factory=lambda: _env_float("MIN_RETRIEVAL_SCORE", 0.4)
    )
    min_evidence_sources: int = field(
        default_factory=lambda: _env_int("MIN_EVIDENCE_SOURCES", 1)
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
    llm_max_retries: int = field(
        default_factory=lambda: _env_int("LLM_MAX_RETRIES", 2)
    )

    api_auth_enabled: bool = field(
        default_factory=lambda: _env_bool("API_AUTH_ENABLED", False)
    )
    api_key: str = field(default_factory=lambda: os.getenv("APP_API_KEY", "").strip())
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: _env_csv(
            "CORS_ORIGINS", ("http://127.0.0.1:8010", "http://localhost:8010")
        )
    )
    rate_limit_per_minute: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_PER_MINUTE", 60)
    )
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 15))
    max_agent_steps: int = field(
        default_factory=lambda: _env_int("MAX_AGENT_STEPS", 16)
    )
    max_history_messages: int = field(
        default_factory=lambda: _env_int("MAX_HISTORY_MESSAGES", 10)
    )

    database_url_override: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "").strip()
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").strip().upper()
    )
    json_logs: bool = field(default_factory=lambda: _env_bool("JSON_LOGS", True))

    def __post_init__(self) -> None:
        # Backward compatibility for the 1.x configuration name.
        if self.embedding_backend == "hybrid_tfidf":
            self.embedding_backend = "hashing"

    @property
    def database_path(self) -> Path:
        return self.storage_dir / "incident_copilot.db"

    @property
    def database_url(self) -> str:
        if self.database_url_override:
            return self.database_url_override
        return f"sqlite:///{self.database_path.as_posix()}"

    @property
    def checkpoint_path(self) -> Path:
        return self.storage_dir / "langgraph_checkpoints.db"

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

    @property
    def semantic_enabled(self) -> bool:
        return self.embedding_backend in {"hashing", "sentence_transformers"}

    def ensure_directories(self) -> None:
        for domain in ("game", "enterprise"):
            self.domain_dir(domain).mkdir(parents=True, exist_ok=True)
            self.upload_dir(domain).mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        if self.chunk_size < 100:
            raise ValueError("CHUNK_SIZE不能小于100")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP必须大于等于0且小于CHUNK_SIZE")
        if self.embedding_backend not in {"disabled", "hashing", "sentence_transformers"}:
            raise ValueError(
                "EMBEDDING_BACKEND只支持disabled、hashing或sentence_transformers"
            )
        if self.vector_backend not in {"faiss", "numpy"}:
            raise ValueError("VECTOR_BACKEND只支持faiss或numpy")
        if self.reranker_backend not in {"disabled", "lightweight", "cross_encoder"}:
            raise ValueError(
                "RERANKER_BACKEND只支持disabled、lightweight或cross_encoder"
            )
        if not 0 <= self.min_retrieval_score <= 1:
            raise ValueError("MIN_RETRIEVAL_SCORE必须在0到1之间")
        if self.initial_recall_k < 5:
            raise ValueError("INITIAL_RECALL_K不能小于5")
        if self.rrf_k < 1:
            raise ValueError("RRF_K必须大于0")
        if self.max_agent_steps < 8:
            raise ValueError("MAX_AGENT_STEPS不能小于8")
        if self.api_auth_enabled and len(self.api_key) < 12:
            raise ValueError("启用API鉴权时APP_API_KEY至少需要12个字符")
        if self.rate_limit_per_minute < 1:
            raise ValueError("RATE_LIMIT_PER_MINUTE必须大于0")
