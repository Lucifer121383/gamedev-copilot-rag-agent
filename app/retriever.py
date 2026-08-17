from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.models import Chunk, SearchHit, VALID_DOMAINS


_EXACT_TOKEN_RE = re.compile(
    r"(?:BUG|INC|ERR|AUTH|MATCH|DB|GW|E)[-_][A-Z0-9-]{2,}|(?<!\d)\d+\.\d+(?:\.\d+)?(?!\d)",
    re.IGNORECASE,
)
_PLATFORM_WORDS = ("android", "ios", "windows", "linux", "web", "客户端", "服务端")
_MODULE_WORDS = (
    "装备",
    "背包",
    "资源加载",
    "登录",
    "订单",
    "支付",
    "数据库",
    "网关",
    "匹配",
    "战斗",
)


class HybridRetriever:
    """面向研发文档的混合检索器。

    默认并行使用中文字符TF-IDF、英文/错误码词项TF-IDF和元数据精确匹配。
    可选sentence-transformers语义向量。所有分量都会返回，便于调试排序原因。
    """

    def __init__(self, storage_dir: Path, backend: str, model_name: str) -> None:
        self.storage_dir = storage_dir
        self.backend = backend
        self.model_name = model_name
        self.chunks: list[Chunk] = []
        self._char_vectorizer: TfidfVectorizer | None = None
        self._word_vectorizer: TfidfVectorizer | None = None
        self._char_matrix: Any = None
        self._word_matrix: Any = None
        self._semantic_matrix: np.ndarray | None = None
        self._semantic_model: Any = None

    @property
    def chunks_path(self) -> Path:
        return self.storage_dir / "chunks.json"

    @property
    def metadata_path(self) -> Path:
        return self.storage_dir / "index_metadata.json"

    @property
    def semantic_path(self) -> Path:
        return self.storage_dir / "semantic_embeddings.npy"

    def rebuild(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("没有可建立索引的文本块")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.chunks = chunks
        self.chunks_path.write_text(
            json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.metadata_path.write_text(
            json.dumps(
                {
                    "backend": self.backend,
                    "model_name": self.model_name,
                    "chunk_count": len(chunks),
                    "domains": sorted({chunk.domain for chunk in chunks}),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self._fit_indices(persist_semantic=True)

    def load(self) -> bool:
        if not self.chunks_path.exists():
            return False
        raw = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        self.chunks = [Chunk.from_dict(item) for item in raw]
        if not self.chunks:
            return False
        self._fit_indices(persist_semantic=False)
        return True

    def _fit_indices(self, persist_semantic: bool) -> None:
        texts = [self._searchable_text(chunk) for chunk in self.chunks]
        self._char_vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 4), min_df=1, sublinear_tf=True, norm="l2"
        )
        self._char_matrix = self._char_vectorizer.fit_transform(texts)

        self._word_vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)[A-Za-z0-9_.:-]{2,}",
            lowercase=True,
            sublinear_tf=True,
            norm="l2",
        )
        try:
            self._word_matrix = self._word_vectorizer.fit_transform(texts)
        except ValueError:
            self._word_vectorizer = None
            self._word_matrix = None

        self._semantic_matrix = None
        if self.backend == "sentence_transformers":
            self._load_semantic_model()
            encode_document = getattr(
                self._semantic_model, "encode_document", self._semantic_model.encode
            )
            self._semantic_matrix = np.asarray(
                encode_document(texts, normalize_embeddings=True, show_progress_bar=False),
                dtype=np.float32,
            )
            if persist_semantic:
                np.save(self.semantic_path, self._semantic_matrix)

    @staticmethod
    def _searchable_text(chunk: Chunk) -> str:
        metadata = " ".join(str(value) for value in chunk.metadata.values())
        return f"{chunk.source} {chunk.doc_type} {metadata}\n{chunk.text}"

    def _load_semantic_model(self) -> None:
        if self._semantic_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "语义检索依赖尚未安装，请执行 pip install -r requirements-semantic.txt"
            ) from exc
        self._semantic_model = SentenceTransformer(self.model_name)

    @staticmethod
    def _metadata_score(query: str, chunk: Chunk) -> float:
        query_lower = query.lower()
        haystack = HybridRetriever._searchable_text(chunk).lower()
        score = 0.0
        exact_tokens = {token.lower() for token in _EXACT_TOKEN_RE.findall(query)}
        if exact_tokens:
            matched = sum(token in haystack for token in exact_tokens)
            score += 0.55 * matched / len(exact_tokens)
            if chunk.doc_type in {"crash_logs", "error_logs"} and matched:
                score += 0.15
        requested_platforms = [word for word in _PLATFORM_WORDS if word in query_lower]
        if requested_platforms and any(word in haystack for word in requested_platforms):
            score += 0.2
        requested_modules = [word for word in _MODULE_WORDS if word in query_lower]
        if requested_modules and any(word in haystack for word in requested_modules):
            score += 0.25
        return min(score, 1.0)

    @staticmethod
    def _matches_filters(chunk: Chunk, filters: dict[str, str] | None) -> bool:
        if not filters:
            return True
        for key, expected in filters.items():
            if not expected:
                continue
            if key == "doc_type":
                actual = chunk.doc_type
            elif key == "source":
                actual = chunk.source
            else:
                actual = str(chunk.metadata.get(key, ""))
            if expected.lower() not in actual.lower():
                return False
        return True

    def search(
        self,
        query: str,
        *,
        domain: str,
        top_k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[SearchHit]:
        query = query.strip()
        if domain not in VALID_DOMAINS:
            raise ValueError(f"不支持的工作空间: {domain}")
        if top_k < 1:
            raise ValueError("top_k必须大于0")
        if not query or not self.chunks or self._char_matrix is None:
            return []

        candidates = [
            index
            for index, chunk in enumerate(self.chunks)
            if chunk.domain == domain and self._matches_filters(chunk, filters)
        ]
        if not candidates:
            return []

        assert self._char_vectorizer is not None
        char_query = self._char_vectorizer.transform([query])
        char_scores = (self._char_matrix @ char_query.T).toarray().ravel()

        word_scores = np.zeros(len(self.chunks), dtype=np.float32)
        if self._word_vectorizer is not None and self._word_matrix is not None:
            word_query = self._word_vectorizer.transform([query])
            word_scores = (self._word_matrix @ word_query.T).toarray().ravel()

        semantic_scores = np.zeros(len(self.chunks), dtype=np.float32)
        if self._semantic_matrix is not None:
            encode_query = getattr(
                self._semantic_model, "encode_query", self._semantic_model.encode
            )
            query_vector = np.asarray(
                encode_query([query], normalize_embeddings=True, show_progress_bar=False)[0],
                dtype=np.float32,
            )
            semantic_scores = np.maximum(self._semantic_matrix @ query_vector, 0)

        scored: list[SearchHit] = []
        exact_query_tokens = {
            token.lower() for token in _EXACT_TOKEN_RE.findall(query)
        }
        has_error_code = any("-" in token for token in exact_query_tokens)
        for index in candidates:
            chunk = self.chunks[index]
            metadata_score = self._metadata_score(query, chunk)
            if self.backend == "sentence_transformers":
                combined = (
                    0.3 * float(char_scores[index])
                    + 0.15 * float(word_scores[index])
                    + 0.45 * float(semantic_scores[index])
                    + 0.1 * metadata_score
                )
            else:
                combined = (
                    0.65 * float(char_scores[index])
                    + 0.25 * float(word_scores[index])
                    + 0.1 * metadata_score
                )
            if has_error_code and chunk.doc_type in {"crash_logs", "error_logs"}:
                combined += 0.06
            scored.append(
                SearchHit(
                    chunk=chunk,
                    score=round(float(combined), 6),
                    char_score=round(float(char_scores[index]), 6),
                    word_score=round(float(word_scores[index]), 6),
                    semantic_score=round(float(semantic_scores[index]), 6),
                    metadata_score=round(metadata_score, 6),
                )
            )
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[: min(top_k, len(scored))]
