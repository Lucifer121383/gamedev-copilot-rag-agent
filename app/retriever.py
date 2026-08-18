from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import HashingVectorizer

from app.config import Settings
from app.models import Chunk, SearchHit, VALID_DOMAINS


_EXACT_TOKEN_RE = re.compile(
    r"[A-Z][A-Z0-9]{0,15}[-_][A-Z0-9-]{2,}|"
    r"(?<!\d)\d+\.\d+(?:\.\d+)?(?!\d)",
    re.IGNORECASE,
)
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9_.:-]{2,}")
_CJK_RE = re.compile(r"[\u3400-\u9fff]+")
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
_OPERATIONAL_FIELD_ALIASES = {
    "pool_active": "连接池活跃连接 活跃连接",
    "pool_idle": "连接池空闲连接 空闲连接",
    "pool_max": "连接池最大连接 最大连接",
    "error_code": "错误码",
    "payment_timeout": "支付超时",
    "endpoint": "接口地址",
}
_INDEX_SCHEMA_VERSION = 3


def _tokenize(text: str) -> list[str]:
    """适用于中英文研发资料的轻量BM25分词。

    英文、版本号和错误码保留为完整词项；中文使用单字与二元组，避免项目
    依赖额外分词词典，同时保留对短错误描述的召回能力。
    """

    lowered = text.lower()
    tokens = _LATIN_TOKEN_RE.findall(lowered)
    for segment in _CJK_RE.findall(lowered):
        tokens.extend(segment)
        tokens.extend(segment[index : index + 2] for index in range(len(segment) - 1))
    return tokens or [lowered.strip()]


def _normalize(values: np.ndarray, indices: list[int]) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float32)
    if not indices:
        return result
    selected = np.asarray([max(float(values[index]), 0.0) for index in indices])
    maximum = float(selected.max(initial=0.0))
    if maximum > 0:
        for index in indices:
            result[index] = max(float(values[index]), 0.0) / maximum
    return result


class HybridRetriever:
    """BM25、稠密向量、精确元数据、RRF融合与Rerank检索器。"""

    def __init__(self, storage_dir: Path, settings: Settings) -> None:
        self.storage_dir = storage_dir
        self.settings = settings
        self.backend = settings.embedding_backend
        self.model_name = settings.embedding_model
        self.chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._dense_matrix: np.ndarray | None = None
        self._dense_model: Any = None
        self._hashing_vectorizer: HashingVectorizer | None = None
        self._faiss_index: Any = None
        self._reranker: Any = None
        self._metadata: dict[str, Any] = {}

    @property
    def chunks_path(self) -> Path:
        return self.storage_dir / "chunks.json"

    @property
    def metadata_path(self) -> Path:
        return self.storage_dir / "index_metadata.json"

    @property
    def dense_path(self) -> Path:
        return self.storage_dir / "dense_embeddings.npy"

    @property
    def faiss_path(self) -> Path:
        return self.storage_dir / "faiss.index"

    @staticmethod
    def _searchable_text(chunk: Chunk) -> str:
        metadata = " ".join(str(value) for value in chunk.metadata.values())
        text = f"{chunk.source} {chunk.doc_type} {metadata}\n{chunk.text}"
        aliases: list[str] = []
        for field, chinese_alias in _OPERATIONAL_FIELD_ALIASES.items():
            match = re.search(
                rf"\b{re.escape(field)}\s*=\s*([^\s]+)",
                chunk.text,
                re.IGNORECASE,
            )
            if not match:
                continue
            value = match.group(1).strip('"')
            aliases.extend(
                (f"{chinese_alias} {value}", f"{chinese_alias}为{value}")
            )
            if field == "pool_active":
                aliases.append(f"活跃连接达到{value}")
        return f"{text}\n{' '.join(aliases)}" if aliases else text

    @staticmethod
    def _corpus_hash(chunks: list[Chunk]) -> str:
        payload = "\n".join(
            f"{chunk.chunk_id}|{chunk.domain}|{chunk.source}|{chunk.text}" for chunk in chunks
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def rebuild(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("没有可建立索引的文本块")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.chunks = chunks
        corpus_hash = self._corpus_hash(chunks)
        self.chunks_path.write_text(
            json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._fit_sparse()
        self._build_dense(persist=True)
        self._metadata = {
            "schema_version": _INDEX_SCHEMA_VERSION,
            "embedding_backend": self.settings.embedding_backend,
            "embedding_model": self.settings.embedding_model,
            "vector_backend": self.settings.vector_backend,
            "reranker_backend": self.settings.reranker_backend,
            "reranker_model": self.settings.reranker_model,
            "chunk_count": len(chunks),
            "domains": sorted({chunk.domain for chunk in chunks}),
            "corpus_hash": corpus_hash,
            "dense_dimension": (
                int(self._dense_matrix.shape[1]) if self._dense_matrix is not None else 0
            ),
        }
        self.metadata_path.write_text(
            json.dumps(self._metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load(self) -> bool:
        if not self.chunks_path.exists():
            return False
        raw = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        self.chunks = [Chunk.from_dict(item) for item in raw]
        if not self.chunks:
            return False
        if self.metadata_path.exists():
            self._metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        expected_hash = self._corpus_hash(self.chunks)
        compatible = (
            self._metadata.get("schema_version") == _INDEX_SCHEMA_VERSION
            and self._metadata.get("corpus_hash") == expected_hash
            and self._metadata.get("embedding_backend") == self.settings.embedding_backend
            and self._metadata.get("embedding_model") == self.settings.embedding_model
            and self._metadata.get("vector_backend") == self.settings.vector_backend
        )
        self._fit_sparse()
        if compatible and self._load_dense_artifacts():
            return True
        self.rebuild(self.chunks)
        return True

    def _fit_sparse(self) -> None:
        corpus = [_tokenize(self._searchable_text(chunk)) for chunk in self.chunks]
        self._bm25 = BM25Okapi(corpus)

    def _load_dense_model(self) -> None:
        if self._dense_model is not None or self.backend != "sentence_transformers":
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "BGE语义检索依赖尚未安装，请执行pip install -r requirements-semantic.txt"
            ) from exc
        self._dense_model = SentenceTransformer(self.settings.embedding_model)

    def _encode_documents(self, texts: list[str]) -> np.ndarray:
        if self.backend == "disabled":
            return np.empty((len(texts), 0), dtype=np.float32)
        if self.backend == "hashing":
            if self._hashing_vectorizer is None:
                self._hashing_vectorizer = HashingVectorizer(
                    analyzer="char",
                    ngram_range=(2, 4),
                    n_features=768,
                    alternate_sign=False,
                    norm="l2",
                )
            return np.asarray(self._hashing_vectorizer.transform(texts).toarray(), dtype=np.float32)
        self._load_dense_model()
        encode = getattr(self._dense_model, "encode_document", self._dense_model.encode)
        return np.asarray(
            encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )

    def _encode_query(self, query: str) -> np.ndarray:
        if self.backend == "hashing":
            if self._hashing_vectorizer is None:
                self._encode_documents([query])
            assert self._hashing_vectorizer is not None
            return np.asarray(
                self._hashing_vectorizer.transform([query]).toarray()[0], dtype=np.float32
            )
        self._load_dense_model()
        encode = getattr(self._dense_model, "encode_query", self._dense_model.encode)
        return np.asarray(
            encode([query], normalize_embeddings=True, show_progress_bar=False)[0],
            dtype=np.float32,
        )

    def _build_dense(self, *, persist: bool) -> None:
        self._dense_matrix = None
        self._faiss_index = None
        if self.backend == "disabled":
            for path in (self.dense_path, self.faiss_path):
                if path.exists():
                    path.unlink()
            return
        texts = [self._searchable_text(chunk) for chunk in self.chunks]
        matrix = np.ascontiguousarray(self._encode_documents(texts), dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-12)
        self._dense_matrix = matrix
        if self.settings.vector_backend == "faiss":
            try:
                import faiss
            except ImportError as exc:
                raise RuntimeError("FAISS尚未安装，请重新安装requirements.txt") from exc
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            self._faiss_index = index
            if persist:
                faiss.write_index(index, str(self.faiss_path))
        if persist:
            np.save(self.dense_path, matrix)

    def _load_dense_artifacts(self) -> bool:
        if self.backend == "disabled":
            self._dense_matrix = None
            self._faiss_index = None
            return True
        if not self.dense_path.exists():
            return False
        matrix = np.load(self.dense_path)
        if matrix.ndim != 2 or matrix.shape[0] != len(self.chunks):
            return False
        self._dense_matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        if self.backend == "hashing":
            self._hashing_vectorizer = HashingVectorizer(
                analyzer="char",
                ngram_range=(2, 4),
                n_features=self._dense_matrix.shape[1],
                alternate_sign=False,
                norm="l2",
            )
        if self.settings.vector_backend == "faiss":
            if not self.faiss_path.exists():
                return False
            try:
                import faiss
            except ImportError as exc:
                raise RuntimeError("FAISS尚未安装，请重新安装requirements.txt") from exc
            self._faiss_index = faiss.read_index(str(self.faiss_path))
            if self._faiss_index.ntotal != len(self.chunks):
                return False
        return True

    @staticmethod
    def _metadata_score(query: str, chunk: Chunk) -> float:
        query_lower = query.lower()
        haystack = HybridRetriever._searchable_text(chunk).lower()
        score = 0.0
        exact_tokens = {token.lower() for token in _EXACT_TOKEN_RE.findall(query)}
        if exact_tokens:
            matched = sum(token in haystack for token in exact_tokens)
            score += 0.6 * matched / len(exact_tokens)
            if chunk.doc_type in {"crash_logs", "error_logs"} and matched:
                score += 0.28
        requested_platforms = [word for word in _PLATFORM_WORDS if word in query_lower]
        if requested_platforms and any(word in haystack for word in requested_platforms):
            score += 0.14
        requested_modules = [word for word in _MODULE_WORDS if word in query_lower]
        if requested_modules and any(word in haystack for word in requested_modules):
            score += 0.18
        operational_patterns = (
            (r"(?:连接池)?活跃连接(?:达到|为|=)?\s*(\d+)", "pool_active"),
            (r"(?:连接池)?空闲(?:连接)?(?:为|=)?\s*(\d+)", "pool_idle"),
            (r"(?:连接池)?最大连接(?:达到|为|=)?\s*(\d+)", "pool_max"),
        )
        for pattern, field in operational_patterns:
            match = re.search(pattern, query_lower)
            if match and re.search(
                rf"\b{field}\s*=\s*{re.escape(match.group(1))}\b",
                chunk.text,
                re.IGNORECASE,
            ):
                score += 0.35
        return min(score, 1.0)

    @staticmethod
    def _matches_filters(chunk: Chunk, filters: dict[str, str] | None) -> bool:
        if not filters:
            return True
        for key, expected in filters.items():
            if not expected:
                continue
            actual = (
                chunk.doc_type
                if key == "doc_type"
                else chunk.source
                if key == "source"
                else str(chunk.metadata.get(key, ""))
            )
            if expected.lower() not in actual.lower():
                return False
        return True

    @staticmethod
    def _rank_map(values: np.ndarray, candidates: list[int]) -> dict[int, int]:
        ordered = sorted(candidates, key=lambda index: float(values[index]), reverse=True)
        return {index: rank for rank, index in enumerate(ordered, start=1)}

    def _dense_scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        if self._dense_matrix is None:
            return scores
        query_vector = np.ascontiguousarray(self._encode_query(query), dtype=np.float32)
        query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
        if self.settings.vector_backend == "faiss" and self._faiss_index is not None:
            distances, indices = self._faiss_index.search(
                query_vector.reshape(1, -1), len(self.chunks)
            )
            for distance, index in zip(distances[0], indices[0], strict=False):
                if index >= 0:
                    scores[int(index)] = max(float(distance), 0.0)
            return scores
        return np.maximum(self._dense_matrix @ query_vector, 0.0).astype(np.float32)

    def _lightweight_rerank(self, query: str, indices: list[int], fusion: np.ndarray) -> np.ndarray:
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        query_tokens = set(_tokenize(query))
        exact_tokens = {token.lower() for token in _EXACT_TOKEN_RE.findall(query)}
        for index in indices:
            document_tokens = set(_tokenize(self._searchable_text(self.chunks[index])))
            coverage = len(query_tokens & document_tokens) / max(len(query_tokens), 1)
            exact_bonus = 0.0
            if exact_tokens:
                haystack = self._searchable_text(self.chunks[index]).lower()
                exact_bonus = sum(token in haystack for token in exact_tokens) / len(exact_tokens)
            scores[index] = min(1.0, 0.6 * float(fusion[index]) + 0.3 * coverage + 0.1 * exact_bonus)
        return scores

    def _cross_encoder_rerank(self, query: str, indices: list[int]) -> np.ndarray:
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "CrossEncoder重排依赖尚未安装，请执行pip install -r requirements-semantic.txt"
                ) from exc
            self._reranker = CrossEncoder(self.settings.reranker_model)
        pairs = [(query, self._searchable_text(self.chunks[index])) for index in indices]
        raw = np.asarray(self._reranker.predict(pairs), dtype=np.float32).reshape(-1)
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for index, value in zip(indices, raw, strict=False):
            clipped = max(min(float(value), 30.0), -30.0)
            scores[index] = 1.0 / (1.0 + math.exp(-clipped))
        return scores

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
        if not query or not self.chunks or self._bm25 is None:
            return []

        candidates = [
            index
            for index, chunk in enumerate(self.chunks)
            if chunk.domain == domain and self._matches_filters(chunk, filters)
        ]
        if not candidates:
            return []

        raw_bm25 = np.asarray(self._bm25.get_scores(_tokenize(query)), dtype=np.float32)
        bm25 = _normalize(raw_bm25, candidates)
        raw_dense = self._dense_scores(query)
        dense = _normalize(raw_dense, candidates)
        exact = np.zeros(len(self.chunks), dtype=np.float32)
        coverage = np.zeros(len(self.chunks), dtype=np.float32)
        query_tokens = set(_tokenize(query))
        for index in candidates:
            exact[index] = self._metadata_score(query, self.chunks[index])
            document_tokens = set(_tokenize(self._searchable_text(self.chunks[index])))
            coverage[index] = len(query_tokens & document_tokens) / max(
                len(query_tokens), 1
            )

        bm25_ranks = self._rank_map(bm25, candidates)
        dense_ranks = self._rank_map(dense, candidates)
        exact_ranks = self._rank_map(exact, candidates)
        rrf = np.zeros(len(self.chunks), dtype=np.float32)
        for index in candidates:
            rrf[index] = (
                (0.45 / (self.settings.rrf_k + bm25_ranks[index]) if bm25[index] > 0 else 0)
                + (0.4 / (self.settings.rrf_k + dense_ranks[index]) if raw_dense[index] > 0.01 else 0)
                + (0.15 / (self.settings.rrf_k + exact_ranks[index]) if exact[index] > 0 else 0)
            )
        fusion = _normalize(rrf, candidates)

        pool_size = min(
            len(candidates), max(self.settings.initial_recall_k, top_k * 4)
        )
        pool = sorted(candidates, key=lambda index: float(fusion[index]), reverse=True)[:pool_size]
        if self.settings.reranker_backend == "cross_encoder":
            rerank = self._cross_encoder_rerank(query, pool)
        elif self.settings.reranker_backend == "lightweight":
            rerank = self._lightweight_rerank(query, pool, fusion)
        else:
            rerank = fusion.copy()

        final = np.zeros(len(self.chunks), dtype=np.float32)
        for index in pool:
            absolute_relevance = max(
                float(raw_dense[index]),
                float(coverage[index]),
                float(exact[index]),
            )
            blended = (
                0.35 * float(fusion[index])
                + 0.35 * float(rerank[index])
                + 0.15 * float(exact[index])
                + 0.15 * absolute_relevance
            )
            final[index] = min(1.0, blended * (0.4 + 0.6 * absolute_relevance))
        ordered = sorted(pool, key=lambda index: float(final[index]), reverse=True)

        hits: list[SearchHit] = []
        for index in ordered[: min(top_k, len(ordered))]:
            hits.append(
                SearchHit(
                    chunk=self.chunks[index],
                    score=round(float(final[index]), 6),
                    bm25_score=round(float(bm25[index]), 6),
                    dense_score=round(float(dense[index]), 6),
                    exact_score=round(float(exact[index]), 6),
                    fusion_score=round(float(fusion[index]), 6),
                    rerank_score=round(float(rerank[index]), 6),
                    char_score=round(float(bm25[index]), 6),
                    word_score=round(float(bm25[index]), 6),
                    semantic_score=round(float(dense[index]), 6),
                    metadata_score=round(float(exact[index]), 6),
                )
            )
        return hits

    def describe(self) -> dict[str, Any]:
        return {
            "sparse": "BM25",
            "embedding_backend": self.settings.embedding_backend,
            "embedding_model": self.settings.embedding_model,
            "vector_backend": self.settings.vector_backend,
            "reranker_backend": self.settings.reranker_backend,
            "reranker_model": (
                self.settings.reranker_model
                if self.settings.reranker_backend == "cross_encoder"
                else self.settings.reranker_backend
            ),
            "fusion": "weighted_rrf",
            "chunk_count": len(self.chunks),
        }
