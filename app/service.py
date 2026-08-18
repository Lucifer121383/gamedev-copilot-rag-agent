from __future__ import annotations

import hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from typing import Any
from uuid import uuid4

from app.agent import BugDiagnosisAgent
from app.chunker import chunk_sections
from app.config import Settings
from app.database import CopilotStore
from app.document_loader import load_workspace
from app.evidence import EvidenceGate
from app.generator import DiagnosisGenerator
from app.llm_client import ChatCompletionClient
from app.models import SearchHit, VALID_DOMAINS
from app.planner import ToolPlanner
from app.retriever import HybridRetriever
from app.tools import DevelopmentTools, build_tool_schemas, create_registry
from app.workflow import IncidentState, IncidentWorkflow


class IncidentCopilotService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.validate()
        self.settings.ensure_directories()
        self.retriever = HybridRetriever(settings.storage_dir, settings)
        self.store = CopilotStore(settings.database_url)
        self.agent = BugDiagnosisAgent()
        self.llm_client = ChatCompletionClient(settings)
        self.generator = DiagnosisGenerator(settings, self.llm_client)
        self.planner = ToolPlanner(self.agent, self.llm_client)
        self.evidence_gate = EvidenceGate(
            settings.min_retrieval_score, settings.min_evidence_sources
        )
        self.tools = DevelopmentTools(self.store, self.search)
        self.registry = create_registry(self.tools)
        self._index_lock = RLock()
        self._workflow_lock = RLock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="index-job")
        self.workflow = IncidentWorkflow(
            settings=settings,
            store=self.store,
            agent=self.agent,
            planner=self.planner,
            generator=self.generator,
            evidence_gate=self.evidence_gate,
            registry=self.registry,
            search=self.search,
        )

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.workflow.close()
        self.store.engine.dispose()

    def load_or_ingest(self) -> None:
        with self._index_lock:
            if self.retriever.load():
                return
        self.ingest()

    def ingest(self, domain: str | None = None) -> dict[str, Any]:
        if domain is not None and domain not in VALID_DOMAINS:
            raise ValueError("不支持的工作空间")
        selected = [domain] if domain else ["game", "enterprise"]
        all_files: list[str] = []
        new_chunks = []
        for selected_domain in selected:
            sections, files = load_workspace(
                self.settings.domain_dir(selected_domain),
                self.settings.upload_dir(selected_domain),
                selected_domain,
            )
            if not sections:
                raise ValueError(f"{selected_domain}工作空间没有可导入资料")
            new_chunks.extend(
                chunk_sections(
                    sections, self.settings.chunk_size, self.settings.chunk_overlap
                )
            )
            all_files.extend(files)

        with self._index_lock:
            retained = [
                chunk for chunk in self.retriever.chunks if chunk.domain not in selected
            ]
            combined = retained + new_chunks
            self.retriever.rebuild(combined)
        return {
            "message": (
                "双场景知识索引构建完成" if domain is None else f"{domain}索引构建完成"
            ),
            "domains": selected,
            "document_count": len(all_files),
            "new_chunk_count": len(new_chunks),
            "total_chunk_count": len(combined),
            "files": all_files,
            "retrieval": self.retriever.describe(),
        }

    def start_index_job(self, domain: str | None = None) -> dict[str, Any]:
        if domain is not None and domain not in VALID_DOMAINS:
            raise ValueError("不支持的工作空间")
        job_id = f"index-{uuid4().hex[:14]}"
        job = self.store.create_index_job(job_id, domain)

        def worker() -> None:
            self.store.update_index_job(
                job_id, status="running", progress=10, message="正在解析文档"
            )
            try:
                result = self.ingest(domain)
                self.store.update_index_job(
                    job_id,
                    status="completed",
                    progress=100,
                    message="索引构建完成",
                    result=result,
                )
            except Exception as exc:  # Job boundary must persist unexpected failures.
                self.store.update_index_job(
                    job_id,
                    status="failed",
                    progress=100,
                    message=f"索引构建失败：{type(exc).__name__}: {exc}",
                )

        self._executor.submit(worker)
        return job

    def search(
        self,
        query: str,
        *,
        domain: str,
        top_k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[SearchHit]:
        with self._index_lock:
            return self.retriever.search(
                query, domain=domain, top_k=top_k, filters=filters
            )

    @staticmethod
    def _source_payload(hit: SearchHit, index: int) -> dict[str, Any]:
        return IncidentWorkflow._source_payload(hit, index)

    def diagnose(
        self,
        *,
        message: str,
        domain: str,
        session_id: str | None = None,
        top_k: int = 5,
        idempotency_key: str | None = None,
        approve_write: bool = False,
    ) -> dict[str, Any]:
        if domain not in VALID_DOMAINS:
            raise ValueError("不支持的工作空间")
        message = message.strip()
        if len(message) < 2:
            raise ValueError("问题至少需要2个字符")
        request_id = f"req-{uuid4().hex[:14]}"
        session_id = session_id or f"session-{domain}-{uuid4().hex[:10]}"
        stable_key = idempotency_key or hashlib.sha256(
            f"{session_id}|{message.lower()}".encode("utf-8")
        ).hexdigest()[:32]
        self.store.ensure_session(session_id, domain)
        self.store.add_message(session_id, domain, "user", message)
        state: IncidentState = {
            "request_id": request_id,
            "session_id": session_id,
            "domain": domain,
            "message": message,
            "top_k": top_k,
            "idempotency_key": stable_key,
            "trace": [],
        }
        with self._workflow_lock:
            response = self.workflow.start(state)
            if approve_write and response["status"] == "awaiting_approval":
                response = self.workflow.resume(
                    request_id, approved=True, reason="调用方已显式确认写操作"
                )
        return response

    def approve(
        self, request_id: str, *, approved: bool, reason: str | None = None
    ) -> dict[str, Any]:
        with self._workflow_lock:
            return self.workflow.resume(request_id, approved=approved, reason=reason)

    def stats(self) -> dict[str, Any]:
        domain_stats: dict[str, Any] = {}
        for domain in sorted(VALID_DOMAINS):
            chunks = [chunk for chunk in self.retriever.chunks if chunk.domain == domain]
            sources = Counter(chunk.source for chunk in chunks)
            types = Counter(chunk.doc_type for chunk in chunks)
            domain_stats[domain] = {
                "document_count": len(sources),
                "chunk_count": len(chunks),
                "document_types": dict(types),
                "tickets": self.store.ticket_stats(domain),
            }
        return {
            "ready": bool(self.retriever.chunks),
            "service": "IncidentCopilot",
            "version": "2.0.0",
            "retrieval": self.retriever.describe(),
            "retrieval_backend": "hybrid_bm25_dense_rrf_rerank",
            "embedding_model": self.settings.embedding_model,
            "llm_enabled": self.settings.llm_enabled,
            "auth_enabled": self.settings.api_auth_enabled,
            "total_chunks": len(self.retriever.chunks),
            "domains": domain_stats,
        }

    @staticmethod
    def workspaces() -> list[dict[str, str]]:
        return [
            {
                "id": "game",
                "name": "游戏研发",
                "description": "策划文档、版本记录、崩溃日志、历史Bug与测试用例",
            },
            {
                "id": "enterprise",
                "name": "企业软件",
                "description": "产品文档、发布记录、错误日志、历史事故与测试用例",
            },
        ]

    @staticmethod
    def tool_schemas() -> list[dict[str, Any]]:
        return build_tool_schemas()


# Preserve the original import used by older scripts and tests.
GameDevCopilotService = IncidentCopilotService
