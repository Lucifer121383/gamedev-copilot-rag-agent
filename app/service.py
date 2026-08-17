from __future__ import annotations

import hashlib
from collections import Counter
from threading import RLock
from time import perf_counter
from typing import Any
from uuid import uuid4

from app.agent import BugDiagnosisAgent
from app.chunker import chunk_sections
from app.config import Settings
from app.database import CopilotStore
from app.document_loader import load_workspace
from app.generator import DiagnosisGenerator
from app.models import SearchHit, VALID_DOMAINS
from app.retriever import HybridRetriever
from app.tools import DevelopmentTools, build_tool_schemas, create_registry


class GameDevCopilotService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.validate()
        self.settings.ensure_directories()
        self.retriever = HybridRetriever(
            settings.storage_dir, settings.embedding_backend, settings.embedding_model
        )
        self.store = CopilotStore(settings.database_path)
        self.agent = BugDiagnosisAgent()
        self.generator = DiagnosisGenerator(settings)
        self.tools = DevelopmentTools(self.store, self.search)
        self.registry = create_registry(self.tools)
        self._lock = RLock()

    def load_or_ingest(self) -> None:
        with self._lock:
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

        with self._lock:
            retained = [
                chunk for chunk in self.retriever.chunks if chunk.domain not in selected
            ]
            combined = retained + new_chunks
            self.retriever.rebuild(combined)
        return {
            "message": "双场景知识索引构建完成" if domain is None else f"{domain}索引构建完成",
            "domains": selected,
            "document_count": len(all_files),
            "new_chunk_count": len(new_chunks),
            "total_chunk_count": len(combined),
            "files": all_files,
            "retrieval_backend": self.settings.embedding_backend,
        }

    def search(
        self,
        query: str,
        *,
        domain: str,
        top_k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[SearchHit]:
        with self._lock:
            return self.retriever.search(
                query, domain=domain, top_k=top_k, filters=filters
            )

    @staticmethod
    def _source_payload(hit: SearchHit, index: int) -> dict[str, Any]:
        return {
            "citation": f"来源{index}",
            "source": hit.chunk.source,
            "domain": hit.chunk.domain,
            "doc_type": hit.chunk.doc_type,
            "page": hit.chunk.page,
            "section": hit.chunk.section,
            "score": round(hit.score, 4),
            "score_breakdown": {
                "char": hit.char_score,
                "word": hit.word_score,
                "semantic": hit.semantic_score,
                "metadata": hit.metadata_score,
            },
            "metadata": hit.chunk.metadata,
            "text": hit.chunk.text,
        }

    def diagnose(
        self,
        *,
        message: str,
        domain: str,
        session_id: str | None = None,
        top_k: int = 5,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if domain not in VALID_DOMAINS:
            raise ValueError("不支持的工作空间")
        message = message.strip()
        if len(message) < 2:
            raise ValueError("问题至少需要2个字符")
        request_started = perf_counter()
        request_id = f"req-{uuid4().hex[:14]}"
        session_id = session_id or f"session-{domain}-{uuid4().hex[:10]}"
        stable_key = idempotency_key or hashlib.sha256(
            f"{session_id}|{message.lower()}".encode("utf-8")
        ).hexdigest()[:32]
        trace: list[dict[str, Any]] = []

        def step(name: str, status: str = "completed", detail: str = "") -> None:
            if len(trace) >= self.settings.max_agent_steps:
                raise RuntimeError("Agent超过最大执行步数，已停止并转人工")
            trace.append(
                {
                    "step": len(trace) + 1,
                    "node": name,
                    "status": status,
                    "detail": detail,
                }
            )

        step("intake", detail=f"工作空间={domain}，会话={session_id}")
        history = self.store.get_messages(session_id, limit=8)
        retrieval_query = self.agent.build_retrieval_query(message, history)

        retrieval_started = perf_counter()
        raw_hits = self.search(retrieval_query, domain=domain, top_k=top_k)
        hits = [
            hit for hit in raw_hits if hit.score >= self.settings.min_retrieval_score
        ]
        retrieval_ms = (perf_counter() - retrieval_started) * 1000
        step("retrieve", detail=f"召回{len(raw_hits)}条，可靠资料{len(hits)}条")

        plan = self.agent.plan(
            message=message,
            domain=domain,
            hits=hits,
            session_id=session_id,
            idempotency_key=stable_key,
        )
        step(
            "diagnose",
            detail=(
                f"{plan.diagnosis.category}/{plan.diagnosis.severity}/"
                f"置信度{plan.diagnosis.confidence:.0%}"
            ),
        )
        step("tool_plan", detail=f"计划调用{len(plan.tool_calls)}个受控工具")

        tool_results: list[dict[str, Any]] = []
        ticket: dict[str, Any] | None = None
        for call in plan.tool_calls:
            result = self.registry.execute(call)
            tool_results.append(result)
            step(
                "tool_execute",
                status="completed" if result["ok"] else "failed",
                detail=f"{call.name}: {'成功' if result['ok'] else result.get('error')}",
            )
            if call.name == "generate_test_cases" and result["ok"]:
                generated = result["data"].get("cases", [])
                merged = list(dict.fromkeys([*plan.diagnosis.regression_tests, *generated]))
                plan.diagnosis.regression_tests = merged[:7]
            if call.name == "create_bug_ticket" and result["ok"]:
                ticket = result["data"]

        generation_started = perf_counter()
        generated = self.generator.generate(
            message, plan.diagnosis, hits, tool_results, ticket
        )
        generation_ms = (perf_counter() - generation_started) * 1000
        step("compose", detail=f"回答模式={generated.mode}")
        step("finish", detail="保存消息、轨迹和诊断结果")

        self.store.add_message(session_id, domain, "user", message)
        self.store.add_message(session_id, domain, "assistant", generated.answer)
        sources = [
            self._source_payload(hit, index)
            for index, hit in enumerate(hits, start=1)
        ]
        action = (
            "create_ticket"
            if ticket
            else "human_review"
            if plan.diagnosis.needs_human_review
            else "answer"
        )
        result = {
            "request_id": request_id,
            "session_id": session_id,
            "domain": domain,
            "message": message,
            "answer": generated.answer,
            "mode": generated.mode,
            "warning": generated.warning,
            "action": action,
            "diagnosis": plan.diagnosis.to_dict(),
            "sources": sources,
            "tool_results": tool_results,
            "ticket": ticket,
            "trace": trace,
            "metrics": {
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": round(generation_ms, 2),
                "total_ms": round((perf_counter() - request_started) * 1000, 2),
            },
        }
        self.store.record_agent_run(
            request_id,
            session_id,
            domain,
            trace,
            {
                "action": action,
                "diagnosis": plan.diagnosis.to_dict(),
                "ticket_no": ticket.get("ticket_no") if ticket else None,
            },
        )
        return result

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
            "retrieval_backend": self.settings.embedding_backend,
            "embedding_model": (
                self.settings.embedding_model
                if self.settings.embedding_backend == "sentence_transformers"
                else "字符TF-IDF + 词项TF-IDF + 元数据精确匹配"
            ),
            "llm_enabled": self.settings.llm_enabled,
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

