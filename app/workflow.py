from __future__ import annotations

import sqlite3
from time import perf_counter
from typing import Any, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.agent import BugDiagnosisAgent
from app.config import Settings
from app.database import CopilotStore
from app.evidence import EvidenceGate
from app.generator import DiagnosisGenerator
from app.models import BugDiagnosis, EvidenceAssessment, SearchHit, ToolCall
from app.planner import ToolPlanner
from app.tools import ToolRegistry


class IncidentState(TypedDict, total=False):
    request_id: str
    session_id: str
    domain: str
    message: str
    top_k: int
    idempotency_key: str
    retrieval_query: str
    hits: list[dict[str, Any]]
    evidence: dict[str, Any]
    diagnosis: dict[str, Any]
    read_tool_calls: list[dict[str, Any]]
    write_tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    ticket: dict[str, Any] | None
    planner_mode: str
    planner_warning: str | None
    approval_status: str
    approval_reason: str | None
    answer: str
    generation_mode: str
    generation_warning: str | None
    citation_valid: bool
    prompt_tokens: int
    completion_tokens: int
    retries: int
    retrieval_ms: float
    generation_ms: float
    trace: list[dict[str, Any]]
    response: dict[str, Any]


class IncidentWorkflow:
    """可恢复、带人工审批的LangGraph研发故障处置工作流。"""

    WRITE_TOOLS = {"create_bug_ticket"}

    def __init__(
        self,
        *,
        settings: Settings,
        store: CopilotStore,
        agent: BugDiagnosisAgent,
        planner: ToolPlanner,
        generator: DiagnosisGenerator,
        evidence_gate: EvidenceGate,
        registry: ToolRegistry,
        search: Any,
    ) -> None:
        self.settings = settings
        self.store = store
        self.agent = agent
        self.planner = planner
        self.generator = generator
        self.evidence_gate = evidence_gate
        self.registry = registry
        self.search = search
        self._checkpoint_connection = sqlite3.connect(
            settings.checkpoint_path, check_same_thread=False
        )
        self.checkpointer = SqliteSaver(self._checkpoint_connection)
        self.graph = self._build_graph()

    def close(self) -> None:
        self._checkpoint_connection.close()

    def _build_graph(self):
        builder = StateGraph(IncidentState)
        builder.add_node("intake", self._intake)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("evidence_gate", self._assess_evidence)
        builder.add_node("diagnose", self._diagnose)
        builder.add_node("plan_tools", self._plan_tools)
        builder.add_node("execute_read_tools", self._execute_read_tools)
        builder.add_node("approval", self._approval)
        builder.add_node("execute_write_tools", self._execute_write_tools)
        builder.add_node("compose", self._compose)
        builder.add_node("persist", self._persist)

        builder.add_edge(START, "intake")
        builder.add_edge("intake", "retrieve")
        builder.add_edge("retrieve", "evidence_gate")
        builder.add_edge("evidence_gate", "diagnose")
        builder.add_edge("diagnose", "plan_tools")
        builder.add_edge("plan_tools", "execute_read_tools")
        builder.add_conditional_edges(
            "execute_read_tools",
            self._route_approval,
            {"approval": "approval", "compose": "compose"},
        )
        builder.add_conditional_edges(
            "approval",
            self._route_after_approval,
            {"execute": "execute_write_tools", "compose": "compose"},
        )
        builder.add_edge("execute_write_tools", "compose")
        builder.add_edge("compose", "persist")
        builder.add_edge("persist", END)
        return builder.compile(checkpointer=self.checkpointer, name="incident-copilot")

    def _step(
        self, state: IncidentState, node: str, detail: str, status: str = "completed"
    ) -> list[dict[str, Any]]:
        trace = list(state.get("trace", []))
        if len(trace) >= self.settings.max_agent_steps:
            raise RuntimeError("Agent超过最大执行步数，已停止并转人工")
        trace.append(
            {
                "step": len(trace) + 1,
                "node": node,
                "status": status,
                "detail": detail,
            }
        )
        return trace

    @staticmethod
    def _hits(state: IncidentState) -> list[SearchHit]:
        return [SearchHit.from_dict(item) for item in state.get("hits", [])]

    @staticmethod
    def _calls(state: IncidentState, key: str) -> list[ToolCall]:
        return [ToolCall.from_dict(item) for item in state.get(key, [])]

    def _intake(self, state: IncidentState) -> dict[str, Any]:
        history = self.store.get_messages(
            state["session_id"], limit=self.settings.max_history_messages
        )
        # The current message is persisted before graph execution so paused
        # runs remain auditable. Exclude that item from contextual rewriting.
        previous_history = history
        if (
            history
            and history[-1].get("role") == "user"
            and history[-1].get("content", "").strip() == state["message"].strip()
        ):
            previous_history = history[:-1]
        retrieval_query = self.agent.build_retrieval_query(
            state["message"], previous_history
        )
        return {
            "retrieval_query": retrieval_query,
            "tool_results": [],
            "ticket": None,
            "approval_status": "not_required",
            "trace": self._step(
                state,
                "intake",
                f"工作空间={state['domain']}，会话={state['session_id']}，已读取{len(previous_history)}条历史消息",
            ),
        }

    def _retrieve(self, state: IncidentState) -> dict[str, Any]:
        started = perf_counter()
        hits = self.search(
            state["retrieval_query"],
            domain=state["domain"],
            top_k=state["top_k"],
        )
        retrieval_ms = (perf_counter() - started) * 1000
        return {
            "hits": [hit.to_dict() for hit in hits],
            "retrieval_ms": round(retrieval_ms, 2),
            "trace": self._step(
                state,
                "hybrid_retrieve",
                f"BM25、向量和精确匹配融合后返回{len(hits)}条候选，耗时{retrieval_ms:.1f}ms",
            ),
        }

    def _assess_evidence(self, state: IncidentState) -> dict[str, Any]:
        assessment = self.evidence_gate.assess(
            state["retrieval_query"], self._hits(state)
        )
        return {
            "evidence": assessment.to_dict(),
            "trace": self._step(
                state,
                "evidence_gate",
                f"{'证据充分' if assessment.sufficient else '证据不足'}：{assessment.reason}",
                "completed" if assessment.sufficient else "needs_review",
            ),
        }

    def _diagnose(self, state: IncidentState) -> dict[str, Any]:
        hits = self._hits(state)
        diagnosis = self.agent.diagnose(state["message"], state["domain"], hits)
        evidence = EvidenceAssessment(**state["evidence"])
        if not evidence.sufficient:
            diagnosis.confidence = min(diagnosis.confidence, evidence.confidence)
            diagnosis.needs_human_review = True
        return {
            "diagnosis": diagnosis.to_dict(),
            "trace": self._step(
                state,
                "diagnose",
                f"{diagnosis.category}/{diagnosis.severity}/置信度{diagnosis.confidence:.0%}",
            ),
        }

    def _plan_tools(self, state: IncidentState) -> dict[str, Any]:
        diagnosis = BugDiagnosis.from_dict(state["diagnosis"])
        plan = self.planner.plan(
            message=state["message"],
            domain=state["domain"],
            diagnosis=diagnosis,
            hits=self._hits(state),
            session_id=state["session_id"],
            idempotency_key=state["idempotency_key"],
        )
        evidence = EvidenceAssessment(**state["evidence"])
        read_calls = [call for call in plan.calls if call.name not in self.WRITE_TOOLS]
        write_calls = [call for call in plan.calls if call.name in self.WRITE_TOOLS]
        if not evidence.sufficient:
            write_calls = []
        detail = (
            f"规划器={plan.mode}，只读工具{len(read_calls)}个，写工具{len(write_calls)}个"
        )
        if plan.warning:
            detail += f"，{plan.warning}"
        return {
            "read_tool_calls": [call.to_dict() for call in read_calls],
            "write_tool_calls": [call.to_dict() for call in write_calls],
            "planner_mode": plan.mode,
            "planner_warning": plan.warning,
            "trace": self._step(state, "plan_tools", detail),
        }

    def _execute_read_tools(self, state: IncidentState) -> dict[str, Any]:
        tool_results = list(state.get("tool_results", []))
        diagnosis = BugDiagnosis.from_dict(state["diagnosis"])
        trace = list(state.get("trace", []))
        working_state = dict(state)
        for call in self._calls(state, "read_tool_calls"):
            result = self.registry.execute(call)
            tool_results.append(result)
            working_state["trace"] = trace
            trace = self._step(
                working_state,
                "execute_read_tool",
                f"{call.name}: {'成功' if result['ok'] else result.get('error')}",
                "completed" if result["ok"] else "failed",
            )
            if call.name == "generate_test_cases" and result["ok"]:
                generated = result["data"].get("cases", [])
                diagnosis.regression_tests = list(
                    dict.fromkeys([*diagnosis.regression_tests, *generated])
                )[:7]
        return {
            "tool_results": tool_results,
            "diagnosis": diagnosis.to_dict(),
            "trace": trace,
        }

    @staticmethod
    def _route_approval(state: IncidentState) -> str:
        return "approval" if state.get("write_tool_calls") else "compose"

    def _approval(self, state: IncidentState) -> dict[str, Any]:
        write_calls = state.get("write_tool_calls", [])
        decision = interrupt(
            {
                "request_id": state["request_id"],
                "type": "write_tool_approval",
                "message": "Agent计划执行有副作用的写操作，请确认是否允许。",
                "tools": write_calls,
            }
        )
        approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
        reason = decision.get("reason") if isinstance(decision, dict) else None
        return {
            "approval_status": "approved" if approved else "rejected",
            "approval_reason": reason,
            "trace": self._step(
                state,
                "human_approval",
                "用户批准写操作" if approved else f"用户拒绝写操作{f'：{reason}' if reason else ''}",
                "completed" if approved else "rejected",
            ),
        }

    @staticmethod
    def _route_after_approval(state: IncidentState) -> str:
        return "execute" if state.get("approval_status") == "approved" else "compose"

    def _execute_write_tools(self, state: IncidentState) -> dict[str, Any]:
        tool_results = list(state.get("tool_results", []))
        trace = list(state.get("trace", []))
        ticket: dict[str, Any] | None = None
        working_state = dict(state)
        for call in self._calls(state, "write_tool_calls"):
            result = self.registry.execute(call)
            tool_results.append(result)
            working_state["trace"] = trace
            trace = self._step(
                working_state,
                "execute_write_tool",
                f"{call.name}: {'成功' if result['ok'] else result.get('error')}",
                "completed" if result["ok"] else "failed",
            )
            if call.name == "create_bug_ticket" and result["ok"]:
                ticket = result["data"]
        return {"tool_results": tool_results, "ticket": ticket, "trace": trace}

    def _compose(self, state: IncidentState) -> dict[str, Any]:
        started = perf_counter()
        evidence = EvidenceAssessment(**state["evidence"])
        generated = self.generator.generate(
            state["message"],
            BugDiagnosis.from_dict(state["diagnosis"]),
            self._hits(state),
            list(state.get("tool_results", [])),
            state.get("ticket"),
            evidence,
        )
        generation_ms = (perf_counter() - started) * 1000
        warning_parts = [
            value
            for value in (state.get("planner_warning"), generated.warning)
            if value
        ]
        return {
            "answer": generated.answer,
            "generation_mode": generated.mode,
            "generation_warning": "；".join(warning_parts) or None,
            "citation_valid": generated.citation_valid,
            "prompt_tokens": generated.prompt_tokens,
            "completion_tokens": generated.completion_tokens,
            "retries": generated.retries,
            "generation_ms": round(generation_ms, 2),
            "trace": self._step(
                state,
                "compose",
                f"回答模式={generated.mode}，引用校验={'通过' if generated.citation_valid else '降级'}",
            ),
        }

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
                "bm25": hit.bm25_score,
                "dense": hit.dense_score,
                "exact": hit.exact_score,
                "fusion": hit.fusion_score,
                "rerank": hit.rerank_score,
                "char": hit.char_score,
                "word": hit.word_score,
                "semantic": hit.semantic_score,
                "metadata": hit.metadata_score,
            },
            "metadata": hit.chunk.metadata,
            "text": hit.chunk.text,
        }

    def _persist(self, state: IncidentState) -> dict[str, Any]:
        diagnosis = BugDiagnosis.from_dict(state["diagnosis"])
        evidence = EvidenceAssessment(**state["evidence"])
        ticket = state.get("ticket")
        action = (
            "create_ticket"
            if ticket
            else "write_rejected"
            if state.get("approval_status") == "rejected"
            else "human_review"
            if diagnosis.needs_human_review or not evidence.sufficient
            else "answer"
        )
        trace = self._step(state, "persist", "保存回答、执行轨迹和评测字段")
        active_ms = float(state.get("retrieval_ms", 0.0)) + float(
            state.get("generation_ms", 0.0)
        )
        response = {
            "request_id": state["request_id"],
            "session_id": state["session_id"],
            "domain": state["domain"],
            "message": state["message"],
            "status": "completed",
            "answer": state["answer"],
            "mode": state["generation_mode"],
            "warning": state.get("generation_warning"),
            "action": action,
            "diagnosis": diagnosis.to_dict(),
            "evidence": evidence.to_dict(),
            "sources": [
                self._source_payload(hit, index)
                for index, hit in enumerate(self._hits(state), start=1)
            ],
            "tool_results": list(state.get("tool_results", [])),
            "ticket": ticket,
            "approval": None,
            "planner_mode": state.get("planner_mode", "deterministic_fallback"),
            "trace": trace,
            "metrics": {
                "retrieval_ms": float(state.get("retrieval_ms", 0.0)),
                "generation_ms": float(state.get("generation_ms", 0.0)),
                "active_ms": round(active_ms, 2),
                "prompt_tokens": int(state.get("prompt_tokens", 0)),
                "completion_tokens": int(state.get("completion_tokens", 0)),
                "total_tokens": int(state.get("prompt_tokens", 0))
                + int(state.get("completion_tokens", 0)),
                "llm_retries": int(state.get("retries", 0)),
                "citation_valid": bool(state.get("citation_valid", True)),
            },
        }
        self.store.add_message(
            state["session_id"], state["domain"], "assistant", state["answer"]
        )
        self.store.record_agent_run(
            state["request_id"],
            state["session_id"],
            state["domain"],
            trace,
            response,
            status="completed",
        )
        return {"trace": trace, "response": response}

    def start(self, state: IncidentState) -> dict[str, Any]:
        config = {"configurable": {"thread_id": state["request_id"]}}
        output = self.graph.invoke(state, config=config)
        if "__interrupt__" not in output:
            return output["response"]

        interrupt_item = output["__interrupt__"][0]
        hits = [SearchHit.from_dict(item) for item in output.get("hits", [])]
        diagnosis = BugDiagnosis.from_dict(output["diagnosis"])
        evidence = EvidenceAssessment(**output["evidence"])
        trace = self._step(
            output,
            "await_approval",
            "工作流已持久化Checkpoint，等待用户批准写操作",
            "paused",
        )
        response = {
            "request_id": output["request_id"],
            "session_id": output["session_id"],
            "domain": output["domain"],
            "message": output["message"],
            "status": "awaiting_approval",
            "answer": "诊断和只读查询已经完成。Agent计划创建工单，请确认是否允许执行写操作。",
            "mode": "approval_required",
            "warning": output.get("planner_warning"),
            "action": "awaiting_approval",
            "diagnosis": diagnosis.to_dict(),
            "evidence": evidence.to_dict(),
            "sources": [
                self._source_payload(hit, index) for index, hit in enumerate(hits, start=1)
            ],
            "tool_results": list(output.get("tool_results", [])),
            "ticket": None,
            "approval": interrupt_item.value,
            "planner_mode": output.get("planner_mode", "deterministic_fallback"),
            "trace": trace,
            "metrics": {
                "retrieval_ms": float(output.get("retrieval_ms", 0.0)),
                "generation_ms": 0.0,
                "active_ms": float(output.get("retrieval_ms", 0.0)),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "llm_retries": 0,
                "citation_valid": True,
            },
        }
        self.store.record_agent_run(
            output["request_id"],
            output["session_id"],
            output["domain"],
            trace,
            response,
            status="awaiting_approval",
        )
        return response

    def resume(self, request_id: str, *, approved: bool, reason: str | None = None) -> dict[str, Any]:
        config = {"configurable": {"thread_id": request_id}}
        snapshot = self.graph.get_state(config)
        if not snapshot.values:
            raise ValueError("找不到待审批的Agent请求")
        if not snapshot.next:
            existing = self.store.get_agent_run(request_id)
            if existing and existing["status"] == "completed":
                return existing["result"]
            raise ValueError("该请求已经结束，不能重复审批")
        output = self.graph.invoke(
            Command(resume={"approved": approved, "reason": reason}), config=config
        )
        if "__interrupt__" in output:
            raise RuntimeError("工作流仍处于等待状态")
        return output["response"]
