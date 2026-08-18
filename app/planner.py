from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.agent import BugDiagnosisAgent
from app.llm_client import ChatCompletionClient
from app.models import BugDiagnosis, SearchHit, ToolCall
from app.tools import build_tool_schemas, validate_tool_call


@dataclass(slots=True)
class ToolPlan:
    calls: list[ToolCall]
    mode: str
    warning: str | None = None


class ToolPlanner:
    """优先使用模型Function Calling，接口不可用时退回确定性安全规则。"""

    def __init__(self, agent: BugDiagnosisAgent, client: ChatCompletionClient) -> None:
        self.agent = agent
        self.client = client

    def _fallback(
        self,
        *,
        message: str,
        domain: str,
        hits: list[SearchHit],
        session_id: str,
        idempotency_key: str,
    ) -> ToolPlan:
        plan = self.agent.plan(
            message=message,
            domain=domain,
            hits=hits,
            session_id=session_id,
            idempotency_key=idempotency_key,
        )
        return ToolPlan(plan.tool_calls, "deterministic_fallback")

    def plan(
        self,
        *,
        message: str,
        domain: str,
        diagnosis: BugDiagnosis,
        hits: list[SearchHit],
        session_id: str,
        idempotency_key: str,
    ) -> ToolPlan:
        fallback = self._fallback(
            message=message,
            domain=domain,
            hits=hits,
            session_id=session_id,
            idempotency_key=idempotency_key,
        )
        if not self.client.enabled:
            return fallback

        system = (
            "你是研发故障处置Agent的工具规划器。只选择完成当前任务必需的工具。"
            "查询工具是只读操作；create_bug_ticket是写操作，只有用户明确要求创建或提交工单时才能选择。"
            "不得虚构工具，不得扩大用户权限。"
        )
        prompt = (
            f"工作空间：{domain}\n用户问题：{message}\n"
            f"初步诊断：{json.dumps(diagnosis.to_dict(), ensure_ascii=False)}\n"
            "请通过Function Calling选择需要的工具。"
        )
        try:
            result = self.client.complete(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                tools=build_tool_schemas(),
                tool_choice="auto",
                temperature=0.0,
                max_tokens=500,
            )
            calls: list[ToolCall] = []
            for index, raw_call in enumerate(result.tool_calls):
                function = raw_call.get("function") or {}
                name = str(function.get("name") or "")
                arguments = function.get("arguments") or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("模型返回的工具参数必须是JSON对象")
                if name == "create_bug_ticket":
                    if not any(
                        word in message.lower() for word in self.agent.ticket_keywords
                    ):
                        continue
                    arguments = {
                        "idempotency_key": idempotency_key,
                        "session_id": session_id,
                        "domain": domain,
                        "category": diagnosis.category,
                        "severity": diagnosis.severity,
                        "platform": diagnosis.platform,
                        "module": diagnosis.module,
                        "title": message[:120],
                        "description": message,
                    }
                elif name == "search_bug_history":
                    arguments.update({"domain": domain})
                    arguments.setdefault("query", message)
                    arguments.setdefault("top_k", 3)
                elif name == "get_version_changes":
                    arguments.update({"domain": domain})
                    arguments.setdefault("version", diagnosis.version or "未知")
                elif name == "generate_test_cases":
                    arguments = {
                        "category": diagnosis.category,
                        "module": diagnosis.module,
                        "platform": diagnosis.platform,
                    }
                call = ToolCall(name, arguments, str(raw_call.get("id") or f"llm-{index}"))
                calls.append(validate_tool_call(call))
            if not calls:
                fallback.warning = "模型没有返回可执行工具，已使用安全规则计划"
                return fallback
            return ToolPlan(calls, "llm_function_calling")
        except (RuntimeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            fallback.warning = f"工具规划失败，已使用安全规则：{type(exc).__name__}"
            return fallback
