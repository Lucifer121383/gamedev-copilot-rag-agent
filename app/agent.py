from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.models import BugDiagnosis, SearchHit, ToolCall


_VERSION_RE = re.compile(r"(?<!\d)(\d+\.\d+(?:\.\d+)?)(?!\d)")
_ERROR_CODE_RE = re.compile(
    r"\b((?:BUG|INC|ERR|AUTH|MATCH|DB|GW|E)[-_][A-Z0-9-]{2,}|[A-Z][A-Z0-9]+-\d{3})\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class AgentPlan:
    diagnosis: BugDiagnosis
    tool_calls: list[ToolCall]
    explicit_ticket: bool


class BugDiagnosisAgent:
    """可测试的故障诊断与工具规划层，不把业务边界隐藏在大模型中。"""

    ticket_keywords = (
        "创建bug",
        "创建 bug",
        "提交bug",
        "提交 bug",
        "创建工单",
        "创建故障工单",
        "提交工单",
        "转人工",
        "帮我登记",
    )
    contextual_words = ("这个", "那个", "它", "刚才", "那", "怎么办", "怎么处理")

    @staticmethod
    def build_retrieval_query(message: str, history: list[dict[str, Any]]) -> str:
        stripped = message.strip()
        if len(stripped) >= 18 and not any(word in stripped for word in BugDiagnosisAgent.contextual_words):
            return stripped
        previous = [item["content"] for item in history if item["role"] == "user"][-2:]
        return " ".join([*previous, stripped]).strip()

    @staticmethod
    def _extract_platform(message: str) -> str:
        lowered = message.lower()
        mapping = {
            "android": "Android",
            "安卓": "Android",
            "ios": "iOS",
            "iphone": "iOS",
            "windows": "Windows",
            "linux": "Linux",
            "web": "Web",
            "网页": "Web",
        }
        return next((value for key, value in mapping.items() if key in lowered), "未知")

    @staticmethod
    def _extract_module(message: str, hits: list[SearchHit]) -> str:
        lowered = message.lower()
        mapping = {
            "装备": "装备系统",
            "背包": "背包系统",
            "匹配": "匹配系统",
            "登录": "登录系统",
            "订单": "订单服务",
            "支付": "支付服务",
            "数据库": "数据库",
            "连接池": "数据库",
            "网关": "网关",
        }
        direct = next((value for key, value in mapping.items() if key in lowered), None)
        if direct:
            return direct
        for hit in hits:
            module = hit.chunk.metadata.get("module")
            if module:
                return str(module)
        return "待确认模块"

    @staticmethod
    def _category_and_severity(message: str, domain: str) -> tuple[str, str]:
        lowered = message.lower()
        if any(word in lowered for word in ("泄露", "被盗", "黑客", "未授权")):
            return "安全事件", "P1"

        # Known database-pool failures stay service incidents even when the
        # user asks what the error code means.
        if "db-pool-503" in lowered:
            return "服务故障", "P1"

        # Separate knowledge/test questions from reports of active incidents.
        # For example, "崩溃排查步骤是什么" is a P3 question, while "刚刚崩溃"
        # remains a P1 incident.
        knowledge_patterns = (
            "代表什么",
            "表示什么",
            "标准排查步骤",
            "如何避免",
            "如何验证",
            "怎样验证",
            "为什么要",
            "能不能",
            "是否可以",
            "哪条测试用例",
            "预期结果",
            "手续费是多少",
            "公司年假",
        )
        history_lookup = "查询" in lowered and any(
            word in lowered for word in ("根因", "修复方案", "处理方案")
        )
        if history_lookup or any(pattern in lowered for pattern in knowledge_patterns):
            return "研发知识咨询", "P3"

        if any(word in lowered for word in ("崩溃", "闪退", "crash", "nullreference")):
            return "客户端崩溃" if domain == "game" else "服务故障", "P1"
        if any(word in lowered for word in ("慢sql", "慢查询", "性能", "延迟", "掉帧", "慢")):
            return "性能问题", "P2"
        if any(word in lowered for word in ("500", "不可用", "连接池", "宕机", "服务中断")):
            return "服务故障", "P1"
        if any(word in lowered for word in ("超时", "timeout", "卡死", "无法登录")):
            return "功能故障", "P2"
        if any(word in lowered for word in ("回调丢失", "失败", "无响应", "无法", "异常")):
            return "功能故障", "P2"
        if any(phrase in lowered for phrase in ("怎么修复", "如何修复")):
            return "一般缺陷", "P2"
        if any(
            word in lowered
            for word in ("怎么", "为什么", "哪些", "什么内容", "说明", "规则", "设计", "发布风险")
        ):
            return "研发知识咨询", "P3"
        return "一般缺陷", "P2"

    @staticmethod
    def _collect_field(hits: list[SearchHit], field: str, limit: int = 4) -> list[str]:
        values: list[str] = []
        marker = f"{field}:"
        for hit in hits:
            for line in hit.chunk.text.splitlines():
                if line.lower().startswith(marker.lower()):
                    raw = line.split(":", 1)[1].strip()
                    for value in re.split(r"[；;]", raw):
                        cleaned = value.strip(" []'\"")
                        if cleaned and cleaned not in values:
                            values.append(cleaned)
                            if len(values) >= limit:
                                return values
        return values

    def diagnose(
        self, message: str, domain: str, hits: list[SearchHit]
    ) -> BugDiagnosis:
        category, severity = self._category_and_severity(message, domain)
        platform = self._extract_platform(message)
        module = self._extract_module(message, hits)
        version_match = _VERSION_RE.search(message)
        error_match = _ERROR_CODE_RE.search(message)

        causes = self._collect_field(hits, "root_cause", 3)
        steps = self._collect_field(hits, "reproduction_steps", 5)
        tests = self._collect_field(hits, "regression_tests", 5)
        if not causes:
            causes = [
                "当前证据不足以确认唯一根因，需要结合错误日志与版本改动继续排查"
            ]
        if not steps:
            steps = [
                "确认版本、平台、设备或服务环境",
                "记录完整错误码、日志时间和操作路径",
                "在隔离测试环境按原路径复现",
            ]
        if not tests:
            tests = [
                f"{module}核心流程回归",
                "异常网络或依赖超时测试",
                "重复操作与边界条件测试",
            ]

        top_score = hits[0].score if hits else 0.0
        confidence = min(0.96, round(0.42 + top_score * 1.5, 2)) if hits else 0.25
        return BugDiagnosis(
            category=category,
            severity=severity,
            platform=platform,
            module=module,
            version=version_match.group(1) if version_match else None,
            error_code=error_match.group(1).upper() if error_match else None,
            possible_causes=causes,
            reproduction_steps=steps,
            regression_tests=tests,
            confidence=confidence,
            needs_human_review=severity == "P1" or confidence < 0.65,
        )

    def plan(
        self,
        *,
        message: str,
        domain: str,
        hits: list[SearchHit],
        session_id: str,
        idempotency_key: str,
    ) -> AgentPlan:
        diagnosis = self.diagnose(message, domain, hits)
        explicit_ticket = any(word in message.lower() for word in self.ticket_keywords)
        calls = [
            ToolCall(
                "search_bug_history",
                {
                    "query": message,
                    "domain": domain,
                    "top_k": 3,
                },
                "tool-history",
            )
        ]
        if diagnosis.version:
            calls.append(
                ToolCall(
                    "get_version_changes",
                    {"version": diagnosis.version, "domain": domain},
                    "tool-version",
                )
            )
        calls.append(
            ToolCall(
                "generate_test_cases",
                {
                    "category": diagnosis.category,
                    "module": diagnosis.module,
                    "platform": diagnosis.platform,
                },
                "tool-tests",
            )
        )
        if explicit_ticket:
            calls.append(
                ToolCall(
                    "create_bug_ticket",
                    {
                        "idempotency_key": idempotency_key,
                        "session_id": session_id,
                        "domain": domain,
                        "category": diagnosis.category,
                        "severity": diagnosis.severity,
                        "platform": diagnosis.platform,
                        "module": diagnosis.module,
                        "title": message[:120],
                        "description": message,
                    },
                    "tool-ticket",
                )
            )
        return AgentPlan(diagnosis, calls, explicit_ticket)
