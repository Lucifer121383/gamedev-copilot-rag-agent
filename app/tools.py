from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.database import CopilotStore
from app.models import SearchHit, ToolCall, VALID_DOMAINS


def build_tool_schemas() -> list[dict[str, Any]]:
    """标准Function Calling工具定义，可传给兼容接口的大模型。"""

    def schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    domain_property = {"type": "string", "enum": ["game", "enterprise"]}
    return [
        schema(
            "search_bug_history",
            "在当前工作空间检索相似历史Bug或故障工单，只读操作",
            {
                "query": {"type": "string"},
                "domain": domain_property,
                "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            ["query", "domain", "top_k"],
        ),
        schema(
            "get_version_changes",
            "查询指定版本的发布记录，只读操作",
            {"version": {"type": "string"}, "domain": domain_property},
            ["version", "domain"],
        ),
        schema(
            "generate_test_cases",
            "根据问题类别、模块和平台生成回归测试建议，只读操作",
            {
                "category": {"type": "string"},
                "module": {"type": "string"},
                "platform": {"type": "string"},
            },
            ["category", "module", "platform"],
        ),
        schema(
            "create_bug_ticket",
            "在用户明确要求后创建Bug工单，属于有副作用操作",
            {
                "idempotency_key": {"type": "string"},
                "session_id": {"type": "string"},
                "domain": domain_property,
                "category": {"type": "string"},
                "severity": {"type": "string", "enum": ["P1", "P2", "P3"]},
                "platform": {"type": "string"},
                "module": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
            },
            [
                "idempotency_key",
                "session_id",
                "domain",
                "category",
                "severity",
                "platform",
                "module",
                "title",
                "description",
            ],
        ),
    ]


_REQUIRED_FIELDS = {
    "search_bug_history": {"query", "domain", "top_k"},
    "get_version_changes": {"version", "domain"},
    "generate_test_cases": {"category", "module", "platform"},
    "create_bug_ticket": {
        "idempotency_key",
        "session_id",
        "domain",
        "category",
        "severity",
        "platform",
        "module",
        "title",
        "description",
    },
}


def validate_tool_call(call: ToolCall) -> ToolCall:
    if call.name not in _REQUIRED_FIELDS:
        raise ValueError(f"未授权工具: {call.name}")
    required = _REQUIRED_FIELDS[call.name]
    if set(call.arguments) != required:
        missing = required - set(call.arguments)
        extra = set(call.arguments) - required
        raise ValueError(f"工具参数不完整，缺少{sorted(missing)}，多余{sorted(extra)}")
    normalized: dict[str, Any] = {}
    for key, value in call.arguments.items():
        if key == "top_k":
            if not isinstance(value, int) or not 1 <= value <= 5:
                raise ValueError("top_k必须是1到5的整数")
            normalized[key] = value
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key}必须是非空字符串")
        normalized[key] = value.strip()
    if "domain" in normalized and normalized["domain"] not in VALID_DOMAINS:
        raise ValueError("domain不受支持")
    if "severity" in normalized and normalized["severity"] not in {"P1", "P2", "P3"}:
        raise ValueError("severity不受支持")
    return ToolCall(call.name, normalized, call.call_id)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., dict[str, Any]]] = {}

    def register(self, name: str, function: Callable[..., dict[str, Any]]) -> None:
        if name in self._tools:
            raise ValueError(f"工具已注册: {name}")
        self._tools[name] = function

    def execute(self, call: ToolCall) -> dict[str, Any]:
        validated = validate_tool_call(call)
        function = self._tools.get(validated.name)
        if function is None:
            raise ValueError(f"工具未安装: {validated.name}")
        try:
            data = function(**validated.arguments)
            return {
                "call_id": validated.call_id,
                "tool": validated.name,
                "ok": True,
                "data": data,
            }
        except (ValueError, RuntimeError) as exc:
            return {
                "call_id": validated.call_id,
                "tool": validated.name,
                "ok": False,
                "error": str(exc),
            }


class DevelopmentTools:
    def __init__(
        self,
        store: CopilotStore,
        search: Callable[..., list[SearchHit]],
    ) -> None:
        self.store = store
        self.search = search

    @staticmethod
    def _hit_payload(hit: SearchHit) -> dict[str, Any]:
        return {
            "source": hit.chunk.source,
            "section": hit.chunk.section,
            "score": hit.score,
            "bug_id": hit.chunk.metadata.get("bug_id"),
            "title": hit.chunk.metadata.get("title"),
            "severity": hit.chunk.metadata.get("severity"),
            "text": hit.chunk.text[:500],
        }

    def search_bug_history(self, query: str, domain: str, top_k: int) -> dict[str, Any]:
        doc_type = "bug_tickets" if domain == "game" else "incident_tickets"
        hits = self.search(
            query, domain=domain, top_k=top_k, filters={"doc_type": doc_type}
        )
        return {"count": len(hits), "items": [self._hit_payload(hit) for hit in hits]}

    def get_version_changes(self, version: str, domain: str) -> dict[str, Any]:
        hits = self.search(
            f"版本 {version} 更新 修改 风险",
            domain=domain,
            top_k=3,
            filters={"doc_type": "release_notes"},
        )
        return {"version": version, "items": [self._hit_payload(hit) for hit in hits]}

    def generate_test_cases(
        self, category: str, module: str, platform: str
    ) -> dict[str, Any]:
        cases = [
            f"验证{platform}平台{module}核心流程能够正常完成",
            f"模拟异常依赖或弱网，验证{category}能够被安全处理",
            "重复执行相同操作，验证节流和幂等保护",
            "验证失败后状态、资源和连接能够正确释放",
        ]
        if category in {"客户端崩溃", "服务故障", "安全事件"}:
            cases.append("执行故障恢复和人工升级流程回归")
        return {"count": len(cases), "cases": cases}

    def create_bug_ticket(self, **arguments: str) -> dict[str, Any]:
        return self.store.create_ticket(**arguments)


def create_registry(tools: DevelopmentTools) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("search_bug_history", tools.search_bug_history)
    registry.register("get_version_changes", tools.get_version_changes)
    registry.register("generate_test_cases", tools.generate_test_cases)
    registry.register("create_bug_ticket", tools.create_bug_ticket)
    return registry

