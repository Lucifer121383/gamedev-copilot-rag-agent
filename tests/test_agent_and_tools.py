import pytest

from app.agent import BugDiagnosisAgent
from app.service import GameDevCopilotService
from app.tools import build_tool_schemas


def test_game_crash_is_p1_with_entities(service: GameDevCopilotService) -> None:
    hits = service.search("Android 1.3装备闪退 E-EQP-500", domain="game", top_k=5)
    diagnosis = service.agent.diagnose(
        "Android 1.3装备闪退 E-EQP-500", "game", hits
    )
    assert diagnosis.category == "客户端崩溃"
    assert diagnosis.severity == "P1"
    assert diagnosis.platform == "Android"
    assert diagnosis.module == "装备系统"
    assert diagnosis.version == "1.3"
    assert diagnosis.error_code == "E-EQP-500"


def test_ticket_tool_only_planned_on_explicit_request(
    service: GameDevCopilotService,
) -> None:
    hits = service.search("装备闪退", domain="game", top_k=3)
    normal = service.agent.plan(
        message="装备闪退",
        domain="game",
        hits=hits,
        session_id="s-1",
        idempotency_key="key-normal",
    )
    explicit = service.agent.plan(
        message="装备闪退，请创建Bug工单",
        domain="game",
        hits=hits,
        session_id="s-1",
        idempotency_key="key-ticket",
    )
    assert "create_bug_ticket" not in [call.name for call in normal.tool_calls]
    assert "create_bug_ticket" in [call.name for call in explicit.tool_calls]


def test_tool_schemas_are_standard_and_complete() -> None:
    schemas = build_tool_schemas()
    names = {item["function"]["name"] for item in schemas}
    assert names == {
        "search_bug_history",
        "get_version_changes",
        "generate_test_cases",
        "create_bug_ticket",
    }
    assert all(item["type"] == "function" for item in schemas)
    assert all(
        item["function"]["parameters"]["additionalProperties"] is False
        for item in schemas
    )


def test_short_follow_up_uses_only_recent_user_context() -> None:
    history = [
        {"role": "user", "content": "我遇到Android装备闪退"},
        {"role": "assistant", "content": "请提供版本"},
    ]
    query = BugDiagnosisAgent.build_retrieval_query("那1.3呢", history)
    assert "Android装备闪退" in query
    assert "请提供版本" not in query


def test_same_explicit_request_creates_one_ticket(
    service: GameDevCopilotService,
) -> None:
    kwargs = {
        "message": "Android装备闪退E-EQP-500，请创建Bug工单",
        "domain": "game",
        "session_id": "idempotent-session",
    }
    first = service.diagnose(**kwargs)
    second = service.diagnose(**kwargs)
    assert first["ticket"]["ticket_no"] == second["ticket"]["ticket_no"]
    assert first["ticket"]["cached"] is False
    assert second["ticket"]["cached"] is True
    assert service.store.ticket_stats("game")["total"] == 1


def test_session_cannot_cross_workspace(service: GameDevCopilotService) -> None:
    service.diagnose(message="装备切换规则是什么", domain="game", session_id="shared")
    with pytest.raises(ValueError):
        service.diagnose(
            message="订单服务规则是什么", domain="enterprise", session_id="shared"
        )

