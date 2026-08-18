from types import SimpleNamespace

from app.agent import BugDiagnosisAgent
from app.planner import ToolPlanner


class FakeClient:
    enabled = True

    def __init__(self, tool_calls):
        self.tool_calls = tool_calls

    def complete(self, *args, **kwargs):
        return SimpleNamespace(tool_calls=self.tool_calls)


def _plan(client: FakeClient, message: str = "装备闪退怎么排查"):
    agent = BugDiagnosisAgent()
    planner = ToolPlanner(agent, client)
    diagnosis = agent.diagnose(message, "game", [])
    return planner.plan(
        message=message,
        domain="game",
        diagnosis=diagnosis,
        hits=[],
        session_id="planner-test",
        idempotency_key="planner-idempotency-key",
    )


def test_model_function_call_arguments_are_normalized() -> None:
    result = _plan(
        FakeClient(
            [
                {
                    "id": "model-call-1",
                    "function": {
                        "name": "search_bug_history",
                        "arguments": '{"query":"装备闪退","top_k":2,"domain":"enterprise"}',
                    },
                }
            ]
        )
    )
    assert result.mode == "llm_function_calling"
    assert result.calls[0].arguments["domain"] == "game"
    assert result.calls[0].arguments["top_k"] == 2


def test_model_cannot_create_ticket_without_explicit_user_intent() -> None:
    result = _plan(
        FakeClient(
            [
                {
                    "id": "unsafe-write",
                    "function": {"name": "create_bug_ticket", "arguments": {}},
                }
            ]
        )
    )
    assert result.mode == "deterministic_fallback"
    assert all(call.name != "create_bug_ticket" for call in result.calls)
    assert result.warning is not None


def test_malformed_model_arguments_fall_back_safely() -> None:
    result = _plan(
        FakeClient(
            [
                {
                    "id": "bad-arguments",
                    "function": {
                        "name": "search_bug_history",
                        "arguments": "[1, 2, 3]",
                    },
                }
            ]
        )
    )
    assert result.mode == "deterministic_fallback"
    assert result.warning is not None
    assert any(call.name == "search_bug_history" for call in result.calls)
