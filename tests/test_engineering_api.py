import time
import asyncio

from fastapi.testclient import TestClient

from app import main as main_module
from app.main import FixedWindowLimiter
from app.service import IncidentCopilotService


def test_api_approval_and_request_id(
    service: IncidentCopilotService, monkeypatch
) -> None:
    monkeypatch.setattr(main_module, "service", service)
    with TestClient(main_module.app) as client:
        pending_response = client.post(
            "/api/diagnose",
            headers={"X-Request-ID": "http-test-approval"},
            json={
                "message": "Android装备闪退E-EQP-500，请创建Bug工单",
                "domain": "game",
                "session_id": "api-approval",
            },
        )
        assert pending_response.status_code == 200
        assert pending_response.headers["X-Request-ID"] == "http-test-approval"
        pending = pending_response.json()
        assert pending["status"] == "awaiting_approval"

        approved = client.post(
            f"/api/approvals/{pending['request_id']}",
            json={"approved": True, "reason": "API测试批准"},
        )
        assert approved.status_code == 200
        assert approved.json()["ticket"] is not None


def test_uniform_validation_error_contains_request_id(
    service: IncidentCopilotService, monkeypatch
) -> None:
    monkeypatch.setattr(main_module, "service", service)
    with TestClient(main_module.app) as client:
        response = client.post(
            "/api/search",
            headers={"X-Request-ID": "validation-123"},
            json={"query": "x", "domain": "game"},
        )
        body = response.json()
        assert response.status_code == 422
        assert body["error"]["code"] == "validation_error"
        assert body["request_id"] == "validation-123"


def test_background_index_job_reaches_terminal_state(
    service: IncidentCopilotService,
) -> None:
    job = service.start_index_job("game")
    current = job
    for _ in range(100):
        current = service.store.get_index_job(job["job_id"])
        if current and current["status"] in {"completed", "failed"}:
            break
        time.sleep(0.02)
    assert current is not None
    assert current["status"] == "completed"
    assert current["progress"] == 100


def test_api_key_can_be_required(service: IncidentCopilotService, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "service", service)
    monkeypatch.setattr(main_module.settings, "api_auth_enabled", True)
    monkeypatch.setattr(main_module.settings, "api_key", "a-secure-test-key")
    with TestClient(main_module.app) as client:
        unauthorized = client.get("/api/health")
        authorized = client.get(
            "/api/health", headers={"X-API-Key": "a-secure-test-key"}
        )
        public = client.get("/healthz")
    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert public.status_code == 200


def test_fixed_window_limiter_rejects_excess_request() -> None:
    limiter = FixedWindowLimiter(limit=1, window_seconds=60)
    assert asyncio.run(limiter.allow("same-client")) is True
    assert asyncio.run(limiter.allow("same-client")) is False


def test_sse_endpoint_emits_trace_and_final_result(
    service: IncidentCopilotService, monkeypatch
) -> None:
    monkeypatch.setattr(main_module, "service", service)
    with TestClient(main_module.app) as client:
        response = client.post(
            "/api/diagnose/stream",
            json={
                "message": "DB-POOL-503表示什么，有哪些常见原因",
                "domain": "enterprise",
                "session_id": "api-stream",
            },
        )
    assert response.status_code == 200
    assert "event: trace" in response.text
    assert "event: result" in response.text
