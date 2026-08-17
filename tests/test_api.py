import os
import tempfile

os.environ.setdefault("RAG_STORAGE_DIR", tempfile.mkdtemp(prefix="gamedev-api-test-"))

from fastapi.testclient import TestClient

from app import main as main_module
from app.service import GameDevCopilotService


def test_health_search_and_diagnose(
    service: GameDevCopilotService, monkeypatch
) -> None:
    monkeypatch.setattr(main_module, "service", service)
    with TestClient(main_module.app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["ready"] is True

        search = client.post(
            "/api/search",
            json={
                "query": "E-EQP-500装备闪退",
                "domain": "game",
                "top_k": 3,
            },
        )
        assert search.status_code == 200
        assert all(item["domain"] == "game" for item in search.json()["sources"])

        response = client.post(
            "/api/diagnose",
            json={
                "message": "1.3版本Android装备闪退E-EQP-500",
                "domain": "game",
                "session_id": "api-game-test",
            },
        )
        data = response.json()
        assert response.status_code == 200
        assert data["diagnosis"]["severity"] == "P1"
        assert data["action"] == "human_review"
        assert len(data["trace"]) >= 8


def test_bad_workspace_and_bad_session_are_rejected(
    service: GameDevCopilotService, monkeypatch
) -> None:
    monkeypatch.setattr(main_module, "service", service)
    with TestClient(main_module.app) as client:
        bad_domain = client.post(
            "/api/search", json={"query": "测试问题", "domain": "medical"}
        )
        bad_session = client.post(
            "/api/diagnose",
            json={"message": "测试问题", "domain": "game", "session_id": "../bad"},
        )
        assert bad_domain.status_code == 422
        assert bad_session.status_code == 422

