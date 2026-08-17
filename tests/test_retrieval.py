from app.service import GameDevCopilotService


def test_exact_bug_id_is_ranked_first(service: GameDevCopilotService) -> None:
    hits = service.search("GAME-BUG-037", domain="game", top_k=3)
    assert hits[0].chunk.metadata["bug_id"] == "GAME-BUG-037"
    assert hits[0].metadata_score > 0


def test_game_equipment_incident_retrieves_history(service: GameDevCopilotService) -> None:
    hits = service.search(
        "Android 1.3连续切换装备闪退 E-EQP-500", domain="game", top_k=3
    )
    assert any("bug_tickets/history.json" in hit.chunk.source for hit in hits)


def test_enterprise_pool_error_is_isolated(service: GameDevCopilotService) -> None:
    hits = service.search("DB-POOL-503 订单500", domain="enterprise", top_k=4)
    assert hits
    assert all(hit.chunk.domain == "enterprise" for hit in hits)
    assert any("order_database.log" in hit.chunk.source for hit in hits)


def test_doc_type_filter_only_returns_history(service: GameDevCopilotService) -> None:
    hits = service.search(
        "装备闪退",
        domain="game",
        top_k=5,
        filters={"doc_type": "bug_tickets"},
    )
    assert hits
    assert all(hit.chunk.doc_type == "bug_tickets" for hit in hits)

