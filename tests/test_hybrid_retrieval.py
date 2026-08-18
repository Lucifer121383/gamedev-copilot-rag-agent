from pathlib import Path

from app.config import Settings
from app.service import IncidentCopilotService


def test_hybrid_scores_and_faiss_artifacts_are_exposed(
    service: IncidentCopilotService,
) -> None:
    hits = service.search("Android装备闪退E-EQP-500", domain="game", top_k=3)
    assert hits
    assert hits[0].bm25_score > 0
    assert hits[0].dense_score > 0
    assert hits[0].fusion_score > 0
    assert hits[0].rerank_score > 0
    assert (service.settings.storage_dir / "faiss.index").exists()
    assert (service.settings.storage_dir / "dense_embeddings.npy").exists()


def test_persisted_index_can_be_reloaded(tmp_path: Path) -> None:
    storage = tmp_path / "reload-storage"
    settings = Settings(
        data_dir=Path(__file__).resolve().parents[1] / "data",
        storage_dir=storage,
        embedding_backend="hashing",
        llm_base_url="",
        llm_api_key="",
        llm_model="",
    )
    first = IncidentCopilotService(settings)
    first.ingest()
    first_result = first.search("DB-POOL-503", domain="enterprise", top_k=3)
    first.close()

    second = IncidentCopilotService(settings)
    second.load_or_ingest()
    second_result = second.search("DB-POOL-503", domain="enterprise", top_k=3)
    assert [hit.chunk.chunk_id for hit in first_result] == [
        hit.chunk.chunk_id for hit in second_result
    ]
    second.close()


def test_metadata_filter_is_applied_before_final_ranking(
    service: IncidentCopilotService,
) -> None:
    hits = service.search(
        "装备闪退",
        domain="game",
        top_k=5,
        filters={"platform": "Android", "doc_type": "bug_tickets"},
    )
    assert hits
    assert all(hit.chunk.doc_type == "bug_tickets" for hit in hits)
    assert all("Android" in str(hit.chunk.metadata.get("platform")) for hit in hits)


def test_chinese_pool_metrics_recall_english_key_value_log(
    service: IncidentCopilotService,
) -> None:
    hits = service.search(
        "订单服务连接池活跃连接达到50且空闲为0如何排查",
        domain="enterprise",
        top_k=3,
    )
    assert any("error_logs/order_database.log" in hit.chunk.source for hit in hits)
