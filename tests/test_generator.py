from pathlib import Path

from app.config import Settings
from app.generator import DiagnosisGenerator
from app.llm_client import ChatResult
from app.models import BugDiagnosis, Chunk, EvidenceAssessment, SearchHit


class FakeGenerationClient:
    enabled = True

    def __init__(self, content: str) -> None:
        self.content = content
        self.called = False

    def complete(self, *args, **kwargs) -> ChatResult:
        self.called = True
        return ChatResult(
            content=self.content,
            prompt_tokens=12,
            completion_tokens=8,
        )


def _inputs(tmp_path: Path):
    settings = Settings(storage_dir=tmp_path / "generator")
    diagnosis = BugDiagnosis(
        category="客户端崩溃",
        severity="P1",
        platform="Android",
        module="装备系统",
        version="1.3",
        error_code="E-EQP-500",
        possible_causes=["资源句柄为空"],
        reproduction_steps=["连续切换装备"],
        regression_tests=["重复点击确认"],
        confidence=0.9,
        needs_human_review=True,
    )
    hit = SearchHit(
        chunk=Chunk(
            "chunk-1",
            "错误码E-EQP-500与装备资源句柄为空有关",
            "game/crash.log",
            "game",
            "crash_logs",
        ),
        score=0.9,
        exact_score=0.8,
    )
    return settings, diagnosis, [hit]


def test_valid_model_citation_is_accepted(tmp_path: Path) -> None:
    settings, diagnosis, hits = _inputs(tmp_path)
    client = FakeGenerationClient("建议先检查资源句柄并人工复核。[来源1]")
    result = DiagnosisGenerator(settings, client).generate(
        "为什么闪退", diagnosis, hits, [], None, EvidenceAssessment(True, 0.9, "充分", 1)
    )
    assert result.mode == "llm_grounded"
    assert result.citation_valid is True
    assert result.total_tokens == 20


def test_invalid_model_citation_uses_verifiable_fallback(tmp_path: Path) -> None:
    settings, diagnosis, hits = _inputs(tmp_path)
    client = FakeGenerationClient("这是没有引用的自由结论")
    result = DiagnosisGenerator(settings, client).generate(
        "为什么闪退", diagnosis, hits, [], None, EvidenceAssessment(True, 0.9, "充分", 1)
    )
    assert result.mode == "structured_fallback"
    assert result.citation_valid is False
    assert "引用" in (result.warning or "")


def test_insufficient_evidence_refuses_without_calling_model(tmp_path: Path) -> None:
    settings, diagnosis, hits = _inputs(tmp_path)
    client = FakeGenerationClient("不应被调用")
    result = DiagnosisGenerator(settings, client).generate(
        "为什么闪退", diagnosis, hits, [], None, EvidenceAssessment(False, 0.1, "不足", 1)
    )
    assert result.mode == "evidence_refusal"
    assert client.called is False
    assert "不能可靠确认根因" in result.answer
