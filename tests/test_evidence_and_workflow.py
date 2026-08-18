from app.evidence import EvidenceGate
from app.models import Chunk, SearchHit
from app.service import IncidentCopilotService


def test_evidence_gate_rejects_empty_results() -> None:
    assessment = EvidenceGate(0.2).assess("未知问题", [])
    assert assessment.sufficient is False
    assert assessment.source_count == 0


def test_evidence_gate_rejects_unmatched_error_code() -> None:
    hit = SearchHit(
        Chunk("1", "普通登录说明", "login.md", "game", "docs"),
        score=0.9,
        exact_score=0.0,
    )
    assessment = EvidenceGate(0.2).assess("出现ERR-NOT-999", [hit])
    assert assessment.sufficient is False
    assert "错误码" in assessment.reason


def test_write_tool_pauses_for_human_approval(
    service: IncidentCopilotService,
) -> None:
    pending = service.diagnose(
        message="Android装备闪退E-EQP-500，请创建Bug工单",
        domain="game",
        session_id="approval-flow",
    )
    assert pending["status"] == "awaiting_approval"
    assert pending["action"] == "awaiting_approval"
    assert pending["approval"]["type"] == "write_tool_approval"
    assert service.store.ticket_stats("game")["total"] == 0

    completed = service.approve(
        pending["request_id"], approved=True, reason="确认创建测试工单"
    )
    assert completed["status"] == "completed"
    assert completed["action"] == "create_ticket"
    assert completed["ticket"] is not None
    assert any(item["node"] == "human_approval" for item in completed["trace"])


def test_rejected_write_does_not_create_ticket(
    service: IncidentCopilotService,
) -> None:
    pending = service.diagnose(
        message="订单服务DB-POOL-503，请创建故障工单",
        domain="enterprise",
        session_id="reject-flow",
    )
    completed = service.approve(
        pending["request_id"], approved=False, reason="只查看诊断，不写入"
    )
    assert completed["action"] == "write_rejected"
    assert completed["ticket"] is None
    assert service.store.ticket_stats("enterprise")["total"] == 0


def test_insufficient_evidence_blocks_write_before_approval(
    service: IncidentCopilotService,
) -> None:
    result = service.diagnose(
        message="火星数据库出现ERR-MARS-999，请创建Bug工单",
        domain="enterprise",
        session_id="no-evidence-write",
    )
    assert result["status"] == "completed"
    assert result["action"] == "human_review"
    assert result["evidence"]["sufficient"] is False
    assert result["ticket"] is None


def test_approval_resume_is_idempotent(service: IncidentCopilotService) -> None:
    pending = service.diagnose(
        message="Android装备闪退E-EQP-500，请创建Bug工单",
        domain="game",
        session_id="resume-idempotent",
    )
    first = service.approve(pending["request_id"], approved=True)
    second = service.approve(pending["request_id"], approved=True)
    assert first["ticket"]["ticket_no"] == second["ticket"]["ticket_no"]
