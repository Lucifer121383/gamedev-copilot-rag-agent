from __future__ import annotations

import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.models import EvidenceAssessment
from app.service import IncidentCopilotService


def _source_rank(results: list[Any], expected_sources: list[str]) -> int | None:
    for rank, hit in enumerate(results, start=1):
        if any(expected in hit.chunk.source for expected in expected_sources):
            return rank
    return None


def evaluate(service: IncidentCopilotService, cases: list[dict[str, Any]]) -> dict[str, Any]:
    hit_at_3 = 0
    reciprocal_sum = 0.0
    recall_sum = 0.0
    category_hits = 0
    severity_hits = 0
    classified = 0
    answerability_hits = 0
    citation_hits = 0
    tool_policy_hits = 0
    latencies: list[float] = []
    bad_cases: list[dict[str, Any]] = []

    for case in cases:
        started = perf_counter()
        results = service.search(case["question"], domain=case["domain"], top_k=5)
        latency_ms = (perf_counter() - started) * 1000
        latencies.append(latency_ms)
        expected_sources = list(case.get("expected_sources") or [])
        rank = _source_rank(results, expected_sources) if expected_sources else None
        retrieved_sources = [hit.chunk.source for hit in results]

        if expected_sources:
            if rank is not None and rank <= 3:
                hit_at_3 += 1
            if rank is not None:
                reciprocal_sum += 1 / rank
            matched = sum(
                any(expected in source for source in retrieved_sources)
                for expected in expected_sources
            )
            recall_sum += matched / len(expected_sources)
        else:
            # No-answer cases do not participate in retrieval hit metrics.
            recall_sum += 1.0

        evidence = service.evidence_gate.assess(case["question"], results)
        answerability_correct = evidence.sufficient == bool(case["should_answer"])
        answerability_hits += answerability_correct
        diagnosis = service.agent.diagnose(case["question"], case["domain"], results)
        if "expected_category" in case:
            classified += 1
            category_hits += diagnosis.category == case["expected_category"]
            severity_hits += diagnosis.severity == case["expected_severity"]

        plan = service.agent.plan(
            message=case["question"],
            domain=case["domain"],
            hits=results,
            session_id=f"eval-{case['case_id'].lower()}",
            idempotency_key=f"eval-key-{case['case_id'].lower()}",
        )
        planned_write = any(call.name == "create_bug_ticket" for call in plan.tool_calls)
        expected_write = bool(case.get("should_write")) and evidence.sufficient
        tool_policy_correct = planned_write == expected_write
        tool_policy_hits += tool_policy_correct

        generated = service.generator.generate(
            case["question"],
            diagnosis,
            results,
            [],
            None,
            EvidenceAssessment(**evidence.to_dict()),
        )
        citation_correct = generated.citation_valid
        citation_hits += citation_correct

        retrieval_correct = (not expected_sources) or rank is not None
        if not (
            retrieval_correct
            and answerability_correct
            and tool_policy_correct
            and citation_correct
        ):
            bad_cases.append(
                {
                    "case_id": case["case_id"],
                    "type": case["type"],
                    "question": case["question"],
                    "expected_sources": expected_sources,
                    "retrieved_sources": retrieved_sources,
                    "rank": rank,
                    "evidence": evidence.to_dict(),
                    "planned_write": planned_write,
                    "expected_write": expected_write,
                    "citation_valid": citation_correct,
                }
            )

    count = len(cases)
    answerable_count = sum(bool(case.get("expected_sources")) for case in cases)
    sorted_latency = sorted(latencies)
    p95_index = max(0, min(len(sorted_latency) - 1, int(len(sorted_latency) * 0.95) - 1))
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "case_count": count,
        "answerable_case_count": answerable_count,
        "no_answer_case_count": count - answerable_count,
        "retrieval": {
            "hit_rate_at_3": round(hit_at_3 / max(answerable_count, 1), 4),
            "mrr_at_5": round(reciprocal_sum / max(answerable_count, 1), 4),
            "recall_at_5": round(recall_sum / count, 4),
        },
        "grounding": {
            "answerability_accuracy": round(answerability_hits / count, 4),
            "citation_validation_rate": round(citation_hits / count, 4),
        },
        "agent": {
            "category_accuracy": round(category_hits / max(classified, 1), 4),
            "severity_accuracy": round(severity_hits / max(classified, 1), 4),
            "tool_policy_accuracy": round(tool_policy_hits / count, 4),
        },
        "latency_ms": {
            "average": round(statistics.fmean(latencies), 2),
            "p95": round(sorted_latency[p95_index], 2),
            "maximum": round(max(latencies), 2),
        },
        "retrieval_config": service.retriever.describe(),
        "bad_case_count": len(bad_cases),
        "bad_cases": bad_cases,
        "notes": [
            "评测集为仓库内构造数据，只用于回归比较，不能代表真实生产效果。",
            "分类与严重度标签经过人工语义复核，标签修订不会删除或替换原问题。",
            "评测脚本禁用真实LLM调用，因此Token成本为0。",
            "Bad Case应人工复核并保留，不应为了追求100%而删除困难样本。",
        ],
    }


def main() -> None:
    settings = Settings(llm_base_url="", llm_api_key="", llm_model="")
    service = IncidentCopilotService(settings)
    try:
        service.load_or_ingest()
        cases_path = settings.data_dir / "eval" / "cases.json"
        cases = json.loads(cases_path.read_text(encoding="utf-8"))
        report = evaluate(service, cases)
        report_path = settings.storage_dir / "evaluation_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("IncidentCopilot离线评测")
        print("=" * 72)
        print(f"评测问题：{report['case_count']}条")
        print(f"Hit Rate@3：{report['retrieval']['hit_rate_at_3']:.2%}")
        print(f"MRR@5：{report['retrieval']['mrr_at_5']:.3f}")
        print(f"Recall@5：{report['retrieval']['recall_at_5']:.2%}")
        print(
            f"可回答/拒答准确率：{report['grounding']['answerability_accuracy']:.2%}"
        )
        print(f"工具策略准确率：{report['agent']['tool_policy_accuracy']:.2%}")
        print(f"平均检索延迟：{report['latency_ms']['average']:.2f}ms")
        print(f"Bad Case：{report['bad_case_count']}条")
        print(f"完整报告：{report_path}")
    finally:
        service.close()


if __name__ == "__main__":
    main()
