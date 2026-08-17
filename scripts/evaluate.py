from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.service import GameDevCopilotService


def main() -> None:
    service = GameDevCopilotService(Settings())
    service.load_or_ingest()
    cases_path = service.settings.data_dir / "eval" / "cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    retrieval_hits = 0
    reciprocal_sum = 0.0
    category_hits = 0
    severity_hits = 0
    print("GameDev Copilot 离线评测")
    print("=" * 72)
    for index, case in enumerate(cases, start=1):
        results = service.search(
            case["question"], domain=case["domain"], top_k=3
        )
        ranks = [
            rank
            for rank, hit in enumerate(results, start=1)
            if case["expected_source"] in hit.chunk.source
        ]
        rank = ranks[0] if ranks else None
        if rank:
            retrieval_hits += 1
            reciprocal_sum += 1 / rank
        diagnosis = service.agent.diagnose(case["question"], case["domain"], results)
        category_hits += diagnosis.category == case["expected_category"]
        severity_hits += diagnosis.severity == case["expected_severity"]
        print(
            f"{index}. [{case['domain']}] {case['question']}\n"
            f"   来源排名={rank or '未命中'} | 分类={diagnosis.category} | "
            f"严重度={diagnosis.severity}"
        )
    count = len(cases)
    print("=" * 72)
    print(f"Hit Rate@3: {retrieval_hits / count:.2%}")
    print(f"MRR@3: {reciprocal_sum / count:.3f}")
    print(f"分类准确率: {category_hits / count:.2%}")
    print(f"严重度准确率: {severity_hits / count:.2%}")
    print("说明：这是小型演示集，不代表真实线上效果。")


if __name__ == "__main__":
    main()
