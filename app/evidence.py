from __future__ import annotations

import re

from app.models import EvidenceAssessment, SearchHit


_ERROR_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]{0,15}[-_][A-Z0-9-]{2,}(?![A-Za-z0-9])",
    re.IGNORECASE,
)


class EvidenceGate:
    """在生成回答前判断证据是否足够，避免“检索到内容就强行回答”。"""

    def __init__(self, minimum_score: float, minimum_sources: int = 1) -> None:
        self.minimum_score = minimum_score
        self.minimum_sources = minimum_sources

    def assess(self, query: str, hits: list[SearchHit]) -> EvidenceAssessment:
        if not hits:
            return EvidenceAssessment(False, 0.0, "没有召回任何资料", 0)
        unique_sources = {hit.chunk.source for hit in hits}
        top_score = hits[0].score
        second_score = hits[1].score if len(hits) > 1 else 0.0
        margin = max(top_score - second_score, 0.0)
        exact_codes = {item.lower() for item in _ERROR_CODE_RE.findall(query)}
        exact_matched = not exact_codes or any(hit.exact_score > 0.25 for hit in hits)

        if top_score < self.minimum_score:
            return EvidenceAssessment(
                False,
                round(top_score, 3),
                f"最高检索分数{top_score:.3f}低于阈值{self.minimum_score:.3f}",
                len(unique_sources),
            )
        if len(unique_sources) < self.minimum_sources:
            return EvidenceAssessment(
                False,
                round(top_score, 3),
                "可靠来源数量不足",
                len(unique_sources),
            )
        if not exact_matched:
            return EvidenceAssessment(
                False,
                round(top_score * 0.7, 3),
                "问题包含错误码，但知识库没有匹配该错误码的证据",
                len(unique_sources),
            )

        confidence = min(
            0.98,
            0.55 * top_score
            + 0.2 * min(len(unique_sources) / 3, 1.0)
            + 0.15 * min(margin * 4, 1.0)
            + 0.1 * (1.0 if exact_matched else 0.0),
        )
        return EvidenceAssessment(
            True,
            round(confidence, 3),
            "证据分数、来源数量和精确实体匹配均达到要求",
            len(unique_sources),
        )
