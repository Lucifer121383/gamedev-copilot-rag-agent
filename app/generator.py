from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings
from app.models import BugDiagnosis, SearchHit


SYSTEM_PROMPT = """你是软件研发故障诊断助手。
只能依据参考资料和工具结果分析，不得把可能原因描述成已确认事实。
每个关键结论必须标注来源编号，例如[来源1]。
P1问题、证据不足或涉及写操作时必须建议人工复核。
不得建议自动删除数据、回滚生产、重启生产服务或执行其他高风险操作。
请使用简洁、专业的中文。"""


@dataclass(slots=True)
class GenerationResult:
    answer: str
    mode: str
    warning: str | None = None


class DiagnosisGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @staticmethod
    def _fallback(
        diagnosis: BugDiagnosis,
        hits: list[SearchHit],
        tool_results: list[dict[str, Any]],
        ticket: dict[str, Any] | None,
    ) -> GenerationResult:
        source_note = ""
        if hits:
            source_note = "\n\n主要证据：\n" + "\n".join(
                f"- [来源{index}] {hit.chunk.source}，相关度{hit.score:.3f}"
                for index, hit in enumerate(hits[:3], start=1)
            )
        else:
            source_note = "\n\n当前没有检索到足够证据，以下内容只能作为排查框架。"

        history_result = next(
            (item for item in tool_results if item.get("tool") == "search_bug_history"),
            None,
        )
        similar_note = ""
        if history_result and history_result.get("ok"):
            items = history_result["data"].get("items", [])
            if items:
                labels = [
                    str(item.get("bug_id") or item.get("section") or item.get("source"))
                    for item in items[:3]
                ]
                similar_note = "\n相似历史问题：" + "、".join(labels)

        ticket_note = ""
        if ticket:
            cache_note = "，重复请求已复用原工单" if ticket.get("cached") else ""
            ticket_note = (
                f"\n\n已创建{ticket['severity']}工单 {ticket['ticket_no']}"
                f"，状态为{ticket['status']}{cache_note}。"
            )
        elif diagnosis.needs_human_review:
            ticket_note = "\n\n该问题需要人工复核。如需登记，请明确回复“创建Bug工单”。"

        answer = (
            f"问题分类：{diagnosis.category}\n"
            f"严重程度：{diagnosis.severity}\n"
            f"平台：{diagnosis.platform}\n"
            f"影响模块：{diagnosis.module}\n"
            f"版本：{diagnosis.version or '待确认'}\n"
            f"错误码：{diagnosis.error_code or '待确认'}\n"
            f"诊断置信度：{diagnosis.confidence:.0%}"
            f"{similar_note}\n\n"
            "可能原因：\n"
            + "\n".join(f"{index}. {value}" for index, value in enumerate(diagnosis.possible_causes, 1))
            + "\n\n建议复现步骤：\n"
            + "\n".join(f"{index}. {value}" for index, value in enumerate(diagnosis.reproduction_steps, 1))
            + "\n\n建议回归测试：\n"
            + "\n".join(f"{index}. {value}" for index, value in enumerate(diagnosis.regression_tests, 1))
            + source_note
            + ticket_note
        )
        return GenerationResult(answer=answer, mode="structured_fallback")

    def generate(
        self,
        question: str,
        diagnosis: BugDiagnosis,
        hits: list[SearchHit],
        tool_results: list[dict[str, Any]],
        ticket: dict[str, Any] | None,
    ) -> GenerationResult:
        fallback = self._fallback(diagnosis, hits, tool_results, ticket)
        if not self.settings.llm_enabled or not hits:
            return fallback

        context_blocks = []
        for index, hit in enumerate(hits, start=1):
            location = hit.chunk.source
            if hit.chunk.page is not None:
                location += f" 第{hit.chunk.page}页"
            if hit.chunk.section:
                location += f" {hit.chunk.section}"
            context_blocks.append(f"[来源{index}] {location}\n{hit.chunk.text}")
        prompt = (
            f"用户问题：{question}\n\n"
            f"结构化初步判断：{diagnosis.to_dict()}\n\n"
            f"工具结果：{tool_results}\n\n"
            "参考资料：\n"
            + "\n\n".join(context_blocks)
            + "\n\n请给出带来源的诊断、复现步骤、回归测试和人工复核建议。"
        )
        url = self.settings.llm_base_url
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.llm_api_key}",
        }
        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
        }
        if "api.deepseek.com" in self.settings.llm_base_url.lower():
            payload["thinking"] = {"type": "disabled"}

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=self.settings.llm_timeout_seconds) as client:
                    response = client.post(url, headers=headers, json=payload)
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        "模型服务临时不可用", request=response.request, response=response
                    )
                response.raise_for_status()
                data = response.json()
                answer = data["choices"][0]["message"]["content"].strip()
                if not answer:
                    raise ValueError("模型返回空内容")
                return GenerationResult(answer=answer, mode="llm")
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                last_error = exc
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500 and exc.response.status_code != 429:
                    break
                if attempt < 2:
                    time.sleep(0.15 * (2**attempt))
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = exc
                break
        fallback.warning = f"模型调用失败，已安全降级：{type(last_error).__name__}"
        return fallback

