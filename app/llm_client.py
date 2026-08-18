from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import Settings


@dataclass(slots=True)
class ChatResult:
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChatCompletionClient:
    """兼容OpenAI Chat Completions协议的可靠客户端。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return self.settings.llm_enabled

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        if not self.enabled:
            raise RuntimeError("大模型接口未配置")
        url = self.settings.llm_base_url
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.llm_api_key}",
        }
        payload: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": (
                self.settings.llm_temperature if temperature is None else temperature
            ),
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if "api.deepseek.com" in self.settings.llm_base_url.lower():
            payload["thinking"] = {"type": "disabled"}

        last_error: Exception | None = None
        max_attempts = self.settings.llm_max_retries + 1
        for attempt in range(max_attempts):
            try:
                with httpx.Client(timeout=self.settings.llm_timeout_seconds) as client:
                    response = client.post(url, headers=headers, json=payload)
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        "模型服务临时不可用",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                data = response.json()
                message = data["choices"][0]["message"]
                usage = data.get("usage") or {}
                return ChatResult(
                    content=(message.get("content") or "").strip(),
                    tool_calls=list(message.get("tool_calls") or []),
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    retries=attempt,
                )
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                last_error = exc
                retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response.status_code == 429 or exc.response.status_code >= 500
                )
                if not retryable or attempt + 1 >= max_attempts:
                    break
                time.sleep(0.2 * (2**attempt))
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = exc
                break
        if last_error is None:
            raise RuntimeError("模型调用失败")
        raise RuntimeError(f"模型调用失败: {type(last_error).__name__}") from last_error
