"""
LLM client — OpenAI-compatible interface for Ollama and other providers.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from docmind.config import settings
from docmind.llm.provider import resolve_provider, normalize_base_url
from docmind.logging_config import get_logger

logger = get_logger("docmind.llm.client")


@dataclass
class LLMResponse:
    """Parsed LLM response with metadata."""
    content: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    raw_response: Optional[dict] = None


class LLMClient:
    """
    OpenAI-compatible LLM client.
    Supports Ollama (localhost) and any OpenAI-compatible API.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        provider = resolve_provider(settings)
        self.base_url = normalize_base_url(base_url, allow_local_http=True) if base_url else provider.base_url
        self.model = model or provider.model
        self.api_key = api_key or provider.api_key
        self.token_parameter = provider.token_parameter
        self.output_mode = provider.output_mode
        self.timeout = timeout_seconds or settings.llm_timeout_seconds
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url.rstrip("/") + "/",
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Call the LLM with chat messages.
        Supports structured output via response_format.
        """
        start = time.monotonic()

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            self.token_parameter: max_tokens or self.max_tokens,
        }

        effective_temperature = temperature if temperature is not None else settings.llm_temperature
        if effective_temperature is not None:
            payload["temperature"] = effective_temperature
        if response_format:
            payload["response_format"] = (
                {"type": "json_schema", "json_schema": response_format}
                if self.output_mode == "json_schema" else {"type": "json_object"}
            )

        try:
            response = await self.client.post("chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()

            choice = data["choices"][0]
            message = choice["message"]
            if choice.get("finish_reason") in {"length", "content_filter"} or message.get("refusal"):
                raise ValueError("LLM response was truncated or refused")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("LLM returned no text content")

            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")

            latency_ms = (time.monotonic() - start) * 1000

            logger.info(
                "llm_completion",
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=round(latency_ms, 2),
            )

            return LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                raw_response=data,
            )

        except httpx.TimeoutException:
            logger.error("llm_timeout", model=self.model, latency_ms=round((time.monotonic() - start) * 1000, 2))
            raise
        except httpx.HTTPStatusError as e:
            logger.error("llm_http_error", status=e.response.status_code)
            raise
        except Exception as e:
            logger.error("llm_error", error_type=type(e).__name__)
            raise

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
    ):
        """Stream chat completion (for future use)."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            self.token_parameter: self.max_tokens,
        }
        effective_temperature = temperature if temperature is not None else settings.llm_temperature
        if effective_temperature is not None:
            payload["temperature"] = effective_temperature
        if response_format:
            payload["response_format"] = (
                {"type": "json_schema", "json_schema": response_format}
                if self.output_mode == "json_schema" else {"type": "json_object"}
            )

        async with self.client.stream("POST", "chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    if line[6:].strip() == "[DONE]":
                        break
                    yield line[6:]


_default_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def reset_llm_client() -> None:
    global _default_client
    if _default_client is not None:
        import asyncio
        asyncio.run(_default_client.close())
    _default_client = None


async def close_llm_client() -> None:
    """Close an existing client without resolving unconfigured provider settings."""
    global _default_client
    if _default_client is not None:
        await _default_client.close()
        _default_client = None
