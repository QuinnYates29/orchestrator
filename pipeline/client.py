"""Thin async client for the orchestrator's OpenAI-compatible + /admin API.

Streaming is exposed as an async generator of StreamEvent so callers (the
Ornith worker loop) can observe reasoning_content and tool_calls as they
arrive, not just after a turn completes - that live visibility is what lets
the supervisor watch for stuck-thinking loops mid-generation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx


class OrchestratorError(RuntimeError):
    """Raised for a non-200 orchestrator response. Callers decide what a
    given failure means (kill+retry for a worker, abort for planner/merger)."""

    def __init__(self, status_code: int, body: Any):
        self.status_code = status_code
        self.body = body
        super().__init__(f"orchestrator returned {status_code}: {body}")


@dataclass
class StreamEvent:
    content_delta: str = ""
    reasoning_delta: str = ""
    tool_calls: list[dict] | None = None   # only set on the final event, fully accumulated + json-parsed
    finish_reason: str | None = None
    usage: dict | None = None


@dataclass
class _ToolCallAccumulator:
    id: str = ""
    name: str = ""
    arguments_buf: str = ""

    def finalize(self) -> dict:
        try:
            args = json.loads(self.arguments_buf) if self.arguments_buf else {}
        except json.JSONDecodeError:
            args = {"_raw": self.arguments_buf}
        return {"id": self.id, "name": self.name, "arguments": args}


class OrchestratorClient:
    def __init__(self, base_url: str, admin_url: str, api_key: str | None = None,
                 timeout_s: float = 3600.0):
        self.base_url = base_url.rstrip("/")
        self.admin_url = admin_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(timeout_s, connect=10.0))

    async def aclose(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    # -- admin: model lifecycle ----------------------------------------

    async def admin_load(self, model: str, profile: str | None = None, wait_s: float = 180.0) -> dict:
        body: dict = {"model": model, "wait_s": wait_s}
        if profile:
            body["profile"] = profile
        r = await self._client.post(f"{self.admin_url}/admin/load", json=body)
        data = r.json()
        if r.status_code != 200:
            raise OrchestratorError(r.status_code, data)
        return data

    async def admin_unload(self, model: str, wait_idle_s: float = 0.0, force: bool = False) -> dict:
        r = await self._client.post(
            f"{self.admin_url}/admin/unload",
            json={"model": model, "wait_idle_s": wait_idle_s, "force": force},
        )
        data = r.json()
        if r.status_code != 200:
            raise OrchestratorError(r.status_code, data)
        return data

    async def admin_status(self) -> dict:
        r = await self._client.get(f"{self.admin_url}/admin/status")
        return r.json()

    # -- chat: non-streaming (planner/merger - no live supervision needed) --

    async def chat_once(self, model: str, messages: list[dict], *, tools: list[dict] | None = None,
                         tool_choice: Any = None, max_tokens: int | None = None,
                         temperature: float | None = None, **extra) -> dict:
        body = self._build_body(model, messages, tools=tools, tool_choice=tool_choice,
                                 max_tokens=max_tokens, temperature=temperature, stream=False, **extra)
        r = await self._client.post(f"{self.base_url}/chat/completions", json=body)
        data = r.json()
        if r.status_code != 200:
            raise OrchestratorError(r.status_code, data)
        return data

    # -- chat: streaming (worker loop - supervisor needs live progress) -----

    async def chat_stream(self, model: str, messages: list[dict], *, tools: list[dict] | None = None,
                           tool_choice: Any = None, max_tokens: int | None = None,
                           temperature: float | None = None, **extra) -> AsyncIterator[StreamEvent]:
        body = self._build_body(model, messages, tools=tools, tool_choice=tool_choice,
                                 max_tokens=max_tokens, temperature=temperature, stream=True, **extra)
        tool_acc: dict[int, _ToolCallAccumulator] = {}
        async with self._client.stream("POST", f"{self.base_url}/chat/completions", json=body) as r:
            if r.status_code != 200:
                content = await r.aread()
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    data = {"error": {"message": content.decode(errors="replace")}}
                raise OrchestratorError(r.status_code, data)

            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                finish_reason = choice.get("finish_reason")

                event = StreamEvent(
                    content_delta=delta.get("content") or "",
                    reasoning_delta=delta.get("reasoning_content") or "",
                    usage=chunk.get("usage"),
                )
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    acc = tool_acc.setdefault(idx, _ToolCallAccumulator())
                    if tc.get("id"):
                        acc.id = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        acc.name = fn["name"]
                    if fn.get("arguments"):
                        acc.arguments_buf += fn["arguments"]

                if finish_reason:
                    event.finish_reason = finish_reason
                    if tool_acc:
                        event.tool_calls = [tool_acc[i].finalize() for i in sorted(tool_acc)]
                yield event

    def _build_body(self, model: str, messages: list[dict], *, tools, tool_choice,
                     max_tokens, temperature, stream: bool, **extra) -> dict:
        body: dict = {"model": model, "messages": messages, "stream": stream}
        if tools:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        body.update(extra)
        return body
