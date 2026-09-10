"""OpenAI 兼容的 ``/chat/completions`` 客户端：只用标准库（urllib），支持 tools 与流式 SSE。

Gemini 的兼容端点（``https://generativelanguage.googleapis.com/v1beta/openai``）、OpenAI、DeepSeek、
本地 vLLM 都是这个协议。``transport`` 可注入，测试不用起服务。
"""

from __future__ import annotations

import json
import logging
import socket
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    def __init__(self, message: str, *, kind: str = "error", status: int | None = None):
        super().__init__(message)
        self.kind = kind  # timeout | network | http | protocol
        self.status = status


@dataclass
class HttpRequest:
    url: str
    headers: dict[str, str]
    body: bytes
    timeout: float
    stream: bool = False


@dataclass
class HttpResponse:
    status: int
    body: bytes | None = None
    stream: Any = None  # 文件对象，流式时逐行读

    def iter_lines(self) -> Iterator[bytes]:
        if self.stream is not None:
            for line in self.stream:
                yield line.rstrip(b"\r\n")
        elif self.body:
            yield from self.body.splitlines()


Transport = Callable[[HttpRequest], HttpResponse]


def urllib_transport(req: HttpRequest) -> HttpResponse:
    r = urllib.request.Request(req.url, data=req.body, headers=req.headers, method="POST")
    try:
        resp = urllib.request.urlopen(r, timeout=req.timeout)  # noqa: S310
    except urllib.error.HTTPError as e:
        return HttpResponse(e.code, e.read())
    except TimeoutError as e:
        raise LLMError(f"timeout after {req.timeout}s", kind="timeout") from e
    except urllib.error.URLError as e:
        if isinstance(e.reason, (TimeoutError, socket.timeout)):
            raise LLMError(f"timeout after {req.timeout}s", kind="timeout") from e
        raise LLMError(f"request failed: {e.reason}", kind="network") from e
    if req.stream:
        return HttpResponse(resp.status, None, resp)
    return HttpResponse(resp.status, resp.read())


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = ""


@dataclass
class ChatResult:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None,
        model: str,
        timeout: float = 10.0,
        transport: Transport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.transport = transport or urllib_transport

    def _request(self, payload: dict[str, Any], *, stream: bool, timeout: float | None) -> HttpResponse:
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = HttpRequest(f"{self.base_url}/chat/completions", headers, body, timeout or self.timeout, stream)
        resp = self.transport(req)
        if resp.status >= 400:
            text = (resp.body or b"")[:500].decode("utf-8", "replace")
            raise LLMError(f"HTTP {resp.status}: {text}", kind="http", status=resp.status)
        return resp

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature}
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if extra:
            payload.update(extra)
        resp = self._request(payload, stream=False, timeout=timeout)
        try:
            data = json.loads(resp.body or b"{}")
        except ValueError as e:
            raise LLMError(f"bad JSON from server: {e}", kind="protocol") from e
        return parse_chat_result(data)

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """流式：逐段 yield 文本。"""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if extra:
            payload.update(extra)
        resp = self._request(payload, stream=True, timeout=timeout)
        yield from iter_sse_content(resp.iter_lines())


def parse_chat_result(data: dict[str, Any]) -> ChatResult:
    try:
        choice = data["choices"][0]
        msg = choice.get("message") or {}
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"unexpected response shape: {str(data)[:200]}", kind="protocol") from e
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                log.warning("tool call %s has non-JSON arguments: %r", fn.get("name"), args)
                args = {"_raw": args}
        calls.append(
            ToolCall(
                str(fn.get("name") or ""), args if isinstance(args, dict) else {}, str(tc.get("id") or "")
            )
        )
    return ChatResult(msg.get("content"), calls, choice.get("finish_reason"), data)


def iter_sse_content(lines: Iterator[bytes]) -> Iterator[str]:
    for raw in lines:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        try:
            data = json.loads(payload)
        except ValueError:
            continue
        for choice in data.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                yield content
