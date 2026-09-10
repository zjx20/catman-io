import json

import pytest

from catman_io.llm import LLMError, OpenAICompatClient
from catman_io.llm.openai_compat import HttpResponse, iter_sse_content, parse_chat_result


class FakeTransport:
    def __init__(self, status=200, body=b"", lines=None, raise_kind=None):
        self.status, self.body, self.lines, self.raise_kind = status, body, lines, raise_kind
        self.requests = []

    def __call__(self, req):
        self.requests.append(req)
        if self.raise_kind:
            raise LLMError("boom", kind=self.raise_kind)
        if self.lines is not None:
            return HttpResponse(self.status, None, iter(self.lines))
        return HttpResponse(self.status, self.body)


def test_chat_sends_tools_and_parses_tool_call():
    body = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {"name": "timer_set", "arguments": '{"duration": "十分鐘"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"total_tokens": 42},
        }
    ).encode()
    t = FakeTransport(body=body)
    c = OpenAICompatClient("https://api.example.com/v1/", api_key="k", model="m", timeout=3, transport=t)
    res = c.chat(
        [{"role": "user", "content": "hi"}], tools=[{"type": "function"}], tool_choice="auto", timeout=2.5
    )
    req = t.requests[0]
    assert req.url == "https://api.example.com/v1/chat/completions" and req.timeout == 2.5
    assert req.headers["Authorization"] == "Bearer k" and not req.stream
    payload = json.loads(req.body)
    assert payload["model"] == "m" and payload["tools"] and payload["tool_choice"] == "auto"
    assert payload["temperature"] == 0.0
    assert res.tool_calls[0].name == "timer_set" and res.tool_calls[0].arguments == {"duration": "十分鐘"}
    assert res.tool_calls[0].id == "c1" and res.finish_reason == "tool_calls" and res.content is None


def test_http_error_and_timeout_map_to_llm_error():
    t = FakeTransport(status=401, body=b'{"error": "bad key"}')
    c = OpenAICompatClient("https://x", api_key=None, model="m", transport=t)
    with pytest.raises(LLMError) as e:
        c.chat([])
    assert e.value.kind == "http" and e.value.status == 401 and "bad key" in str(e.value)
    assert "Authorization" not in t.requests[0].headers
    with pytest.raises(LLMError) as e:
        OpenAICompatClient(
            "https://x", api_key=None, model="m", transport=FakeTransport(raise_kind="timeout")
        ).chat([])
    assert e.value.kind == "timeout"
    with pytest.raises(LLMError) as e:
        OpenAICompatClient("https://x", api_key=None, model="m", transport=FakeTransport(body=b"nope")).chat(
            []
        )
    assert e.value.kind == "protocol"


def test_parse_chat_result_tolerates_bad_arguments():
    res = parse_chat_result(
        {
            "choices": [
                {
                    "message": {
                        "content": "hi",
                        "tool_calls": [{"function": {"name": "x", "arguments": "{oops"}}],
                    }
                }
            ]
        }
    )
    assert res.content == "hi" and res.tool_calls[0].arguments == {"_raw": "{oops"}
    with pytest.raises(LLMError):
        parse_chat_result({"nothing": 1})


def test_stream_yields_content_until_done():
    def chunk(text):
        return b"data: " + json.dumps({"choices": [{"delta": {"content": text}}]}).encode()

    lines = [
        b": ping",
        chunk("你"),
        b"",
        chunk("好"),
        b'data: {"choices": [{"delta": {}}]}',
        b"data: [DONE]",
        chunk("x"),
    ]
    assert list(iter_sse_content(iter(lines))) == ["你", "好"]
    t = FakeTransport(lines=lines)
    c = OpenAICompatClient("https://x", api_key="k", model="m", transport=t)
    assert "".join(c.stream([{"role": "user", "content": "hi"}], max_tokens=10)) == "你好"
    payload = json.loads(t.requests[0].body)
    assert payload["stream"] is True and payload["max_tokens"] == 10 and t.requests[0].stream
