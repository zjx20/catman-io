import json

import pytest

from catman_io.brain import build_brain
from catman_io.brain.catman import CatmanClient, CatmanError, build_catman
from catman_io.brain.llm import DEFAULT_SYSTEM_PROMPT, LLMBrain
from catman_io.config import Config


class FakeClient:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def stream(self, messages, **kw):
        self.calls.append((messages, kw))
        yield from self.chunks


def test_llm_brain_streams_and_remembers():
    clock = {"t": 1000.0}
    client = FakeClient(["你", "好", "。"])
    brain = LLMBrain(client, memory_minutes=10, memory_turns=2, clock=lambda: clock["t"])
    assert "".join(brain.chat("hi")) == "你好。"
    messages, kw = client.calls[0]
    assert messages[0] == {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
    assert messages[-1] == {"role": "user", "content": "hi"} and kw["max_tokens"] == 400
    list(brain.chat("再講"))
    messages, _ = client.calls[1]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[2]["content"] == "你好。"
    clock["t"] += 11 * 60
    list(brain.chat("仲記唔記得"))
    assert [m["role"] for m in client.calls[2][0]] == ["system", "user"]  # 记忆过期
    brain.forget()
    assert brain._memory == {}


def test_llm_brain_cancel_stops_mid_stream():
    def chunks():
        yield "第一句。"
        yield "第二句。"

    client = FakeClient(chunks())
    brain = LLMBrain(client)
    it = brain.chat("x")
    assert next(it) == "第一句。"
    brain.cancel()
    assert list(it) == []
    # 已经说出的部分仍进记忆
    assert brain._memory["default"][0].assistant == "第一句。"


def test_catman_client_post_and_probe():
    calls = []

    def http(method, url, headers, body, timeout):
        calls.append((method, url, headers, body))
        if url.endswith("/health"):
            return 200, json.dumps({"bootOk": True, "inFlight": {"foreground": 0}}).encode()
        return (403, b"forbidden") if headers.get("X-Catman-Token") != "tok" else (200, b'{"ok":true}')

    c = CatmanClient("http://catman:8787/", token="tok", http=http)
    c.post("你好")
    method, url, headers, body = calls[0]
    assert method == "POST" and url == "http://catman:8787/api/chat" and json.loads(body) == {"text": "你好"}
    assert headers["X-Catman-Token"] == "tok"
    assert c.probe()["bootOk"] is True
    bad = CatmanClient("http://catman:8787", token="nope", http=http)
    with pytest.raises(CatmanError, match="403"):
        bad.post("x")

    def down(*a):
        raise OSError("refused")

    assert CatmanClient("http://x", token="t", http=down).probe() is None
    with pytest.raises(CatmanError, match="unreachable"):
        CatmanClient("http://x", token="t", http=down).post("x")


def test_builders_follow_config(monkeypatch):
    cfg = Config.load(None)
    assert build_brain(cfg) is None and build_catman(cfg) is None
    cfg.brain.llm.enabled = True
    cfg.brain.llm.model = "m"
    monkeypatch.setenv("CATMAN_IO_LLM_API_KEY", "k")
    brain = build_brain(cfg)
    assert isinstance(brain, LLMBrain) and brain.client.model == "m"
    cfg.brain.catman.base_url = "http://catman:8787"
    monkeypatch.setenv("CATMAN_ADMIN_TOKEN", "t")
    assert build_catman(cfg).token == "t"
