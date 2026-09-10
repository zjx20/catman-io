import datetime as dt
import random
import threading

import pytest

from catman_io.config import Config, IntentContextConfig
from catman_io.intent import BUILTIN_RULES, Intent, rule_store
from catman_io.intent.llm import CHAT, DELEGATE, LLMIntentRecognizer, normalize_slots
from catman_io.intent.router import IntentRouter, build_router
from catman_io.intent.rules import RuleSet
from catman_io.llm import ChatResult, LLMError, ToolCall


class FakeClient:
    def __init__(self, result=None, error=None, delay=0.0):
        self.result, self.error, self.delay = result, error, delay
        self.calls = []

    def chat(self, messages, **kw):
        self.calls.append((messages, kw))
        if self.delay:
            import time

            time.sleep(self.delay)
        if self.error:
            raise self.error
        return self.result


def tool(name, **args):
    return ChatResult(None, [ToolCall(name, args)])


@pytest.fixture
def rules():
    return RuleSet.load([BUILTIN_RULES])


def recognizer(client, rules, **kw):
    ctx = IntentContextConfig(rooms=["客廳|廳", "睡房|房"], home="三房一廳")
    return LLMIntentRecognizer(
        client, lambda: rules, context=ctx, now=lambda: dt.datetime(2026, 9, 10, 14, 5), **kw
    )


def test_tool_call_becomes_intent_with_normalized_slots(rules):
    client = FakeClient(tool("timer_set", duration="十分鐘"))
    got = recognizer(client, rules).recognize("幫我計十分鐘")
    assert got == Intent("timer.set", {"duration": 600}, 0.8, "llm", None, got.raw)
    messages, kw = client.calls[0]
    assert messages[0]["role"] == "system" and "2026年9月10日 星期四 14:05" in messages[0]["content"]
    assert "三房一廳" in messages[0]["content"] and "客廳、睡房" in messages[0]["content"]
    names = [t["function"]["name"] for t in kw["tools"]]
    assert names[-2:] == [CHAT, DELEGATE] and "timer_set" in names
    assert kw["tool_choice"] == "auto" and kw["temperature"] == 0.0


def test_chat_delegate_and_plain_text(rules):
    got = recognizer(FakeClient(tool(CHAT, reply="我係貓人")), rules).recognize("你叫咩名")
    assert got.name == CHAT and got.slots == {"reply": "我係貓人"}
    got = recognizer(FakeClient(tool(DELEGATE, task="睇下路由器內存")), rules).recognize("幫我睇下路由器")
    assert got.name == DELEGATE and got.slots["task"] == "睇下路由器內存"
    got = recognizer(FakeClient(ChatResult("而家好熱")), rules).recognize("熱唔熱")
    assert got.name == CHAT and got.slots["reply"] == "而家好熱" and got.confidence == 0.6
    assert recognizer(FakeClient(ChatResult("")), rules).recognize("x") is None
    assert recognizer(FakeClient(tool("bogus_tool")), rules).recognize("x") is None


def test_normalize_slots_accepts_model_formats(rules):
    types = {"duration": "duration", "time": "time", "date": "date", "percent": "percent", "room": "room"}
    got = normalize_slots(
        spec_types=types,
        values={"duration": 90, "time": "15:30", "date": "2026-10-01", "percent": "50", "room": "廳"},
        rs=rules,
    )
    assert got == {
        "duration": 90,
        "time": {"hour": 15, "minute": 30, "period": "pm"},
        "date": "2026-10-01",
        "percent": 50,
        "room": "客廳",
    }
    got = normalize_slots(
        spec_types=types, values={"duration": "半個鐘", "time": "夜晚八點", "date": "聽日"}, rs=rules
    )
    assert got["duration"] == 1800 and got["time"]["hour"] == 20 and len(got["date"]) == 10
    assert normalize_slots(spec_types=types, values={"duration": "亂講"}, rs=rules) == {}


def make_store(tmp_path):
    cfg = Config.load(None)
    cfg.data_dir = str(tmp_path)
    return rule_store(cfg)


def test_router_rule_then_llm_then_fallthrough(tmp_path, rules):
    store = make_store(tmp_path)
    client = FakeClient(tool("volume_set", percent=30))
    router = IntentRouter(store, recognizer(client, rules))
    res = router.route("而家幾點呀")
    assert res.tier == "rule" and res.intent.name == "time.now" and client.calls == []
    res = router.route("聲音調去三成")
    assert res.tier == "llm" and res.intent.name == "volume.set" and res.intent.slots == {"percent": 30}
    assert "rules_miss_llm_hit" in res.flags and res.elapsed_llm >= 0
    client.result = tool(CHAT)
    res = router.route("你鍾意食乜")
    assert res.tier == "llm" and res.intent.name == CHAT and "rules_miss_llm_hit" not in res.flags
    assert router.route("").tier == "none" and "asr_empty" in router.route("").flags


def test_router_llm_failures_are_flagged(tmp_path, rules):
    store = make_store(tmp_path)
    router = IntentRouter(store, recognizer(FakeClient(error=LLMError("t", kind="timeout")), rules))
    res = router.route("你鍾意食乜")
    assert res.tier == "none" and res.flags == ["llm_timeout", "fallthrough_chat"] and res.error == "t"
    router = IntentRouter(store, recognizer(FakeClient(error=RuntimeError("x")), rules))
    assert router.route("你鍾意食乜").flags == ["llm_error", "fallthrough_chat"]
    assert IntentRouter(store, None).route("你鍾意食乜").flags == ["fallthrough_chat"]


def test_router_shadow_check_runs_in_background(tmp_path, rules):
    store = make_store(tmp_path)
    seen = []
    done = threading.Event()

    def on_shadow(text, rule_intent, llm_intent, tag):
        seen.append((text, rule_intent.name, llm_intent.name if llm_intent else None, tag))
        done.set()

    client = FakeClient(tool("date_today"))
    router = IntentRouter(
        store, recognizer(client, rules), shadow_rate=1.0, rng=random.Random(1), on_shadow=on_shadow
    )
    res = router.route("而家幾點", shadow_tag="turn-1")
    assert res.tier == "rule"
    assert done.wait(2.0) and seen == [("而家幾點", "time.now", "date.today", "turn-1")]


def test_build_router_needs_key_only_when_enabled(tmp_path, monkeypatch):
    cfg = Config.load(None)
    cfg.data_dir = str(tmp_path)
    assert build_router(cfg).llm is None
    cfg.intent.llm.enabled = True
    cfg.intent.llm.model = "m"
    monkeypatch.delenv("CATMAN_IO_LLM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="CATMAN_IO_LLM_API_KEY"):
        build_router(cfg)
    monkeypatch.setenv("CATMAN_IO_LLM_API_KEY", "k")
    router = build_router(cfg)
    assert router.llm is not None and router.llm.client.model == "m"
