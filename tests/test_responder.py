import pytest

from catman_io.actions import ActionContext, default_registry
from catman_io.actions.timers import TimerService
from catman_io.brain.catman import CatmanError
from catman_io.config import Config
from catman_io.dialog import Turn
from catman_io.intent import Intent, rule_store
from catman_io.intent.llm import CHAT, DELEGATE
from catman_io.intent.router import IntentRouter
from catman_io.journal import Journal
from catman_io.responder import (
    BRAIN_ERROR_PHRASE,
    DELEGATED_PHRASE,
    NO_BRAIN_PHRASE,
    NO_CATMAN_PHRASE,
    IntentResponder,
)
from tests.test_pipeline import FakeASR  # noqa: F401  (确保 tests 可作为包导入)


class FakeLLM:
    def __init__(self, intent=None, error=None):
        self.intent, self.error = intent, error

    def recognize(self, text):
        if self.error:
            raise self.error
        return self.intent


class FakeBrain:
    def __init__(self, chunks=("你好，", "我係貓人。", "有乜可以幫你"), error=None):
        self.chunks, self.error, self.cancelled, self.calls = chunks, error, 0, []

    def chat(self, text, *, session="default"):
        self.calls.append((text, session))
        if self.error:
            raise self.error
        yield from self.chunks

    def cancel(self):
        self.cancelled += 1


class FakeCatman:
    def __init__(self, fail=False):
        self.posted, self.fail = [], fail

    def post(self, text):
        if self.fail:
            raise CatmanError("down")
        self.posted.append(text)


class FakeSpeaker:
    volume = 0.8

    def stop(self):
        pass


def make(tmp_path, *, llm=None, brain=None, catman=None, answer_chat=False):
    cfg = Config.load(None)
    cfg.data_dir = str(tmp_path)
    cfg.intent.llm.answer_chat = answer_chat
    store = rule_store(cfg)
    router = IntentRouter(store, llm)
    clock = {"t": 50.0}
    ctx = ActionContext(
        cfg=cfg, timers=TimerService(), speaker=FakeSpeaker(), brain=brain, clock=lambda: clock["t"]
    )
    journal = Journal(tmp_path / "journal")
    r = IntentResponder(
        cfg,
        router,
        default_registry(),
        ctx,
        brain=brain,
        catman=catman,
        journal=journal,
        clock=lambda: clock["t"],
    )
    return r, ctx


def run(responder, text, transcript_text=None):
    from catman_io.asr import Transcript

    turn = Turn(id="t1", gen=1)
    spoken = []
    resp = responder.respond(turn, Transcript(transcript_text or text), lambda s: spoken.append(s) or True)
    return turn, spoken, resp


def test_rule_hit_runs_action(tmp_path):
    r, ctx = make(tmp_path)
    turn, spoken, resp = run(r, "而家幾點呀")
    assert spoken and spoken[0].startswith("而家") and resp.ok and not resp.followup
    assert (
        turn.info["intent"] == "time.now"
        and turn.info["tier"] == "rule"
        and turn.info["action"] == "builtin.time"
    )
    assert (
        turn.info["action_ok"]
        and turn.info["reply_source"] == "rule"
        and ctx.state["last_reply"] == spoken[0]
    )
    assert turn.info["t_intent_done"] == 50.0 and turn.info["t_action_done"] == 50.0
    turn, spoken, resp = run(r, "計時三秒")
    assert spoken == ["好，三秒後叫你。"] and ctx.timers.pending()[0].seconds == 3
    turn, spoken, resp = run(r, "再講一次")
    assert spoken == ["好，三秒後叫你。"]
    turn, spoken, resp = run(r, "set個timer")
    assert resp.followup and "幾耐" in spoken[0]


def test_asked_slot_is_filled_by_the_next_utterance(tmp_path):
    """计时没说时长：动作追问 → 下一句「三十分鐘」直接填进去执行；答非所问就作废；追问会过期。"""
    r, ctx = make(tmp_path)
    turn, spoken, resp = run(r, "幫我校個鬧鐘")
    assert spoken == ["要計幾耐呀？"] and resp.followup and turn.info["ask"] == "duration"
    assert ctx.state["pending"]["intent"] == "timer.set" and ctx.state["pending"]["slot"] == "duration"

    turn, spoken, resp = run(r, "三十分钟")  # 识别器给的是简体，归一化后再解析
    assert spoken == ["好，半個鐘後叫你。"] and resp.ok and not resp.followup  # 1800 s 念成半個鐘
    assert turn.info["intent"] == "timer.set" and turn.info["tier"] == "followup"
    assert turn.info["slots"] == {"duration": 1800} and turn.info["route_flags"] == ["slot_filled"]
    assert ctx.timers.pending()[0].seconds == 1800 and "pending" not in ctx.state

    # 答非所问：走正常路由，追问作废
    ctx.timers.cancel_all()
    run(r, "set個timer")
    turn, spoken, resp = run(r, "而家幾點呀")
    assert turn.info["intent"] == "time.now" and turn.info["tier"] == "rule" and "pending" not in ctx.state
    turn, spoken, resp = run(r, "十個字")
    assert turn.info["intent"] is None if "intent" in turn.info else True
    assert spoken == [NO_BRAIN_PHRASE] and not ctx.timers.pending()

    # 追问过期：30 秒后再答也不算
    run(r, "set個timer")
    ctx.state["pending"]["expires"] = r.clock() - 1
    turn, spoken, resp = run(r, "十個字")
    assert spoken == [NO_BRAIN_PHRASE] and not ctx.timers.pending() and "pending" not in ctx.state


def test_llm_intent_and_candidate_case(tmp_path):
    r, ctx = make(tmp_path, llm=FakeLLM(Intent("volume.set", {"percent": 30}, 0.8, "llm", None, "raw")))
    turn, spoken, resp = run(r, "聲音調去三成")
    assert spoken == ["音量30%。"] and turn.info["tier"] == "llm"
    assert turn.info["candidate_case"]["intent"] == "volume.set" and turn.info["llm_raw"] == "raw"
    assert "rules_miss_llm_hit" in turn.info["route_flags"]


def test_chat_streams_sentences_to_brain(tmp_path):
    brain = FakeBrain()
    r, ctx = make(tmp_path, brain=brain)
    turn, spoken, resp = run(r, "你係邊個呀你")
    assert spoken == ["你好，我係貓人。", "有乜可以幫你"] and resp.followup and resp.ok
    assert brain.calls == [("你係邊個呀你", "voice")] and turn.info["reply_source"] == "brain"
    assert turn.info["t_brain_first"] == 50.0 and ctx.state["last_reply"] == "你好，我係貓人。有乜可以幫你"
    r.cancel()
    assert brain.cancelled == 1


def test_chat_without_brain_and_brain_error(tmp_path):
    r, ctx = make(tmp_path)
    turn, spoken, resp = run(r, "你鍾意食乜")
    assert spoken == [NO_BRAIN_PHRASE] and not resp.ok and turn.info["reply_source"] == "brain_missing"
    r, ctx = make(tmp_path, brain=FakeBrain(error=RuntimeError("boom")))
    turn, spoken, resp = run(r, "你鍾意食乜")
    assert (
        spoken == [BRAIN_ERROR_PHRASE] and turn.info["reply_source"] == "brain_error" and resp.error == "boom"
    )


def test_llm_chat_reply_spoken_directly_only_when_configured(tmp_path):
    llm = FakeLLM(Intent(CHAT, {"reply": "我係貓人"}, 0.8, "llm"))
    brain = FakeBrain()
    r, ctx = make(tmp_path, llm=llm, brain=brain, answer_chat=True)
    turn, spoken, resp = run(r, "你叫咩名")
    assert spoken == ["我係貓人"] and turn.info["reply_source"] == "llm" and brain.calls == []
    r, ctx = make(tmp_path, llm=llm, brain=brain, answer_chat=False)
    turn, spoken, resp = run(r, "你叫咩名")
    assert brain.calls and turn.info["reply_source"] == "brain"


def test_delegate_posts_to_catman(tmp_path):
    llm = FakeLLM(Intent(DELEGATE, {"task": "睇下路由器"}, 0.8, "llm"))
    catman = FakeCatman()
    r, ctx = make(tmp_path, llm=llm, catman=catman)
    turn, spoken, resp = run(r, "幫我睇下路由器")
    assert (
        catman.posted == ["睇下路由器"] and spoken == [DELEGATED_PHRASE] and turn.info["action"] == "delegate"
    )
    r, ctx = make(tmp_path, llm=llm, catman=None)
    turn, spoken, resp = run(r, "幫我睇下路由器")
    assert spoken == [NO_CATMAN_PHRASE] and not resp.ok
    r, ctx = make(tmp_path, llm=llm, catman=FakeCatman(fail=True))
    turn, spoken, resp = run(r, "幫我睇下路由器")
    assert spoken == [NO_CATMAN_PHRASE] and "down" in resp.error


def test_deliver_alarm_and_shadow_amend(tmp_path):
    r, ctx = make(tmp_path)
    turn = Turn(id="a1", gen=2, kind="alarm")
    turn.payload = ctx.timers.add(0.0, 600, "十分鐘")
    spoken = []
    resp = r.deliver(turn, lambda s: spoken.append(s) or True)
    assert spoken == ["十分鐘到喇"] and resp.followup and turn.info["intent"] == "alarm"
    r.journal.write({"turn_id": "t9", "flags": []})
    r._on_shadow("而家幾點", Intent("time.now"), Intent("date.today", tier="llm"), "t9")
    rec = r.journal.get("t9")
    assert rec["shadow"] == {"intent": "date.today", "slots": {}, "agree": False} and rec["flags"] == [
        "rules_llm_disagree"
    ]


def test_cancelled_turn_does_nothing_more(tmp_path):
    r, ctx = make(tmp_path, brain=FakeBrain())
    turn = Turn(id="t1", gen=1)
    turn.cancelled.set()
    from catman_io.asr import Transcript

    spoken = []
    r.respond(turn, Transcript("你好"), lambda s: spoken.append(s) or True)
    assert spoken == []


@pytest.mark.parametrize("text", ["唔該幫我較個五分鐘嘅鬧鐘", "聽日會唔會落雨"])
def test_common_commands_route_by_rules(tmp_path, text):
    r, ctx = make(tmp_path)
    turn, spoken, resp = run(r, text)
    assert turn.info["tier"] == "rule" and spoken
