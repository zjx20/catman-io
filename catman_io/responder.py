"""应答器：听清一句话之后——路由意图 → 执行动作 / 问大脑 / 交给 catman → 按句念出来，
并把过程记进 turn.info（日志用）。"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from typing import Any

from catman_io.actions import ActionContext, ActionRegistry
from catman_io.asr import Transcript
from catman_io.brain import Brain
from catman_io.brain.catman import CatmanClient, CatmanError
from catman_io.config import Config
from catman_io.dialog import Turn
from catman_io.intent import Intent
from catman_io.intent.llm import CHAT, DELEGATE
from catman_io.intent.router import IntentRouter
from catman_io.journal import Journal
from catman_io.pipeline import Response
from catman_io.tts import SENTENCE_END

log = logging.getLogger(__name__)

NO_BRAIN_PHRASE = "我淨係識做啲簡單嘢，呢個我答唔到。"
BRAIN_ERROR_PHRASE = "大腦連唔到，等陣再試。"
DELEGATED_PHRASE = "好，交咗俾 catman，搞掂會喺微信話你知。"
NO_CATMAN_PHRASE = "而家未接到 catman，做唔到呢件事。"


class IntentResponder:
    def __init__(
        self,
        cfg: Config,
        router: IntentRouter,
        registry: ActionRegistry,
        ctx: ActionContext,
        *,
        brain: Brain | None = None,
        catman: CatmanClient | None = None,
        journal: Journal | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cfg = cfg
        self.router = router
        self.registry = registry
        self.ctx = ctx
        self.brain = brain
        self.catman = catman
        self.journal = journal
        self.clock = clock
        router.on_shadow = self._on_shadow

    # ---- Responder 接口 ----

    def respond(self, turn: Turn, transcript: Transcript, speak: Callable[[str], bool]) -> Response:
        info = turn.info
        route = self.router.route(transcript.text, shadow_tag=turn.id)
        info["t_intent_done"] = self.clock()
        info["text_normalized"] = route.normalized
        info["tier"] = route.tier
        info["route_flags"] = route.flags
        info["intent_ms"] = round((route.elapsed_rule + route.elapsed_llm) * 1000, 1)
        if route.error:
            info["route_error"] = route.error
        intent = route.intent
        if intent is not None:
            info["intent"] = intent.name
            info["rule_id"] = intent.rule_id
            info["slots"] = intent.slots
            info["confidence"] = intent.confidence
            if intent.tier == "llm":
                info["llm_raw"] = intent.raw
                if intent.name not in (CHAT, DELEGATE):
                    info["candidate_case"] = {
                        "text": transcript.text,
                        "intent": intent.name,
                        "slots": intent.slots,
                        "source": "llm",
                        "turn_id": turn.id,
                    }
        log.info(
            "turn %s intent: %s (%s) %s", turn.id, intent.name if intent else None, route.tier, route.flags
        )
        if turn.cancelled.is_set():
            return Response(info={})
        if intent is None or intent.name == CHAT:
            reply = (intent.slots.get("reply") if intent else None) or ""
            if reply and self.cfg.intent.llm.answer_chat:
                return self._finish(turn, speak, reply, "llm")
            return self._chat(turn, transcript.text, speak)
        if intent.name == DELEGATE:
            return self._delegate(turn, intent, speak)
        return self._act(turn, intent, speak)

    def deliver(self, turn: Turn, speak: Callable[[str], bool]) -> Response:
        payload = turn.payload
        say = (
            getattr(payload, "say", None)
            or (payload.get("say") if isinstance(payload, dict) else None)
            or "時間到喇"
        )
        turn.info["intent"] = "alarm"
        turn.info["tier"] = "timer"
        return self._finish(turn, speak, str(say), "timer", followup=True)

    def cancel(self) -> None:
        if self.brain is not None:
            self.brain.cancel()

    # ---- 各条路 ----

    def _act(self, turn: Turn, intent: Intent, speak: Callable[[str], bool]) -> Response:
        spec = self.router.rules.intents.get(intent.name)
        info = turn.info
        info["action"] = (
            spec.action
            if spec and isinstance(spec.action, str)
            else (f"http:{intent.name}" if spec else None)
        )
        result = self.registry.run(intent, spec, self.ctx)
        info["t_action_done"] = self.clock()
        info["action_ok"] = result.ok
        if result.error:
            info["action_error"] = result.error
        if result.data:
            info["action_data"] = _jsonable(result.data)
        if intent.name in ("repeat", "stop"):
            return self._finish(
                turn, speak, result.say, "rule", followup=result.followup, remember=False, ok=result.ok
            )
        return self._finish(
            turn, speak, result.say, intent.tier, followup=result.followup, ok=result.ok, error=result.error
        )

    def _delegate(self, turn: Turn, intent: Intent, speak: Callable[[str], bool]) -> Response:
        task = str(intent.slots.get("task") or "").strip()
        turn.info["delegated_task"] = task
        if self.catman is None:
            turn.info["action_ok"] = False
            return self._finish(
                turn, speak, NO_CATMAN_PHRASE, "delegate", ok=False, error="catman not configured"
            )
        try:
            self.catman.post(task)
        except CatmanError as e:
            log.warning("delegate failed: %s", e)
            turn.info["action_ok"] = False
            return self._finish(turn, speak, NO_CATMAN_PHRASE, "delegate", ok=False, error=str(e))
        turn.info["action"] = "delegate"
        turn.info["action_ok"] = True
        turn.info["t_action_done"] = self.clock()
        return self._finish(turn, speak, DELEGATED_PHRASE, "delegate")

    def _chat(self, turn: Turn, text: str, speak: Callable[[str], bool]) -> Response:
        if self.brain is None:
            return self._finish(
                turn, speak, NO_BRAIN_PHRASE, "brain_missing", ok=False, error="no brain configured"
            )
        spoken: list[str] = []
        try:
            for sentence in _sentences(self.brain.chat(text, session="voice"), turn):
                if "t_brain_first" not in turn.info:
                    turn.info["t_brain_first"] = self.clock()
                if turn.cancelled.is_set():
                    break
                speak(sentence)
                spoken.append(sentence)
        except Exception as e:  # noqa: BLE001
            log.warning("brain failed: %s", e)
            if not spoken and not turn.cancelled.is_set():
                return self._finish(turn, speak, BRAIN_ERROR_PHRASE, "brain_error", ok=False, error=str(e))
            turn.info["error"] = str(e)
        reply = "".join(spoken)
        turn.info["reply_source"] = "brain"
        if reply:
            self.ctx.state["last_reply"] = reply
        return Response(ok=True, followup=True)

    def _finish(
        self,
        turn: Turn,
        speak: Callable[[str], bool],
        say: str | None,
        source: str,
        *,
        followup: bool = False,
        remember: bool = True,
        ok: bool = True,
        error: str | None = None,
    ) -> Response:
        turn.info["reply_source"] = source
        if say:
            speak(say)
            if remember:
                self.ctx.state["last_reply"] = say
        return Response(ok=ok, error=error, followup=followup)

    def _on_shadow(self, text: str, rule_intent: Intent, llm_intent: Intent | None, tag: Any) -> None:
        agree = llm_intent is not None and llm_intent.name == rule_intent.name
        shadow = {
            "intent": llm_intent.name if llm_intent else None,
            "slots": llm_intent.slots if llm_intent else None,
            "agree": agree,
        }
        log.info(
            "shadow check for %s: rule=%s llm=%s agree=%s", tag, rule_intent.name, shadow["intent"], agree
        )
        if self.journal is not None and tag:
            self.journal.amend(
                str(tag), patch={"shadow": shadow}, flags_add=[] if agree else ["rules_llm_disagree"]
            )


def _sentences(chunks: Iterator[str], turn: Turn) -> Iterator[str]:
    """把流式文本按句切开：遇到句末标点就吐出一句，剩余的最后吐。"""
    buf = ""
    for chunk in chunks:
        if turn.cancelled.is_set():
            return
        buf += chunk
        while True:
            idx = next((i for i, ch in enumerate(buf) if ch in SENTENCE_END), -1)
            if idx < 0:
                break
            sentence, buf = buf[: idx + 1].strip(), buf[idx + 1 :]
            if sentence:
                yield sentence
    tail = buf.strip()
    if tail:
        yield tail


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, dict)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def build_responder(
    cfg: Config, *, speaker=None, journal: Journal | None = None, timers=None, clock=time.monotonic
):
    """按配置组装：规则库 → 路由器（含 LLM 层）→ 动作 → 大脑 / catman。"""
    from catman_io.actions import default_registry
    from catman_io.actions.timers import TimerService
    from catman_io.brain import build_brain
    from catman_io.brain.catman import build_catman
    from catman_io.intent.router import build_router

    router = build_router(cfg)
    brain = build_brain(cfg)
    catman = build_catman(cfg)
    timers = timers or TimerService()
    ctx = ActionContext(cfg=cfg, timers=timers, speaker=speaker, brain=brain, catman=catman, clock=clock)
    responder = IntentResponder(
        cfg, router, default_registry(), ctx, brain=brain, catman=catman, journal=journal, clock=clock
    )
    if catman is not None:
        health = catman.probe()
        log.info("catman at %s: %s", cfg.brain.catman.base_url, "reachable" if health else "unreachable")
    log.info(
        "responder: %d intents, llm=%s, brain=%s, catman=%s",
        len(router.rules.intents),
        "on" if router.llm else "off",
        "on" if brain else "off",
        "on" if catman else "off",
    )
    return responder
