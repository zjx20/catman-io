"""三层意图的调度：规则 → LLM → 落空（交给大脑）。规则命中后可抽样让 LLM 做影子对照。"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from catman_io.llm import LLMError

from . import Intent
from .normalize import normalize
from .rules import RuleSet, RuleStore

if TYPE_CHECKING:
    from catman_io.config import Config

    from .llm import LLMIntentRecognizer

log = logging.getLogger(__name__)


@dataclass
class RouteResult:
    text: str
    normalized: str
    intent: Intent | None
    tier: str  # rule | llm | none
    elapsed_rule: float = 0.0
    elapsed_llm: float = 0.0
    flags: list[str] = field(default_factory=list)
    error: str | None = None


ShadowCallback = Callable[[str, Intent, Intent | None], None]


class IntentRouter:
    def __init__(
        self,
        store: RuleStore,
        llm: LLMIntentRecognizer | None = None,
        *,
        shadow_rate: float = 0.0,
        rng: random.Random | None = None,
        on_shadow: ShadowCallback | None = None,
    ):
        self.store = store
        self.llm = llm
        self.shadow_rate = shadow_rate
        self.rng = rng or random.Random()
        self.on_shadow = on_shadow

    @property
    def rules(self) -> RuleSet:
        return self.store.current

    def route(self, text: str, *, use_rules: bool = True, use_llm: bool = True) -> RouteResult:
        self.store.reload_if_changed()
        normalized = normalize(text)
        result = RouteResult(text, normalized, None, "none")
        if not normalized:
            result.flags.append("asr_empty")
            return result
        if use_rules:
            t0 = time.perf_counter()
            hit = self.store.current.match(normalized, normalized=True)
            result.elapsed_rule = time.perf_counter() - t0
            if hit is not None:
                result.intent, result.tier = hit, "rule"
                if self.llm is not None and self.shadow_rate > 0 and self.rng.random() < self.shadow_rate:
                    self._shadow(text, hit)
                return result
        if use_llm and self.llm is not None:
            t0 = time.perf_counter()
            try:
                got = self.llm.recognize(text)
                result.elapsed_llm = time.perf_counter() - t0
                if got is not None:
                    result.intent, result.tier = got, "llm"
                    if got.name not in ("chat", "delegate"):
                        result.flags.append("rules_miss_llm_hit")
                    return result
                result.flags.append("llm_no_intent")
            except LLMError as e:
                result.elapsed_llm = time.perf_counter() - t0
                result.error = str(e)
                result.flags.append("llm_timeout" if e.kind == "timeout" else "llm_error")
                log.warning("llm intent failed (%s): %s", e.kind, e)
            except Exception as e:  # noqa: BLE001
                result.elapsed_llm = time.perf_counter() - t0
                result.error = str(e)
                result.flags.append("llm_error")
                log.exception("llm intent crashed")
        result.flags.append("fallthrough_chat")
        return result

    def _shadow(self, text: str, rule_intent: Intent) -> None:
        def run() -> None:
            try:
                got = self.llm.recognize(text) if self.llm is not None else None
            except Exception as e:  # noqa: BLE001
                log.warning("shadow llm failed: %s", e)
                return
            if self.on_shadow is not None:
                self.on_shadow(text, rule_intent, got)

        threading.Thread(target=run, name="intent-shadow", daemon=True).start()


def build_router(
    cfg: Config, store: RuleStore | None = None, *, on_shadow: ShadowCallback | None = None
) -> IntentRouter:
    """按配置组装：规则总是有；intent.llm.enabled 才建 LLM 层（缺 API key 立刻报错）。"""
    from catman_io.config import secret

    from . import rule_store

    store = store or rule_store(cfg)
    llm = None
    c = cfg.intent.llm
    if c.enabled:
        from catman_io.llm import OpenAICompatClient

        from .llm import LLMIntentRecognizer

        if not c.model:
            raise ValueError("intent.llm.model is empty")
        client = OpenAICompatClient(
            c.base_url, api_key=secret(c.api_key_env, what="intent.llm"), model=c.model, timeout=c.timeout
        )
        llm = LLMIntentRecognizer(
            client,
            lambda: store.current,
            context=cfg.intent.context,
            tool_choice=c.tool_choice,
            timeout=c.timeout,
            max_tokens=c.max_tokens,
        )
    return IntentRouter(store, llm, shadow_rate=c.shadow_rate if llm else 0.0, on_shadow=on_shadow)
