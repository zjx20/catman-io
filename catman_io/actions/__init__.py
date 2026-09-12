"""动作层：意图 → 做事 → 一句要念的话。

内置动作（``builtin.py``）覆盖報時 / 計時 / 音量 / 停 / 天氣 等；通用 ``http`` 动作（``http.py``）
让新技能只改 YAML 就能加。动作拿到的是 :class:`ActionContext`（时钟、扬声器、定时器、大脑、catman…），
返回 :class:`ActionResult`。
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from catman_io.intent import Intent

if TYPE_CHECKING:
    from catman_io.brain import Brain
    from catman_io.brain.catman import CatmanClient
    from catman_io.config import Config
    from catman_io.intent.rules import IntentSpec
    from catman_io.tts.speaker import Speaker

    from .timers import TimerService

log = logging.getLogger(__name__)


@dataclass
class ActionResult:
    say: str | None = None  # 要念的话；None = 不出声
    followup: bool = False  # 期待用户接着说（更长的跟进窗口）
    # 追问的槽位名（例如 "duration"）：缺了它做不了事，say 是反问；用户接着说的话先按这个槽位解析，
    # 解析得到就带着它再执行本意图（责任在 responder），解析不到才当一句新话路由
    ask: str | None = None
    ok: bool = True
    error: str | None = None
    data: dict[str, Any] = field(default_factory=dict)  # 写进日志


HttpFn = Callable[[str, str, dict[str, str], bytes | None, float], tuple[int, bytes]]


@dataclass
class ActionContext:
    cfg: Config
    timers: TimerService
    speaker: Speaker | None = None
    brain: Brain | None = None
    catman: CatmanClient | None = None
    http: HttpFn | None = None
    now: Callable[[], dt.datetime] = dt.datetime.now
    clock: Callable[[], float] = time.monotonic
    state: dict[str, Any] = field(default_factory=dict)  # last_reply 之类跨回合的小状态


ActionFn = Callable[[Intent, ActionContext], ActionResult]


class ActionRegistry:
    def __init__(self) -> None:
        self._fns: dict[str, ActionFn] = {}

    def register(self, name: str, fn: ActionFn) -> None:
        self._fns[name] = fn

    def names(self) -> list[str]:
        return sorted(self._fns)

    def run(self, intent: Intent, spec: IntentSpec | None, ctx: ActionContext) -> ActionResult:
        action = spec.action if spec is not None else None
        say_template = spec.say if spec is not None else None
        if action is None:
            return ActionResult(ok=False, error=f"intent {intent.name} has no action")
        if isinstance(action, str):
            fn = self._fns.get(action)
            if fn is None:
                return ActionResult(ok=False, error=f"unknown action {action!r}")
            result = fn(intent, ctx)
        elif isinstance(action, dict) and action.get("type") == "http":
            from .http import http_action

            result = http_action(action, intent, ctx)
        else:
            return ActionResult(ok=False, error=f"bad action spec for {intent.name}: {action!r}")
        if say_template and result.ok:
            from .http import render

            result.say = render(say_template, {**intent.slots, "text": intent.raw or "", "r": result.data})
        if spec is not None and spec.followup:
            result.followup = True
        return result


def default_registry() -> ActionRegistry:
    from . import builtin

    reg = ActionRegistry()
    builtin.register_all(reg)
    return reg


__all__ = ["ActionContext", "ActionFn", "ActionRegistry", "ActionResult", "default_registry"]
