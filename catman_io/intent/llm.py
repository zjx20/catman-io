"""意图第 1 层：flash 级模型做一次 function calling。工具集由规则文件生成，外加 chat 与 delegate。

- 模型调了某个意图工具 → Intent(tier="llm")，槽位值用和规则层一样的解析器归一化。
- 调了 chat（闲聊 / 问答）→ Intent("chat")，slots 里可能带一句可直接念的 reply。
- 调了 delegate（要动手做事）→ Intent("delegate", {"task": ...})。
- 没调工具、直接回文本 → 当 chat，文本放进 reply。
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from collections.abc import Callable
from typing import Any

from catman_io.config import IntentContextConfig
from catman_io.llm import LLMError, OpenAICompatClient

from . import Intent
from .rules import RuleSet
from .slots import parse_date, parse_duration, parse_number, parse_percent, parse_time

log = logging.getLogger(__name__)

CHAT = "chat"
DELEGATE = "delegate"
WEEKDAYS = "一二三四五六日"

CHAT_TOOL = {
    "type": "function",
    "function": {
        "name": CHAT,
        "description": (
            "唔係指令，係閒聊、問知識、問意見之類。如果一兩句粵語就答到，直接寫喺 reply。"
        ),
        "parameters": {
            "type": "object",
            "properties": {"reply": {"type": "string", "description": "可以直接讀出嘅簡短粵語回答（可選）"}},
            "required": [],
        },
    },
}
DELEGATE_TOOL = {
    "type": "function",
    "function": {
        "name": DELEGATE,
        "description": (
            "要人動手做嘅任務：查系統、跑命令、幫手辦事、要用工具或者要花時間嘅嘢，交俾後台助手 catman。"
        ),
        "parameters": {
            "type": "object",
            "properties": {"task": {"type": "string", "description": "用一句話講清楚要做乜"}},
            "required": ["task"],
        },
    },
}

SYSTEM_PROMPT = """你係一個粵語語音助手嘅意圖分類器。用戶講咗一句話（語音識別結果，可能有錯字），你要揀一個最合適嘅工具：
- 係明確指令就揀對應嘅工具，並填好參數；參數可以照抄原話（例如「十分鐘」），系統會自己換算。
- 唔係指令（閒聊、問嘢、講笑）就用 chat；如果一兩句粵語就答到，直接寫喺 reply。
- 要人動手做、查系統、跑命令、或者要花時間嘅任務，用 delegate。
只揀一個工具，唔好解釋。
而家係 {now}。{context}"""


def _context_text(ctx: IntentContextConfig) -> str:
    parts = []
    if ctx.home:
        parts.append(ctx.home)
    if ctx.rooms:
        parts.append("屋企有呢啲房間：" + "、".join(r.split("|")[0] for r in ctx.rooms))
    if ctx.extra:
        parts.append(ctx.extra)
    return " ".join(parts)


def _now_text(now: dt.datetime) -> str:
    return f"{now.year}年{now.month}月{now.day}日 星期{WEEKDAYS[now.weekday()]} {now.strftime('%H:%M')}"


class LLMIntentRecognizer:
    def __init__(
        self,
        client: OpenAICompatClient,
        rules: Callable[[], RuleSet],
        *,
        context: IntentContextConfig | None = None,
        tool_choice: str = "auto",
        timeout: float = 4.0,
        max_tokens: int = 256,
        now: Callable[[], dt.datetime] = dt.datetime.now,
    ):
        self.client = client
        self.rules = rules
        self.context = context or IntentContextConfig()
        self.tool_choice = tool_choice
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.now = now
        self.last_elapsed = 0.0

    def messages(self, text: str) -> list[dict[str, Any]]:
        system = SYSTEM_PROMPT.format(now=_now_text(self.now()), context=_context_text(self.context))
        return [{"role": "system", "content": system}, {"role": "user", "content": text}]

    def recognize(self, text: str) -> Intent | None:
        rs = self.rules()
        tools = rs.tools() + [CHAT_TOOL, DELEGATE_TOOL]
        t0 = time.perf_counter()
        result = self.client.chat(
            self.messages(text),
            tools=tools,
            tool_choice=self.tool_choice if self.tool_choice != "auto" else "auto",
            temperature=0.0,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
        self.last_elapsed = time.perf_counter() - t0
        raw = str(result.raw.get("choices", [{}])[0].get("message", ""))[:500] if result.raw else None
        if not result.tool_calls:
            content = (result.content or "").strip()
            if not content:
                return None
            return Intent(CHAT, {"reply": content}, 0.6, "llm", None, raw)
        call = result.tool_calls[0]
        if call.name == CHAT:
            slots = {}
            reply = str(call.arguments.get("reply") or "").strip()
            if reply:
                slots["reply"] = reply
            return Intent(CHAT, slots, 0.8, "llm", None, raw)
        if call.name == DELEGATE:
            task = str(call.arguments.get("task") or text).strip()
            return Intent(DELEGATE, {"task": task}, 0.8, "llm", None, raw)
        name = rs.intent_for_tool(call.name)
        if name is None:
            log.warning("LLM picked unknown tool %r", call.name)
            return None
        spec = rs.intents[name]
        slots = normalize_slots(
            spec_types={s: spec.slot_type(s) for s in spec.slots}, values=call.arguments, rs=rs
        )
        return Intent(name, slots, 0.8, "llm", None, raw)


def normalize_slots(*, spec_types: dict[str, str], values: dict[str, Any], rs: RuleSet) -> dict[str, Any]:
    """模型给的参数可能是原话、数字或标准格式，统一成规则层同样的规范值；转不了的丢掉。"""
    out: dict[str, Any] = {}
    for slot, type_name in spec_types.items():
        if slot not in values or values[slot] in (None, ""):
            continue
        v = values[slot]
        got = _coerce(type_name, v, rs)
        if got is not None:
            out[slot] = got
    return out


def _coerce(type_name: str, v: Any, rs: RuleSet) -> Any:
    s = str(v).strip()
    if type_name == "duration":
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
        if s.isdigit():
            return int(s)
        return parse_duration(s)
    if type_name == "time":
        if isinstance(v, dict) and "hour" in v:
            return {"hour": int(v["hour"]), "minute": int(v.get("minute", 0)), "period": v.get("period")}
        parts = s.split(":")
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            h, m = int(parts[0]), int(parts[1])
            if 0 <= h <= 23 and 0 <= m <= 59:
                return {"hour": h, "minute": m, "period": "am" if h < 12 else "pm"}
        return parse_time(s)
    if type_name == "date":
        try:
            return dt.date.fromisoformat(s).isoformat()
        except ValueError:
            return parse_date(s)
    if type_name == "percent":
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(max(0, min(100, v)))
        if s.isdigit():
            return int(max(0, min(100, int(s))))
        return parse_percent(s) or parse_percent(s + "%")
    if type_name == "number":
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return v
        return parse_number(s)
    if type_name == "room":
        st = rs.types.get("room")
        return (st.parse(s) if st else None) or s
    return s


__all__ = ["CHAT", "DELEGATE", "LLMError", "LLMIntentRecognizer", "normalize_slots"]
