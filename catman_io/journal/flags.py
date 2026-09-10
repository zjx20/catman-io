"""自动标记：从一条回合记录（和上一回合）判断它是不是 bad case，以及为什么。

标记只是给人 / 给 catman 看的线索，不影响运行。``BAD_FLAGS`` 是值得复盘的那些；
``user_retry`` / ``user_negation`` 是从下一句反推上一句错了，所以要补记到上一回合。
"""

from __future__ import annotations

import difflib
from typing import Any

from catman_io.intent.normalize import normalize

NEGATION_PREFIXES = (
    "唔係",
    "唔系",
    "錯咗",
    "錯左",
    "取消",
    "唔啱",
    "唔要",
    "唔係咁",
    "唔係呢個",
    "唔係呢",
    "咪",
)
RETRY_WINDOW_SECONDS = 30.0
RETRY_SIMILARITY = 0.6
BARGE_IN_EARLY_MS = 2000

BAD_FLAGS = {
    "wake_no_speech",
    "asr_empty",
    "asr_short",
    "rules_miss_llm_hit",
    "rules_llm_disagree",
    "fallthrough_chat",
    "llm_chat",
    "llm_timeout",
    "llm_error",
    "llm_no_intent",
    "brain_error",
    "brain_missing",
    "action_failed",
    "user_retry",
    "user_negation",
    "barge_in_early",
    "slow",
    "turn_timeout",
    "worker_error",
}
FLAG_HELP = {
    "wake_no_speech": "唤醒后没人说话：多半是误唤醒，音频可拿去做唤醒词的反例",
    "asr_empty": "识别结果为空",
    "asr_short": "识别结果只有一个字，多半没听清",
    "rules_miss_llm_hit": "规则没中、LLM 判出了已知意图：规则缺覆盖，可把这句加进规则与用例",
    "rules_llm_disagree": "规则命中但影子 LLM 不同意：规则可能误命中",
    "fallthrough_chat": "规则与 LLM 都没判出意图，进了大脑（或没配 LLM）",
    "llm_chat": "LLM 判为闲聊：看看是不是漏了应该有的指令",
    "llm_timeout": "LLM 超时",
    "llm_error": "LLM 出错",
    "llm_no_intent": "LLM 没给出任何意图",
    "brain_error": "大脑（对话模型）出错",
    "brain_missing": "没配大脑，答不了",
    "action_failed": "动作执行失败",
    "user_retry": "用户 30 秒内又说了一句很像的话：上一次多半没做对",
    "user_negation": "用户下一句是否定（唔係 / 取消…）：上一次多半做错",
    "barge_in_early": "回复开始 2 秒内被打断：回复多半不对或太长",
    "slow": "说完到开口太慢",
    "turn_timeout": "思考超时",
    "worker_error": "处理线程出错",
}


def similarity(a: str, b: str) -> float:
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def compute_flags(
    rec: dict[str, Any], prev: dict[str, Any] | None, *, slow_threshold_ms: float = 3000.0
) -> tuple[list[str], dict[str, Any], list[str]]:
    """返回 (本回合标记, 标记详情, 要补记到上一回合的标记)。"""
    flags: list[str] = []
    detail: dict[str, Any] = {}
    prev_flags: list[str] = []
    status = rec.get("status")
    kind = rec.get("turn_kind", "voice")
    text = rec.get("text") or ""
    tier = rec.get("tier")
    intent = rec.get("intent")

    def add(f: str, **d: Any) -> None:
        if f not in flags:
            flags.append(f)
        if d:
            detail[f] = d

    if kind == "voice":
        if status == "no_speech":
            add("wake_no_speech")
        elif status == "empty" or (status in ("ok", "error") and not text and rec.get("audio_utterance")):
            add("asr_empty")
        elif text and len(normalize(text)) <= 1:
            add("asr_short", text=text)
    if status == "timeout":
        add("turn_timeout")
    if status == "error" and rec.get("error") == "worker error":
        add("worker_error")
    for f in rec.get("route_flags") or []:
        if f in BAD_FLAGS and f not in ("asr_empty",):
            add(f)
    if tier == "llm" and intent and intent not in ("chat", "delegate"):
        add("rules_miss_llm_hit", intent=intent, slots=rec.get("slots"))
    if tier == "llm" and intent == "chat":
        add("llm_chat")
    shadow = rec.get("shadow")
    if shadow and shadow.get("agree") is False:
        add("rules_llm_disagree", rule=intent, llm=shadow.get("intent"))
    if rec.get("action") and rec.get("action_ok") is False:
        add("action_failed", error=rec.get("action_error"))
    src = rec.get("reply_source")
    if src == "brain_error":
        add("brain_error", error=rec.get("error"))
    if src == "brain_missing":
        add("brain_missing")
    ms = rec.get("interrupted_after_ms")
    if rec.get("t_interrupted") is not None and ms is not None and ms < BARGE_IN_EARLY_MS:
        add("barge_in_early", after_ms=ms)
    lat = rec.get("lat_first_audio_ms")
    if tier in ("rule", "llm") and lat is not None and lat > slow_threshold_ms:
        add("slow", first_audio_ms=lat)
    if prev and kind == "voice" and text and prev.get("turn_kind", "voice") == "voice" and prev.get("text"):
        gap = (rec.get("at") or 0) - (prev.get("at") or 0)
        if 0 <= gap <= RETRY_WINDOW_SECONDS:
            n = normalize(text)
            if any(n.startswith(p) for p in NEGATION_PREFIXES):
                prev_flags.append("user_negation")
                add("negation_of", turn_id=prev.get("turn_id"))
            else:
                sim = similarity(text, prev["text"])
                if sim >= RETRY_SIMILARITY:
                    prev_flags.append("user_retry")
                    add("retry_of", turn_id=prev.get("turn_id"), similarity=round(sim, 2))
    return flags, detail, prev_flags


def is_bad(rec: dict[str, Any]) -> bool:
    return bool(BAD_FLAGS & set(rec.get("flags") or []))
