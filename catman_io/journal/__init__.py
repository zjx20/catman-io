"""回合日志与 bad case 自动收集。

- :class:`Journal`：JSONL 追加写 + amend 合并读 + 音频落盘 + 按天清理。
- :func:`compute_flags`：一条记录 + 上一条 → 标记（``flags.py``）。
- :class:`JournalWriter`：pipeline 的 ``on_turn`` 回调，把 Turn 变成记录、存音频、打标记、补记上一回合。
- ``review.py``：按标记分组的复盘包（给 catman 的 API 与 flywheel 命令用）。
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from catman_io.intent.normalize import to_hk

from .flags import BAD_FLAGS, FLAG_HELP, compute_flags, is_bad
from .journal import Journal

if TYPE_CHECKING:
    from catman_io.config import Config
    from catman_io.dialog import Turn

log = logging.getLogger(__name__)


def build_record(turn: Turn, status: str, *, asr_backend: str = "") -> dict[str, Any]:
    info = turn.info
    rec: dict[str, Any] = {
        "turn_id": turn.id,
        "at": time.time(),
        "gen": turn.gen,
        "turn_kind": turn.kind,  # voice | alarm（"kind" 是日志行类型 turn / amend，不能撞名）
        "followup": turn.followup,
        "reprompted": turn.reprompted,
        "barge_in": turn.barge_in,
        "status": status,
        "error": info.get("error"),
        "wake": {"model": turn.wake_model, "score": turn.wake_score} if turn.wake_model else None,
        "forced_end": turn.forced_end,
        "speech_ms": _ms(turn.t_speech_start, turn.t_speech_end),
        "asr_backend": asr_backend,
        "asr_raw": info.get("asr_text"),
        "text": to_hk(info["asr_text"]) if info.get("asr_text") else None,
        "asr_lang": info.get("asr_lang"),
        "intent": info.get("intent"),
        "tier": info.get("tier"),
        "rule_id": info.get("rule_id"),
        "slots": info.get("slots"),
        "confidence": info.get("confidence"),
        "route_flags": info.get("route_flags"),
        "route_error": info.get("route_error"),
        "llm_raw": info.get("llm_raw"),
        "shadow": info.get("shadow"),
        "action": info.get("action"),
        "action_ok": info.get("action_ok"),
        "action_error": info.get("action_error"),
        "reply_text": info.get("reply_text"),
        "reply_source": info.get("reply_source"),
        "candidate_case": info.get("candidate_case"),
        "t_wake": turn.t_wake,
        "t_speech_start": turn.t_speech_start,
        "t_speech_end": turn.t_speech_end,
        "t_asr_done": info.get("t_asr_done"),
        "t_intent_done": info.get("t_intent_done"),
        "t_action_done": info.get("t_action_done"),
        "t_brain_first": info.get("t_brain_first"),
        "t_first_audio": info.get("t_first_audio"),
        "t_reply_done": turn.t_reply_done,
        "t_interrupted": turn.t_interrupted,
        "interrupted_after_ms": _ms(turn.t_reply_started, turn.t_interrupted),
        "lat_asr_ms": _ms(turn.t_speech_end, info.get("t_asr_done")),
        "lat_intent_ms": _ms(info.get("t_asr_done"), info.get("t_intent_done")),
        "lat_action_ms": _ms(info.get("t_intent_done"), info.get("t_action_done")),
        "lat_brain_first_ms": _ms(info.get("t_intent_done"), info.get("t_brain_first")),
        "lat_first_audio_ms": _ms(turn.t_speech_end, info.get("t_first_audio")),
        "lat_total_ms": _ms(turn.t_speech_end, turn.t_reply_done),
        "asr_seconds": info.get("asr_seconds"),
        "tts_seconds": info.get("tts_seconds"),
    }
    return {k: v for k, v in rec.items() if v is not None}


def _ms(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round((b - a) * 1000.0, 1)


class JournalWriter:
    """pipeline 的 on_turn：存音频、建记录、打标记、补记上一回合。"""

    def __init__(self, journal: Journal, cfg: Config):
        self.journal = journal
        self.cfg = cfg
        self.prev: dict[str, Any] | None = None
        self.count = 0

    def __call__(self, turn: Turn, status: str) -> dict[str, Any]:
        rec = build_record(
            turn, status, asr_backend=f"{self.cfg.asr.backend}:{self.cfg.asr.model or 'default'}"
        )
        try:
            if turn.wake_audio is not None and turn.kind == "voice":
                p = self.journal.save_audio(turn.id, "wake", turn.wake_audio)
                if p:
                    rec["audio_wake"] = p
            if turn.audio is not None:
                p = self.journal.save_audio(turn.id, "utterance", turn.audio)
                if p:
                    rec["audio_utterance"] = p
        except Exception:  # noqa: BLE001
            log.exception("saving audio failed for %s", turn.id)
        flags, detail, prev_flags = compute_flags(
            rec, self.prev, slow_threshold_ms=self.cfg.dialog.slow_threshold * 1000.0
        )
        rec["flags"] = flags
        if detail:
            rec["flag_details"] = detail
        self.journal.write(rec)
        if prev_flags and self.prev is not None:
            self.journal.amend(self.prev["turn_id"], flags_add=prev_flags, detail={"next_turn": turn.id})
        if turn.kind == "voice":
            self.prev = rec
        self.count += 1
        if flags:
            log.info("turn %s flags: %s", turn.id, ",".join(flags))
        return rec


def build_journal(cfg: Config) -> Journal | None:
    if not cfg.journal.enabled:
        return None
    j = Journal(cfg.journal_dir, save_audio=cfg.journal.save_audio, keep_days=cfg.journal.keep_days)
    try:
        removed = j.cleanup()
        if removed:
            log.info("journal cleanup removed %d old file(s)", removed)
    except Exception:  # noqa: BLE001
        log.exception("journal cleanup failed")
    return j


__all__ = [
    "BAD_FLAGS",
    "FLAG_HELP",
    "Journal",
    "JournalWriter",
    "build_journal",
    "build_record",
    "compute_flags",
    "is_bad",
]
