import json
import time

import numpy as np

from catman_io.config import Config
from catman_io.dialog import Turn
from catman_io.journal import Journal, JournalWriter, build_record, compute_flags, is_bad
from catman_io.journal.review import build_review, read_ack, render_markdown, write_ack


def test_journal_write_amend_records_and_audio(tmp_path):
    clock = {"t": 1_700_000_000.0}
    j = Journal(tmp_path, clock=lambda: clock["t"])
    j.write({"turn_id": "a", "text": "x", "flags": []})
    clock["t"] += 5
    j.write({"turn_id": "b", "text": "y", "flags": ["asr_short"]})
    j.amend("a", flags_add=["user_retry"], detail={"next_turn": "b"}, patch={"shadow": {"agree": False}})
    j.amend("zzz", flags_add=["x"])  # 不存在的回合，忽略
    recs = j.records()
    assert [r["turn_id"] for r in recs] == ["a", "b"]
    assert recs[0]["flags"] == ["user_retry"] and recs[0]["shadow"] == {"agree": False}
    assert recs[0]["flag_details"] == {"next_turn": "b"}
    assert j.records(flags={"asr_short"})[0]["turn_id"] == "b"
    assert j.records(since=clock["t"] - 1)[0]["turn_id"] == "b"
    assert j.records(limit=1)[0]["turn_id"] == "b" and j.get("a")["text"] == "x" and j.get("nope") is None
    p = j.save_audio("a", "utterance", np.ones(1600, dtype=np.int16))
    assert p and (tmp_path / p).exists() and p.endswith("a-utterance.wav")
    assert Journal(tmp_path, save_audio=False).save_audio("a", "x", np.ones(10, dtype=np.int16)) is None
    lines = (tmp_path / "2023-11-14.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["kind"] == "turn" and json.loads(lines[2])["kind"] == "amend"


def test_journal_cleanup_by_days(tmp_path):
    j = Journal(tmp_path, keep_days=7)
    (tmp_path / "2000-01-01.jsonl").write_text("{}\n")
    (tmp_path / "audio" / "2000-01-01").mkdir(parents=True)
    today = time.strftime("%Y-%m-%d")
    (tmp_path / f"{today}.jsonl").write_text("")
    assert (
        j.cleanup() == 2
        and not (tmp_path / "2000-01-01.jsonl").exists()
        and (tmp_path / f"{today}.jsonl").exists()
    )


def flags_of(rec, prev=None):
    rec.setdefault("at", 1000.0)
    rec.setdefault("turn_kind", "voice")
    return compute_flags(rec, prev)


def test_compute_flags_scenarios():
    assert flags_of({"status": "no_speech"})[0] == ["wake_no_speech"]
    assert flags_of({"status": "empty"})[0] == ["asr_empty"]
    assert flags_of({"status": "ok", "text": "吓"})[0] == ["asr_short"]
    f, d, _ = flags_of(
        {
            "status": "ok",
            "text": "聲音調去三成",
            "tier": "llm",
            "intent": "volume.set",
            "slots": {"percent": 30},
        }
    )
    assert f == ["rules_miss_llm_hit"] and d["rules_miss_llm_hit"]["intent"] == "volume.set"
    assert flags_of({"status": "ok", "text": "你好", "tier": "llm", "intent": "chat"})[0] == ["llm_chat"]
    assert flags_of(
        {"status": "ok", "text": "x1", "tier": "none", "route_flags": ["llm_timeout", "fallthrough_chat"]}
    )[0] == [
        "llm_timeout",
        "fallthrough_chat",
    ]
    assert flags_of(
        {"status": "ok", "text": "而家幾點", "tier": "rule", "intent": "time.now", "shadow": {"agree": False}}
    )[0] == ["rules_llm_disagree"]
    assert flags_of({"status": "ok", "text": "開燈", "action": "http", "action_ok": False})[0] == [
        "action_failed"
    ]
    assert flags_of({"status": "ok", "text": "你好", "reply_source": "brain_error"})[0] == ["brain_error"]
    assert flags_of(
        {"status": "cancelled", "text": "你好", "t_interrupted": 5.0, "interrupted_after_ms": 800}
    )[0] == ["barge_in_early"]
    assert flags_of(
        {"status": "ok", "text": "幾點", "tier": "rule", "intent": "time.now", "lat_first_audio_ms": 4000}
    )[0] == ["slow"]
    assert (
        flags_of(
            {"status": "ok", "text": "幾點", "tier": "rule", "intent": "time.now", "lat_first_audio_ms": 400}
        )[0]
        == []
    )
    assert flags_of({"status": "timeout", "text": "x1"})[0] == ["turn_timeout"]


def test_compute_flags_retry_and_negation():
    prev = {"turn_id": "p", "at": 1000.0, "turn_kind": "voice", "text": "幫我開廳燈"}
    f, d, pf = compute_flags(
        {"turn_id": "n", "at": 1010.0, "turn_kind": "voice", "status": "ok", "text": "幫我開廳嘅燈"}, prev
    )
    assert pf == ["user_retry"] and f == ["retry_of"] and d["retry_of"]["turn_id"] == "p"
    f, d, pf = compute_flags(
        {"turn_id": "n", "at": 1010.0, "turn_kind": "voice", "status": "ok", "text": "唔係呀"}, prev
    )
    assert pf == ["user_negation"] and f == ["negation_of"]
    _, _, pf = compute_flags(
        {"turn_id": "n", "at": 1100.0, "turn_kind": "voice", "status": "ok", "text": "幫我開廳嘅燈"}, prev
    )
    assert pf == []
    _, _, pf = compute_flags(
        {"turn_id": "n", "at": 1010.0, "turn_kind": "voice", "status": "ok", "text": "今日天氣點"}, prev
    )
    assert pf == []


def test_journal_writer_builds_records_and_amends_previous(tmp_path):
    cfg = Config.load(None)
    j = Journal(tmp_path)
    w = JournalWriter(j, cfg)
    t1 = Turn(
        id="t1", gen=1, t_wake=10.0, wake_model="m", wake_score=0.9, t_speech_start=10.5, t_speech_end=12.0
    )
    t1.audio = np.ones(1600, dtype=np.int16)
    t1.wake_audio = np.ones(800, dtype=np.int16)
    t1.t_reply_started, t1.t_reply_done = 12.9, 14.0
    t1.info.update(
        {
            "asr_text": "帮我开厅灯",
            "t_asr_done": 12.3,
            "intent": "light.on",
            "tier": "rule",
            "rule_id": "light.on#0",
            "slots": {"room": "客廳"},
            "t_intent_done": 12.31,
            "action": "http",
            "action_ok": True,
            "t_action_done": 12.8,
            "reply_text": "開咗燈",
            "reply_source": "rule",
            "t_first_audio": 12.9,
            "route_flags": [],
        }
    )
    rec = w(t1, "ok")
    assert rec["text"] == "幫我開廳燈" and rec["asr_raw"] == "帮我开厅灯"
    assert rec["lat_asr_ms"] == 300.0 and rec["lat_first_audio_ms"] == 900.0 and rec["lat_total_ms"] == 2000.0
    assert rec["speech_ms"] == 1500.0 and rec["audio_utterance"] and rec["audio_wake"] and rec["flags"] == []
    t2 = Turn(id="t2", gen=2, t_speech_end=20.0)
    t2.info.update(
        {
            "asr_text": "幫我開廳嘅燈",
            "tier": "none",
            "route_flags": ["fallthrough_chat"],
            "reply_source": "brain_missing",
        }
    )
    rec2 = w(t2, "ok")
    assert set(rec2["flags"]) == {"fallthrough_chat", "brain_missing", "retry_of"}
    recs = j.records()
    assert recs[0]["flags"] == ["user_retry"] and recs[0]["flag_details"]["next_turn"] == "t2"
    assert is_bad(recs[0]) and is_bad(recs[1]) and w.count == 2
    rec3 = build_record(Turn(id="a1", gen=3, kind="alarm"), "ok")
    assert rec3["turn_kind"] == "alarm" and "text" not in rec3


def test_review_groups_and_ack(tmp_path):
    j = Journal(tmp_path)
    j.write(
        {
            "turn_id": "a",
            "at": 100.0,
            "status": "ok",
            "text": "聲音調去三成",
            "intent": "volume.set",
            "tier": "llm",
            "slots": {"percent": 30},
            "flags": ["rules_miss_llm_hit"],
            "candidate_case": {
                "text": "聲音調去三成",
                "intent": "volume.set",
                "slots": {"percent": 30},
                "source": "llm",
            },
        }
    )
    j.write(
        {
            "turn_id": "b",
            "at": 200.0,
            "status": "no_speech",
            "flags": ["wake_no_speech"],
            "audio_wake": "audio/x.wav",
        }
    )
    j.write(
        {
            "turn_id": "c",
            "at": 300.0,
            "status": "ok",
            "text": "而家幾點",
            "intent": "time.now",
            "tier": "rule",
            "flags": [],
        }
    )
    review = build_review(j, intents={"time.now": "報時"})
    assert review["count"] == 2 and set(review["groups"]) == {"rules_miss_llm_hit", "wake_no_speech"}
    assert review["candidate_cases"][0]["intent"] == "volume.set" and review["until"] == 200.0
    md = render_markdown(review)
    assert (
        "rules_miss_llm_hit" in md and "聲音調去三成" in md and "```jsonl" in md and "`time.now`：報時" in md
    )
    assert read_ack(j) is None
    write_ack(j, 200.0)
    assert read_ack(j) == 200.0
    assert build_review(j)["count"] == 0 and build_review(j, include_acked=True)["count"] == 2
