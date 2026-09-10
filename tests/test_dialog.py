"""对话状态机：逐帧喂假数据，检查状态与命令。"""

import numpy as np

from catman_io.config import DialogConfig, VadConfig
from catman_io.dialog import (
    Beep,
    Cancel,
    Deliver,
    Dialog,
    ReplyDone,
    ReplyStarted,
    StartTurn,
    State,
    StopSpeaking,
    TimerFired,
    Transcribe,
    TranscriptEmpty,
    TurnEnded,
    TurnFailed,
)

FRAME = np.zeros(1280, dtype=np.int16)


class Det:
    model = "siu_maau_jan"
    score = 0.9


class Clock:
    def __init__(self):
        self.t = 100.0

    def frame(self, d: Dialog, prob=0.0, wake=False):
        self.t += 0.08
        return d.on_frame(self.t, FRAME, [Det()] if wake else [], prob)

    def frames(self, d: Dialog, probs, wake=False):
        out = []
        for p in probs:
            out += self.frame(d, p, wake)
        return out


def make(**kw):
    vad = VadConfig(start_frames=1, trailing_silence_ms=240, no_speech_timeout_ms=800, followup_guard_ms=0)
    return Dialog(DialogConfig(**kw), vad), Clock()


def kinds(cmds):
    return [type(c).__name__ for c in cmds]


def test_wake_listen_think_speak_followup_idle():
    d, clk = make(followup_seconds=1.0)
    cmds = clk.frame(d, wake=True)
    assert kinds(cmds) == ["Beep", "StartTurn"] and cmds[0].kind == "wake"
    turn = cmds[1].turn
    assert d.state == State.LISTENING and turn.t_wake == clk.t
    cmds = clk.frames(d, [0.9] * 6 + [0.0] * 3)
    assert kinds(cmds) == ["Transcribe"] and cmds[0].turn is turn
    assert d.state == State.THINKING and turn.audio is not None and turn.t_speech_end == clk.t
    assert d.on_event(clk.t, ReplyStarted(turn.id)) == []
    assert d.state == State.SPEAKING
    cmds = d.on_event(clk.t, ReplyDone(turn.id))
    assert kinds(cmds) == ["TurnEnded"] and cmds[0].status == "ok" and turn.status == "ok"
    assert d.state == State.FOLLOWUP and d.turn is None
    # 跟进窗口内没人说话 → IDLE
    assert clk.frames(d, [0.0] * 14) == []
    assert d.state == State.IDLE


def test_no_speech_after_wake_beeps_and_ends_turn():
    d, clk = make()
    turn = clk.frame(d, wake=True)[1].turn
    cmds = clk.frames(d, [0.0] * 12)
    assert kinds(cmds) == ["Beep", "TurnEnded"]
    assert cmds[0].kind == "timeout" and cmds[1].status == "no_speech" and turn.status == "no_speech"
    assert d.state == State.IDLE


def test_barge_in_while_speaking_cancels_and_starts_new_turn():
    d, clk = make()
    t1 = clk.frame(d, wake=True)[1].turn
    clk.frames(d, [0.9] * 4 + [0.0] * 3)
    d.on_event(clk.t, ReplyStarted(t1.id))
    cmds = clk.frame(d, wake=True)
    assert kinds(cmds) == ["StopSpeaking", "Cancel", "TurnEnded", "Beep", "StartTurn"]
    assert isinstance(cmds[0], StopSpeaking) and cmds[1].turn is t1 and cmds[2].status == "cancelled"
    t2 = cmds[4].turn
    assert t2.barge_in and t2.gen == t1.gen + 1 and t1.t_interrupted == clk.t
    assert d.state == State.LISTENING and d.turn is t2
    # 过期回合的事件被忽略
    assert d.on_event(clk.t, ReplyDone(t1.id)) == []


def test_wake_while_thinking_cancels():
    d, clk = make()
    t1 = clk.frame(d, wake=True)[1].turn
    clk.frames(d, [0.9] * 4 + [0.0] * 3)
    cmds = clk.frame(d, wake=True)
    assert kinds(cmds) == ["Cancel", "TurnEnded", "Beep", "StartTurn"]
    assert cmds[0].turn is t1 and not cmds[3].turn.barge_in


def test_wake_again_while_listening_restarts_listening():
    d, clk = make()
    t1 = clk.frame(d, wake=True)[1].turn
    clk.frames(d, [0.0] * 3)
    cmds = clk.frame(d, wake=True)
    assert kinds(cmds) == ["Beep"] and d.turn is t1 and d.state == State.LISTENING
    # 超时从重新开始算
    assert clk.frames(d, [0.0] * 9) == []
    assert kinds(clk.frames(d, [0.0] * 3)) == ["Beep", "TurnEnded"]


def test_empty_transcript_reprompts_once():
    d, clk = make(reprompt_on_empty=True)
    turn = clk.frame(d, wake=True)[1].turn
    clk.frames(d, [0.9] * 4 + [0.0] * 3)
    cmds = d.on_event(clk.t, TranscriptEmpty(turn.id))
    assert kinds(cmds) == ["Beep"] and cmds[0].kind == "reprompt"
    assert d.state == State.LISTENING and turn.reprompted
    cmds = clk.frames(d, [0.9] * 4 + [0.0] * 3)
    assert kinds(cmds) == ["Transcribe"]
    cmds = d.on_event(clk.t, TranscriptEmpty(turn.id))
    assert kinds(cmds) == ["TurnEnded"] and cmds[0].status == "empty" and d.state == State.IDLE


def test_thinking_beep_and_turn_timeout():
    d, clk = make(thinking_beep_after=0.5, turn_timeout=2.0)
    turn = clk.frame(d, wake=True)[1].turn
    clk.frames(d, [0.9] * 4 + [0.0] * 3)
    cmds = clk.frames(d, [0.0] * 7)
    assert [c for c in cmds if isinstance(c, Beep)][0].kind == "thinking"
    assert len([c for c in cmds if isinstance(c, Beep)]) == 1
    cmds = clk.frames(d, [0.0] * 20)
    assert kinds(cmds) == ["Cancel", "TurnEnded"] and cmds[1].status == "timeout"
    assert d.state == State.IDLE and turn.status == "timeout"


def test_turn_failed_beeps_error():
    d, clk = make()
    turn = clk.frame(d, wake=True)[1].turn
    clk.frames(d, [0.9] * 4 + [0.0] * 3)
    cmds = d.on_event(clk.t, TurnFailed(turn.id, "boom"))
    assert kinds(cmds) == ["Beep", "TurnEnded"] and cmds[0].kind == "error" and cmds[1].status == "error"


def test_followup_speech_starts_new_turn_without_wake():
    d, clk = make(followup_seconds=3.0)
    t1 = clk.frame(d, wake=True)[1].turn
    clk.frames(d, [0.9] * 4 + [0.0] * 3)
    d.on_event(clk.t, ReplyStarted(t1.id))
    d.on_event(clk.t, ReplyDone(t1.id))
    assert d.state == State.FOLLOWUP
    cmds = clk.frames(d, [0.0] * 3 + [0.95] * 3)
    assert kinds(cmds) == ["StartTurn"]
    t2 = cmds[0].turn
    assert t2.followup and t2.t_wake is None and d.state == State.LISTENING
    cmds = clk.frames(d, [0.95] * 3 + [0.0] * 3)
    assert kinds(cmds) == ["Transcribe"] and cmds[0].turn is t2
    # 跟进模式下前面的 pre-roll 也在句子里
    assert len(t2.audio) >= 6 * 1280


def test_followup_disabled_goes_idle():
    d, clk = make(followup_seconds=0.0)
    t1 = clk.frame(d, wake=True)[1].turn
    clk.frames(d, [0.9] * 4 + [0.0] * 3)
    cmds = d.on_event(clk.t, ReplyDone(t1.id))
    assert kinds(cmds) == ["TurnEnded"] and d.state == State.IDLE


def test_timer_delivered_when_idle_and_parked_when_busy():
    d, clk = make(followup_seconds=0.0)
    cmds = d.on_event(clk.t, TimerFired({"label": "十分鐘"}))
    assert kinds(cmds) == ["StartTurn", "Beep", "Deliver"] and cmds[1].kind == "alarm"
    alarm = cmds[0].turn
    assert alarm.kind == "alarm" and alarm.payload == {"label": "十分鐘"} and d.state == State.THINKING
    d.on_event(clk.t, ReplyStarted(alarm.id))
    d.on_event(clk.t, ReplyDone(alarm.id))
    assert d.state == State.IDLE
    # 忙的时候先存起来，回到 IDLE 再送
    turn = clk.frame(d, wake=True)[1].turn
    assert d.on_event(clk.t, TimerFired("t2")) == []
    clk.frames(d, [0.9] * 4 + [0.0] * 3)
    cmds = d.on_event(clk.t, ReplyDone(turn.id))
    assert kinds(cmds) == ["TurnEnded", "StartTurn", "Beep", "Deliver"]
    assert isinstance(cmds[3], Deliver) and cmds[3].turn.payload == "t2"


def test_commands_are_plain_dataclasses():
    d, clk = make()
    cmds = clk.frame(d, wake=True)
    assert isinstance(cmds[1], StartTurn) and isinstance(cmds[0], Beep)
    assert not isinstance(cmds[0], (Cancel, Transcribe, TurnEnded))
