"""对话状态机：唤醒 → 聆听 → 思考 → 播报 → 跟进。纯逻辑，单写者（只有帧循环线程碰它）。

输入是每帧的（唤醒检测结果、人声概率）和其他线程送来的事件；输出是一串命令，由 pipeline 执行，
命令的执行都不阻塞（塞队列、置 Event、停播放）。这样状态机可以离线用假数据逐帧测试。

    IDLE ─唤醒─▶ LISTENING ─说完─▶ THINKING ─开口─▶ SPEAKING ─播完─▶ FOLLOWUP ─超时─▶ IDLE
"""

from __future__ import annotations

import enum
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from catman_io.config import DialogConfig, VadConfig
from catman_io.vad.endpoint import Endpointer, NoSpeech, SpeechStart, Utterance


class State(str, enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    FOLLOWUP = "followup"


@dataclass
class Turn:
    """一回合：从唤醒（或跟进时开口）到回复播完。时间戳是 pipeline 传进来的 now（秒）。"""

    id: str
    gen: int
    kind: str = "voice"  # voice | alarm
    t_wake: float | None = None
    wake_model: str | None = None
    wake_score: float | None = None
    barge_in: bool = False
    followup: bool = False
    reprompted: bool = False
    t_speech_start: float | None = None
    t_speech_end: float | None = None
    t_reply_started: float | None = None
    t_reply_done: float | None = None
    t_interrupted: float | None = None
    forced_end: bool = False
    audio: np.ndarray | None = None
    wake_audio: np.ndarray | None = None  # 唤醒前后那一小段，留作训练样本
    status: str = "open"
    payload: Any = None
    info: dict[str, Any] = field(default_factory=dict)  # pipeline / worker 往里放识别结果、延迟等
    cancelled: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)


# ---- 其他线程送来的事件 ----


@dataclass(frozen=True)
class TranscriptEmpty:
    turn_id: str


@dataclass(frozen=True)
class ReplyStarted:
    turn_id: str


@dataclass(frozen=True)
class ReplyDone:
    turn_id: str
    followup: bool = False  # 动作要求追问（用更长的跟进窗口）


@dataclass(frozen=True)
class TurnFailed:
    turn_id: str
    error: str


@dataclass(frozen=True)
class TimerFired:
    timer: Any


Event = TranscriptEmpty | ReplyStarted | ReplyDone | TurnFailed | TimerFired


# ---- 状态机发出的命令 ----


@dataclass(frozen=True)
class Beep:
    kind: str  # wake | timeout | error | thinking | alarm | reprompt


@dataclass(frozen=True)
class StartTurn:
    turn: Turn


@dataclass(frozen=True)
class Transcribe:
    turn: Turn


@dataclass(frozen=True)
class Cancel:
    turn: Turn


@dataclass(frozen=True)
class StopSpeaking:
    pass


@dataclass(frozen=True)
class TurnEnded:
    turn: Turn
    status: str  # ok | no_speech | empty | cancelled | timeout | error


@dataclass(frozen=True)
class Deliver:
    turn: Turn  # kind == alarm：播闹铃并念定时器内容


Command = Beep | StartTurn | Transcribe | Cancel | StopSpeaking | TurnEnded | Deliver


class Dialog:
    def __init__(self, cfg: DialogConfig | None = None, vad: VadConfig | None = None):
        self.cfg = cfg or DialogConfig()
        self.endpointer = Endpointer(vad or VadConfig())
        self.state = State.IDLE
        self.turn: Turn | None = None
        self.gen = 0
        self.t_state = 0.0
        self._thinking_beeped = False
        self._followup_window = 0.0
        self._pending_timers: list[Any] = []
        # 状态变化的回调（在调用 on_frame / on_event 的那个线程里被调）：网页 demo 用它把状态推给页面
        self.on_state: Callable[[State, float], None] | None = None

    # ---- 输入 ----

    def on_frame(self, now: float, frame: np.ndarray, wake: Sequence[Any], prob: float) -> list[Command]:
        """每帧调用。wake 是这一帧的唤醒检测结果（有则视为唤醒），prob 是这一帧的人声概率。"""
        if wake:
            return self._on_wake(now, wake[0])
        if self.state == State.LISTENING:
            return self._listen(now, frame, prob)
        if self.state == State.FOLLOWUP:
            if now - self.t_state >= self._followup_window:
                return self._enter_idle(now)
            if (now - self.t_state) * 1000 < self.endpointer.cfg.followup_guard_ms:
                return []
            for ev in self.endpointer.feed(frame, prob):
                if isinstance(ev, SpeechStart):
                    turn = self._new_turn(now, kind="voice")
                    turn.followup = True
                    turn.t_speech_start = now
                    self._enter(State.LISTENING, now)
                    return [StartTurn(turn)]
            return []
        if self.state == State.THINKING:
            cmds: list[Command] = []
            if not self._thinking_beeped and now - self.t_state >= self.cfg.thinking_beep_after:
                self._thinking_beeped = True
                cmds.append(Beep("thinking"))
            if now - self.t_state >= self.cfg.turn_timeout and self.turn is not None:
                turn = self.turn
                cmds += [Cancel(turn), self._end_turn(turn, "timeout")]
                cmds += self._enter_idle(now)
            return cmds
        return []

    def on_event(self, now: float, ev: Event) -> list[Command]:
        if isinstance(ev, TimerFired):
            return self._on_timer(now, ev.timer)
        turn = self.turn
        if turn is None or turn.id != ev.turn_id:
            return []  # 过期回合的事件
        if isinstance(ev, TranscriptEmpty) and self.state == State.THINKING:
            if self.cfg.reprompt_on_empty and not turn.reprompted and turn.kind == "voice":
                turn.reprompted = True
                self.endpointer.reset()
                self._enter(State.LISTENING, now)
                return [Beep("reprompt")]
            cmds: list[Command] = [self._end_turn(turn, "empty")]
            return cmds + self._enter_idle(now)
        if isinstance(ev, ReplyStarted) and self.state == State.THINKING:
            turn.t_reply_started = now
            self._enter(State.SPEAKING, now)
            return []
        if isinstance(ev, ReplyDone) and self.state in (State.THINKING, State.SPEAKING):
            turn.t_reply_done = now
            cmds = [self._end_turn(turn, "ok")]
            window = self.cfg.followup_question_seconds if ev.followup else self.cfg.followup_seconds
            if window > 0:
                self._followup_window = window
                self.endpointer.reset(followup=True)
                self._enter(State.FOLLOWUP, now)
                return cmds
            return cmds + self._enter_idle(now)
        if isinstance(ev, TurnFailed) and self.state in (State.THINKING, State.SPEAKING):
            cmds = [Beep("error"), self._end_turn(turn, "error")]
            return cmds + self._enter_idle(now)
        return []

    # ---- 内部 ----

    def _on_wake(self, now: float, det: Any) -> list[Command]:
        cmds: list[Command] = []
        if self.state == State.LISTENING and self.turn is not None and self.turn.kind == "voice":
            # 聆听中再叫一次：重新开始听，不开新回合
            self.turn.t_wake = now
            self.endpointer.reset()
            self._enter(State.LISTENING, now)
            return [Beep("wake")]
        barge_in = False
        if self.state in (State.THINKING, State.SPEAKING) and self.turn is not None:
            old = self.turn
            if self.state == State.SPEAKING:
                old.t_interrupted = now
                barge_in = True
                cmds.append(StopSpeaking())
            cmds += [Cancel(old), self._end_turn(old, "cancelled")]
        turn = self._new_turn(now, kind="voice")
        turn.t_wake = now
        turn.wake_model = getattr(det, "model", None)
        turn.wake_score = getattr(det, "score", None)
        turn.barge_in = barge_in
        self.endpointer.reset()
        self._enter(State.LISTENING, now)
        cmds += [Beep("wake"), StartTurn(turn)]
        return cmds

    def _listen(self, now: float, frame: np.ndarray, prob: float) -> list[Command]:
        turn = self.turn
        assert turn is not None
        for ev in self.endpointer.feed(frame, prob):
            if isinstance(ev, SpeechStart):
                turn.t_speech_start = now
            elif isinstance(ev, Utterance):
                turn.audio = ev.audio
                turn.t_speech_end = now
                turn.forced_end = ev.forced
                if turn.t_speech_start is None:
                    turn.t_speech_start = now - (ev.t_end - ev.t_start)
                self._enter(State.THINKING, now)
                return [Transcribe(turn)]
            elif isinstance(ev, NoSpeech):
                cmds: list[Command] = [Beep("timeout"), self._end_turn(turn, "no_speech")]
                return cmds + self._enter_idle(now)
        return []

    def _on_timer(self, now: float, timer: Any) -> list[Command]:
        if self.state not in (State.IDLE, State.FOLLOWUP):
            self._pending_timers.append(timer)
            return []
        turn = self._new_turn(now, kind="alarm")
        turn.payload = timer
        self._enter(State.THINKING, now)
        return [StartTurn(turn), Beep("alarm"), Deliver(turn)]

    def _new_turn(self, now: float, *, kind: str) -> Turn:
        self.gen += 1
        turn = Turn(id=f"{time.strftime('%Y%m%d-%H%M%S')}-{self.gen:04d}", gen=self.gen, kind=kind)
        self.turn = turn
        return turn

    def _end_turn(self, turn: Turn, status: str) -> TurnEnded:
        turn.status = status
        if self.turn is turn:
            self.turn = None
        return TurnEnded(turn, status)

    def _enter(self, state: State, now: float) -> None:
        self.state = state
        self.t_state = now
        if state == State.THINKING:
            self._thinking_beeped = False
        if self.on_state is not None:
            self.on_state(state, now)

    def _enter_idle(self, now: float) -> list[Command]:
        self._enter(State.IDLE, now)
        self.turn = None
        if self._pending_timers:
            return self._on_timer(now, self._pending_timers.pop(0))
        return []
