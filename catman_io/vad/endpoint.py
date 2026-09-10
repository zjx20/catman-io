"""端点检测：由逐帧的人声概率判断一句话何时开始、何时说完。纯逻辑，不含模型，方便单测。

状态：WAIT（等人开口）→ SPEECH（在说）→ TRAIL（停顿中，可能只是换气）→ 发出 Utterance 后进入 DONE，
直到 :meth:`Endpointer.reset`。时间都以帧数计（一帧 80 ms），事件里的秒数相对于上次 reset。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from catman_io.audio.frames import FRAME_SECONDS
from catman_io.config import VadConfig


@dataclass(frozen=True)
class SpeechStart:
    t: float


@dataclass(frozen=True)
class Utterance:
    audio: np.ndarray
    t_start: float
    t_end: float
    forced: bool  # True = 超过最长时长被强制切断


@dataclass(frozen=True)
class NoSpeech:
    t: float


EndpointEvent = SpeechStart | Utterance | NoSpeech

WAIT, SPEECH, TRAIL, DONE = "wait", "speech", "trail", "done"


def _frames(ms: int) -> int:
    return max(0, int(round(ms / 1000.0 / FRAME_SECONDS)))


class Endpointer:
    def __init__(self, cfg: VadConfig | None = None):
        self.cfg = cfg or VadConfig()
        self.reset()

    def reset(self, *, followup: bool = False) -> None:
        """回到 WAIT。followup=True 用更严格的起声阈值、并且不做"没人说话"超时（由对话层管窗口）。"""
        cfg = self.cfg
        self.start_threshold = cfg.followup_start_threshold if followup else cfg.start_threshold
        self.start_frames = max(1, cfg.followup_start_frames if followup else cfg.start_frames)
        self.timeout_frames = 0 if followup else _frames(cfg.no_speech_timeout_ms)
        self.state = WAIT
        self._n = 0  # 自 reset 起喂进来的帧数
        self._run = 0  # 连续高于起声阈值的帧数
        self._sil = 0  # TRAIL 里连续静音帧数
        self._pre: deque[np.ndarray] = deque(maxlen=_frames(cfg.pre_roll_ms) + self.start_frames)
        self._buf: list[np.ndarray] = []
        self._speech_first = 0  # _buf 里第一帧人声的下标
        self._speech_last = 0  # _buf 里最后一帧人声的下标

    @property
    def elapsed(self) -> float:
        return self._n * FRAME_SECONDS

    def feed(self, frame: np.ndarray, prob: float) -> list[EndpointEvent]:
        if self.state == DONE:
            return []
        self._n += 1
        cfg = self.cfg
        if self.state == WAIT:
            self._pre.append(frame)
            self._run = self._run + 1 if prob >= self.start_threshold else 0
            if self._run >= self.start_frames:
                self._buf = list(self._pre)
                self._speech_first = max(0, len(self._buf) - self.start_frames)
                self._speech_last = len(self._buf) - 1
                self._run = 0
                self.state = SPEECH
                return [SpeechStart(t=(self._n - self.start_frames) * FRAME_SECONDS)]
            if self.timeout_frames and self._n >= self.timeout_frames:
                self.state = DONE
                return [NoSpeech(t=self.elapsed)]
            return []

        self._buf.append(frame)
        if prob >= cfg.end_threshold:
            self._speech_last = len(self._buf) - 1
            self._sil = 0
            self.state = SPEECH
        else:
            self._sil += 1
            self.state = TRAIL
        speech_frames = self._speech_last - self._speech_first + 1
        if speech_frames * FRAME_SECONDS * 1000 >= cfg.max_utterance_ms:
            return [self._emit(forced=True)]
        if self.state == TRAIL and self._sil * FRAME_SECONDS * 1000 >= cfg.trailing_silence_ms:
            if speech_frames * FRAME_SECONDS * 1000 < cfg.min_speech_ms:
                # 太短，当噪声：回到 WAIT，但超时计数继续
                self._buf = []
                self._sil = 0
                self.state = WAIT
                if self.timeout_frames and self._n >= self.timeout_frames:
                    self.state = DONE
                    return [NoSpeech(t=self.elapsed)]
                return []
            return [self._emit(forced=False)]
        return []

    def _emit(self, *, forced: bool) -> Utterance:
        end = min(len(self._buf), self._speech_last + 1 + _frames(self.cfg.tail_pad_ms))
        audio = np.concatenate(self._buf[:end]) if self._buf else np.zeros(0, dtype=np.int16)
        t_end = self.elapsed - (len(self._buf) - end) * FRAME_SECONDS
        t_start = t_end - end * FRAME_SECONDS
        self.state = DONE
        self._buf = []
        return Utterance(audio=audio, t_start=max(0.0, t_start), t_end=t_end, forced=forced)
