"""语音流水线骨架：采集 → 唤醒 → （VAD → 识别 / 推流 → 意图 → 回复 → 播放）。

目前只接通了前两级。后面每一级都以"消费 80 ms 帧"的方式挂进来，唤醒后的状态切换在这里做。
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from catman_io.audio.capture import MicCapture
from catman_io.config import Config
from catman_io.wakeword import Detection, WakeWordDetector

log = logging.getLogger(__name__)


class VoicePipeline:
    def __init__(self, config: Config, on_wake: Callable[[Detection], None] | None = None):
        self.config = config
        self.on_wake = on_wake or (lambda d: log.info("wake: %s score=%.2f", d.model, d.score))
        w = config.wakeword
        self.detector = WakeWordDetector(
            w.models or None,
            threshold=w.threshold,
            patience=w.patience,
            cooldown=w.cooldown,
            vad_threshold=w.vad_threshold,
        )

    def run(self) -> None:
        a = self.config.audio
        with MicCapture(device=a.device, channel=a.channel, sample_rate=a.sample_rate) as mic:
            log.info("listening for %s ...", ", ".join(self.detector.names))
            for frame in mic.frames():
                for det in self.detector.process(frame):
                    self.on_wake(det)
                    # TODO: 唤醒后进入"聆听"状态：VAD 切句 → ASR / 推流到后端 → 意图 → TTS 播放
