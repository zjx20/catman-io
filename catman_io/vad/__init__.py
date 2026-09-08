"""语音活动检测（VAD）——规划中。

职责：唤醒之后判断用户什么时候开始说、什么时候说完（端点检测），给 ASR / 推流切句。
计划：silero-vad（openWakeWord 已随包带了 silero_vad.onnx，可直接复用，无需额外下载）。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class VoiceActivityDetector(Protocol):
    def process(self, frame: np.ndarray) -> float:
        """输入一帧 16 kHz int16，返回这一帧是人声的概率。"""

    def reset(self) -> None: ...
