"""语音活动检测（VAD）与端点检测。

- :class:`SileroVAD`：逐帧给出人声概率（silero v4，openWakeWord 随包带的模型，无需额外下载）。
- :class:`Endpointer`（``endpoint.py``）：由概率序列判断一句话何时开始、何时说完，给 ASR 切句。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .endpoint import Endpointer, NoSpeech, SpeechStart, Utterance
from .silero import SileroVAD


class VoiceActivityDetector(Protocol):
    def process(self, frame: np.ndarray) -> float:
        """输入一帧 16 kHz int16，返回这一帧是人声的概率。"""

    def reset(self) -> None: ...


__all__ = ["Endpointer", "NoSpeech", "SileroVAD", "SpeechStart", "Utterance", "VoiceActivityDetector"]
