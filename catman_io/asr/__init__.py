"""粤语语音转文字（ASR）——规划中。

候选（都支持粤语 yue）：SenseVoice-Small（速度快、自带情绪/事件标签）、Whisper 系列、
FunASR Paraformer 粤语模型。
要求：CPU 可跑、离线可用、支持流式或准流式。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Transcript:
    text: str
    language: str = "yue"
    confidence: float | None = None


class SpeechRecognizer(Protocol):
    def transcribe(self, audio: np.ndarray) -> Transcript:
        """整段识别：16 kHz int16 单声道 → 文本。"""
