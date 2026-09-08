"""粤语语音合成（TTS）——规划中。

第一版打算直接用 edge-tts 的 zh-HK 音色（联网、免费、音质好），训练唤醒词用的也是它；
后续再评估可离线的粤语 TTS。产出 16 kHz PCM 交给 catman_io.audio.playback 播放。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Synthesizer(Protocol):
    def synthesize(self, text: str) -> np.ndarray:
        """文本 → 16 kHz int16 单声道 PCM。"""
