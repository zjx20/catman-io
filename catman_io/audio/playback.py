"""扬声器播放（语音输出的最后一步）。TTS 模块产出 PCM 后由这里播出。"""

from __future__ import annotations

import numpy as np

from .frames import SAMPLE_RATE, to_int16


def play(
    audio: np.ndarray, sample_rate: int = SAMPLE_RATE, device: int | str | None = None, blocking: bool = True
) -> None:
    import sounddevice as sd

    sd.play(to_int16(audio), samplerate=sample_rate, device=device, blocking=blocking)


def stop() -> None:
    import sounddevice as sd

    sd.stop()
