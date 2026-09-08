"""音频基础：采样率 / 帧长约定、格式转换、麦克风采集与播放。

整个软件栈内部统一用 **16 kHz、单声道、int16**，以 **80 ms（1280 采样点）** 为一帧——
这是 openWakeWord 的处理单位，也足够作为 VAD / 推流的粒度。
"""

from .frames import (
    FRAME_SAMPLES,
    FRAME_SECONDS,
    SAMPLE_RATE,
    iter_frames,
    read_wav,
    resample,
    to_float32,
    to_int16,
)

__all__ = [
    "FRAME_SAMPLES",
    "FRAME_SECONDS",
    "SAMPLE_RATE",
    "iter_frames",
    "read_wav",
    "resample",
    "to_float32",
    "to_int16",
]
