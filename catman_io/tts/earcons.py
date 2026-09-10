"""提示音：代码生成的短音，不带音频素材。唤醒、超时、出错、思考中、闹铃、没听清。"""

from __future__ import annotations

from functools import cache

import numpy as np

from catman_io.audio.frames import SAMPLE_RATE, to_int16

KINDS = ("wake", "timeout", "error", "thinking", "alarm", "reprompt")


def tone(
    freq: float, seconds: float, *, volume: float = 0.3, fade: float = 0.01, harmonics: int = 1
) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    x = np.zeros(n, dtype=np.float32)
    for k in range(1, harmonics + 1):
        x += (np.sin(2 * np.pi * freq * k * t) / k).astype(np.float32)
    x *= volume / max(1e-6, np.abs(x).max())
    f = max(1, int(fade * SAMPLE_RATE))
    ramp = np.linspace(0, 1, f, dtype=np.float32)
    x[:f] *= ramp
    x[-f:] *= ramp[::-1]
    return x


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


@cache
def earcon(kind: str) -> np.ndarray:
    """→ 16 kHz int16。未知类型给一声短音而不是报错（提示音不该让管线挂掉）。"""
    if kind == "wake":
        x = np.concatenate([tone(880, 0.09), tone(1320, 0.11)])
    elif kind == "timeout":
        x = tone(440, 0.2, volume=0.25)
    elif kind == "error":
        x = np.concatenate([tone(330, 0.12, harmonics=3), silence(0.04), tone(220, 0.18, harmonics=3)])
    elif kind == "thinking":
        x = tone(660, 0.08, volume=0.15)
    elif kind == "alarm":
        one = np.concatenate([tone(880, 0.12), tone(1100, 0.12), tone(1320, 0.18), silence(0.15)])
        x = np.concatenate([one, one, one])
    elif kind == "reprompt":
        x = np.concatenate([tone(1320, 0.09), tone(880, 0.11)])
    else:
        x = tone(1000, 0.1)
    return to_int16(x)
