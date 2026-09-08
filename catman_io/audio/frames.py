"""帧与格式工具（只依赖 numpy / scipy）。"""

from __future__ import annotations

import wave
from collections.abc import Iterator
from math import gcd
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # 80 ms @ 16 kHz，openWakeWord 的最小处理单位
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE


def to_int16(x: np.ndarray) -> np.ndarray:
    """float(-1..1) 或 int16 → int16。"""
    x = np.asarray(x)
    if x.dtype == np.int16:
        return x
    if np.issubdtype(x.dtype, np.floating):
        return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
    if x.dtype == np.int32:
        return (x >> 16).astype(np.int16)
    return x.astype(np.int16)


def to_float32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.floating):
        return x.astype(np.float32)
    if x.dtype == np.int16:
        return (x / 32768.0).astype(np.float32)
    if x.dtype == np.int32:
        return (x / 2147483648.0).astype(np.float32)
    return x.astype(np.float32)


def resample(x: np.ndarray, sr_from: int, sr_to: int = SAMPLE_RATE) -> np.ndarray:
    """多相重采样，保持 dtype（int16 进 int16 出）。"""
    if sr_from == sr_to:
        return x
    from scipy.signal import resample_poly

    g = gcd(sr_from, sr_to)
    y = resample_poly(to_float32(x), sr_to // g, sr_from // g)
    return to_int16(y) if x.dtype == np.int16 else y.astype(np.float32)


def read_wav(path: str | Path, channel: int = 0) -> np.ndarray:
    """读 WAV 为 16 kHz 单声道 int16（多声道取指定声道，采样率不同则重采样）。"""
    with wave.open(str(path), "rb") as w:
        n_ch, width, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16)
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32)
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128) << 8
    else:
        raise ValueError(f"unsupported sample width {width} bytes in {path}")
    if n_ch > 1:
        data = data.reshape(-1, n_ch)[:, min(channel, n_ch - 1)]
    data = to_int16(data)
    return resample(data, sr, SAMPLE_RATE)


def iter_frames(x: np.ndarray, frame_samples: int = FRAME_SAMPLES, pad: bool = True) -> Iterator[np.ndarray]:
    """把一段音频切成定长帧；最后不足一帧的补零（pad=False 则丢弃）。"""
    x = np.asarray(x)
    n_full = len(x) // frame_samples
    for i in range(n_full):
        yield x[i * frame_samples : (i + 1) * frame_samples]
    rest = len(x) - n_full * frame_samples
    if rest and pad:
        tail = np.zeros(frame_samples, dtype=x.dtype)
        tail[:rest] = x[-rest:]
        yield tail
