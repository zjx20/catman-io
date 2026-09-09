"""数据增强（纯 numpy / scipy，不依赖 torch 和 speechbrain）。

思路沿用 openWakeWord 的训练脚本：把每条 TTS 短句放进固定长度（默认 2 秒）的窗口里，
正样本**右对齐**（唤醒词刚说完就是模型该触发的时刻），再随机叠加环境噪声、人声嘈杂、
有色噪声、房间混响、滤波、失真和音量变化。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt

from .config import AugmentConfig

log = logging.getLogger(__name__)

SR = 16000
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def load_audio_16k(path: Path) -> np.ndarray:
    """任意音频 → 16 kHz float32 单声道。"""
    from .tts import decode_audio, resample

    x, sr = decode_audio(Path(path))
    return resample(x, sr, SR).astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def active_rms(x: np.ndarray, floor: float = 1e-4) -> float:
    """只在非静音采样上算 RMS，避免窗口里的补零把信号能量摊薄。"""
    mask = np.abs(x) > floor
    return rms(x[mask]) if mask.any() else rms(x)


def trim_to_speech(
    x: np.ndarray, sr: int = SR, pad_before: float = 0.1, pad_after: float = 0.08, min_db: float = -50.0
) -> np.ndarray:
    """按相对本底噪声的阈值裁掉首尾静音，给真人录音用（录音前后总有一段房间噪声）。

    阈值 = max(本底 + 6 dB, min_db)，本底取 10 ms 帧能量的第 20 百分位；起点前多留 100 ms 保住轻辅音。
    """
    hop = sr // 100
    if len(x) < 2 * hop:
        return x
    n = len(x) // hop
    frames = x[: n * hop].reshape(n, hop)
    db = 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-9)
    threshold = max(float(np.percentile(db, 20)) + 6.0, min_db)
    active = np.where(db > threshold)[0]
    if len(active) == 0:
        return x
    start = max(0, int(active[0] * hop - pad_before * sr))
    end = min(len(x), int((active[-1] + 1) * hop + pad_after * sr))
    return x[start:end]


class AudioPool:
    """一堆音频拼在一起随机切片：环境噪声池、人声嘈杂池都用它。"""

    def __init__(self, clips: list[np.ndarray]):
        self.clips = [c.astype(np.float32) for c in clips if len(c) > SR // 10]
        if not self.clips:
            raise ValueError("audio pool is empty")
        lengths = np.array([len(c) for c in self.clips], dtype=np.float64)
        self.weights = lengths / lengths.sum()

    def __len__(self) -> int:
        return len(self.clips)

    @property
    def seconds(self) -> float:
        return sum(len(c) for c in self.clips) / SR

    def random_segment(self, n: int, rng: np.random.Generator) -> np.ndarray:
        clip = self.clips[int(rng.choice(len(self.clips), p=self.weights))]
        if len(clip) < n:
            clip = np.tile(clip, int(np.ceil(n / len(clip))))
        start = int(rng.integers(0, len(clip) - n + 1))
        return clip[start : start + n]

    @classmethod
    def from_dirs(
        cls, dirs: list[str | Path], cache: Path | None = None, max_seconds: float = 3600
    ) -> AudioPool | None:
        files = sorted(p for d in dirs for p in Path(d).rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
        if not files:
            return None
        if cache is not None and cache.exists():
            meta = json.loads(cache.with_suffix(".json").read_text())
            if meta.get("files") == [str(f) for f in files]:
                data = np.load(cache)
                bounds = meta["bounds"]
                return cls([data[a:b] for a, b in zip(bounds[:-1], bounds[1:], strict=True)])
        clips: list[np.ndarray] = []
        total = 0.0
        for f in files:
            try:
                x = load_audio_16k(f)
            except Exception as e:  # noqa: BLE001
                log.warning("skip %s: %s", f, e)
                continue
            peak = float(np.abs(x).max()) if len(x) else 0.0
            if peak > 0:
                clips.append(x / peak)
                total += len(x) / SR
            if total >= max_seconds:
                break
        if not clips:
            return None
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            bounds = np.concatenate([[0], np.cumsum([len(c) for c in clips])]).tolist()
            np.save(cache, np.concatenate(clips))
            cache.with_suffix(".json").write_text(
                json.dumps({"files": [str(f) for f in files], "bounds": bounds})
            )
        log.info("audio pool: %d files, %.0f s", len(clips), total)
        return cls(clips)

    @classmethod
    def from_clips(cls, clips: list[np.ndarray]) -> AudioPool | None:
        return cls(clips) if clips else None


def load_rirs(rir_dir: Path | None) -> list[np.ndarray]:
    """房间冲激响应：归一化并把直达声之前的静音切掉，卷积后就不会引入额外延迟。"""
    if rir_dir is None or not Path(rir_dir).exists():
        return []
    rirs = []
    for f in sorted(Path(rir_dir).rglob("*.wav")):
        try:
            r = load_audio_16k(f)
        except Exception as e:  # noqa: BLE001
            log.warning("skip RIR %s: %s", f, e)
            continue
        peak = float(np.abs(r).max())
        if peak <= 0:
            continue
        r = r / peak
        onset = int(np.argmax(np.abs(r) > 0.01))
        r = r[onset : onset + SR]  # 最多留 1 秒尾巴
        rirs.append(r.astype(np.float32))
    log.info("loaded %d RIRs from %s", len(rirs), rir_dir)
    return rirs


# ---------------------------------------------------------------- 单项变换
def speed_perturb(x: np.ndarray, factor: float) -> np.ndarray:
    """变速同时变调（像磁带快放），factor>1 变快变短。"""
    if abs(factor - 1.0) < 1e-3:
        return x
    return resample_poly(x, 100, int(round(100 * factor))).astype(np.float32)


def fit_clip(
    x: np.ndarray, total: int, rng: np.random.Generator, end_jitter: float = 0.2, align: str = "right"
) -> np.ndarray:
    """把片段放进固定长度窗口。right：末尾距窗口末尾 0~end_jitter 秒；random：随机位置 / 随机裁剪。"""
    out = np.zeros(total, dtype=np.float32)
    if len(x) >= total:
        if align == "right":
            return x[-total:].astype(np.float32)
        start = int(rng.integers(0, len(x) - total + 1))
        return x[start : start + total].astype(np.float32)
    if align == "right":
        end = total - int(rng.uniform(0, end_jitter) * SR)
        start = max(0, end - len(x))
    else:
        start = int(rng.integers(0, total - len(x) + 1))
    out[start : start + len(x)] = x
    return out


def colored_noise(n: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """1/f^beta 噪声：beta=0 白噪声，1 粉红，2 棕色。输出 RMS≈1。"""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]
    spec = spec / (freqs ** (beta / 2))
    x = np.fft.irfft(spec, n)
    return (x / (rms(x) + 1e-9)).astype(np.float32)


def mix_at_snr(sig: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    s = active_rms(sig)
    n = rms(noise)
    if n <= 0 or s <= 0:
        return sig
    gain = s / (n * 10 ** (snr_db / 20))
    return (sig + noise[: len(sig)] * gain).astype(np.float32)


def apply_rir(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    y = fftconvolve(x, rir)[: len(x)]
    ry, rx = rms(y), rms(x)
    if ry > 0:
        y = y * (rx / ry)
    return y.astype(np.float32)


def bandstop(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    center = 10 ** rng.uniform(np.log10(200), np.log10(4000))
    bw = center * rng.uniform(0.3, 1.0)
    lo, hi = max(50.0, center - bw / 2), min(7900.0, center + bw / 2)
    sos = butter(2, [lo, hi], btype="bandstop", fs=SR, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def lowpass(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sos = butter(4, rng.uniform(2500, 6000), btype="lowpass", fs=SR, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def tanh_distortion(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    peak = float(np.abs(x).max())
    if peak <= 0:
        return x
    drive = 1.0 + rng.uniform(0.5, 4.0)
    y = np.tanh(x / peak * drive)
    return (y / (np.abs(y).max() + 1e-9) * peak).astype(np.float32)


# ---------------------------------------------------------------- 组合
class Augmenter:
    def __init__(
        self,
        cfg: AugmentConfig,
        background: AudioPool | None = None,
        babble: AudioPool | None = None,
        rirs: list[np.ndarray] | None = None,
        seed: int = 0,
    ):
        self.cfg = cfg
        self.background = background
        self.babble = babble
        self.rirs = rirs or []
        self.rng = np.random.default_rng(seed)
        self.total = int(round(cfg.clip_seconds * SR))

    def __call__(self, x: np.ndarray, positive: bool) -> np.ndarray:
        cfg, rng = self.cfg, self.rng
        x = x.astype(np.float32)
        if rng.random() < cfg.p_speed:
            x = speed_perturb(x, float(rng.choice(cfg.speed_factors)))
        # 负样本一半右对齐（"小貓"刚说完这种最容易混的对齐），一半随机位置
        align = "right" if positive or rng.random() < 0.5 else "random"
        x = fit_clip(x, self.total, rng, cfg.end_jitter, align)
        if self.rirs and rng.random() < cfg.p_rir:
            x = apply_rir(x, self.rirs[int(rng.integers(len(self.rirs)))])
        if self.background is not None and rng.random() < cfg.p_background:
            x = mix_at_snr(x, self.background.random_segment(self.total, rng), rng.uniform(*cfg.snr_db))
        if self.babble is not None and rng.random() < cfg.p_babble:
            x = mix_at_snr(x, self.babble.random_segment(self.total, rng), rng.uniform(*cfg.babble_snr_db))
        if rng.random() < cfg.p_colored_noise:
            x = mix_at_snr(
                x,
                colored_noise(self.total, float(rng.choice([0.0, 1.0, 2.0])), rng),
                rng.uniform(*cfg.colored_snr_db),
            )
        if rng.random() < cfg.p_bandstop:
            x = bandstop(x, rng)
        if rng.random() < cfg.p_lowpass:
            x = lowpass(x, rng)
        if rng.random() < cfg.p_distortion:
            x = tanh_distortion(x, rng)
        peak = float(np.abs(x).max())
        if peak > 0:
            x = x / peak * rng.uniform(*cfg.peak_range)
        return np.clip(x, -1.0, 1.0).astype(np.float32)


def to_int16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
