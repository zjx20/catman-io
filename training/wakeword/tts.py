"""用 edge-tts 把文本清单展开成大量 16 kHz 单声道 WAV。

每条 (文本, 音色, 语速, 音高) 组合一个文件，文件名由这四元组的哈希决定，所以重复运行只补缺失的。
产物目录结构::

    <workdir>/clips/
        positive_<hash>.wav
        negative_<hash>.wav
        manifest.jsonl        # 每行一条：text/voice/rate/pitch/label/lang/wav/duration/split
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import ssl
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .config import TrainingConfig

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class SynthJob:
    text: str
    voice: str
    rate: str
    pitch: str
    label: str  # "positive" | "adversarial" | "homophone" | "general"
    lang: str  # "zh-HK" | "zh-CN" | "en-US"

    @property
    def key(self) -> str:
        h = hashlib.sha1(f"{self.text}|{self.voice}|{self.rate}|{self.pitch}".encode()).hexdigest()
        return h[:16]

    @property
    def stem(self) -> str:
        prefix = "positive" if self.label == "positive" else "negative"
        return f"{prefix}_{self.key}"


@dataclass
class ClipRecord:
    text: str
    voice: str
    rate: str
    pitch: str
    label: str
    lang: str
    wav: str
    duration: float
    split: str  # "train" | "val"

    @property
    def is_positive(self) -> bool:
        return self.label == "positive"


# ---------------------------------------------------------------- 计划
def plan_jobs(cfg: TrainingConfig) -> list[SynthJob]:
    tts, data = cfg.tts, cfg.data
    jobs: list[SynthJob] = []

    positives = [
        SynthJob(text, voice, rate, pitch, "positive", "zh-HK")
        for text in data.resolved_positive_phrases()
        for voice in tts.voices
        for rate in tts.rates
        for pitch in tts.pitches
    ]
    negatives = [
        SynthJob(text, voice, rate, pitch, label, "zh-HK")
        for label, texts in (
            ("adversarial", data.resolved_adversarial_phrases()),
            ("homophone", data.resolved_homophone_phrases()),
            ("general", data.resolved_general_phrases()),
        )
        for text in texts
        for voice in tts.voices
        for rate in tts.negative_rates
        for pitch in tts.negative_pitches
    ]
    for lang, texts in data.resolved_other_language_phrases().items():
        voices = tts.other_language_voices.get(lang, [])
        negatives += [
            SynthJob(text, voice, rate, "+0Hz", "general", lang)
            for text in texts
            for voice in voices
            for rate in tts.negative_rates[:2]
        ]

    wake = cfg.wake_phrase
    bad = [j.text for j in negatives if wake in j.text]
    if bad:
        raise ValueError(
            f"negative phrases must not contain the wake phrase {wake!r}: {sorted(set(bad))[:5]}"
        )

    rng = random.Random(1234)
    if tts.max_positive_clips and len(positives) > tts.max_positive_clips:
        positives = rng.sample(positives, tts.max_positive_clips)
    if tts.max_negative_clips and len(negatives) > tts.max_negative_clips:
        negatives = rng.sample(negatives, tts.max_negative_clips)

    jobs = positives + negatives
    log.info("planned %d positive + %d negative clips", len(positives), len(negatives))
    return jobs


def split_for(key: str, val_fraction: float) -> str:
    """按哈希做确定性的 train/val 划分，重复运行不会变。"""
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_fraction else "train"


# ---------------------------------------------------------------- 合成
def _ssl_context():
    """edge-tts 固定用 certifi 的根证书；这里尊重 SSL_CERT_FILE，方便在公司代理 / 自签 CA 环境下使用。"""
    cafile = os.environ.get("SSL_CERT_FILE")
    if cafile and os.path.exists(cafile):
        return ssl.create_default_context(cafile=cafile)
    return None


async def _synth_one(job: SynthJob, mp3_path: Path, sem: asyncio.Semaphore, retries: int) -> None:
    import edge_tts

    delay = 1.0
    for attempt in range(1, retries + 1):
        try:
            async with sem:
                # 轻微错峰，避免并发瞬间打满
                await asyncio.sleep(random.uniform(0, 0.2))
                com = edge_tts.Communicate(job.text, job.voice, rate=job.rate, pitch=job.pitch)
                await com.save(str(mp3_path))
            if mp3_path.stat().st_size == 0:
                raise RuntimeError("empty mp3")
            return
        except Exception as e:  # noqa: BLE001 - 网络/服务端错误五花八门，统一重试
            if mp3_path.exists():
                mp3_path.unlink()
            if attempt == retries:
                raise
            log.warning(
                "tts failed (%s/%s) for %r [%s]: %s; retrying in %.0fs",
                attempt,
                retries,
                job.text,
                job.voice,
                e,
                delay,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)


async def _run_jobs(jobs: list[SynthJob], out_dir: Path, concurrency: int, retries: int) -> list[SynthJob]:
    import edge_tts.communicate as ec

    ctx = _ssl_context()
    if ctx is not None and hasattr(ec, "_SSL_CTX"):
        ec._SSL_CTX = ctx

    from tqdm import tqdm

    sem = asyncio.Semaphore(concurrency)
    failed: list[SynthJob] = []

    async def worker(job: SynthJob) -> None:
        mp3 = out_dir / f"{job.stem}.mp3"
        try:
            await _synth_one(job, mp3, sem, retries)
        except Exception as e:  # noqa: BLE001
            log.error("giving up on %r [%s]: %s", job.text, job.voice, e)
            failed.append(job)

    tasks = [asyncio.create_task(worker(j)) for j in jobs]
    for t in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="edge-tts"):
        await t
    return failed


# ---------------------------------------------------------------- 解码 / 裁剪
def decode_audio(path: Path) -> tuple[np.ndarray, int]:
    """读任意音频为 float32 单声道；优先 soundfile（libsndfile>=1.1 支持 mp3），不行就退到 ffmpeg。"""
    try:
        import soundfile as sf

        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return x.mean(axis=1), int(sr)
    except Exception as e:  # noqa: BLE001
        log.debug("soundfile failed for %s (%s); trying ffmpeg", path, e)
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy(), SAMPLE_RATE


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr_from, sr_to)
    return resample_poly(x, sr_to // g, sr_from // g).astype(np.float32)


def trim_silence(
    x: np.ndarray, sr: int, threshold_db: float = -45.0, pad_before: float = 0.03, pad_after: float = 0.06
) -> np.ndarray:
    """按 10 ms 帧能量裁掉首尾静音，前后各留一点余量。"""
    hop = int(sr * 0.01)
    if len(x) < hop:
        return x
    n = len(x) // hop
    frames = x[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames**2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    active = np.where(db > threshold_db)[0]
    if len(active) == 0:
        return x
    start = max(0, int(active[0] * hop - pad_before * sr))
    end = min(len(x), int((active[-1] + 1) * hop + pad_after * sr))
    return x[start:end]


def mp3_to_wav16k(mp3_path: Path, wav_path: Path, trim_db: float) -> float:
    """解码 → 16 kHz → 裁静音 → 峰值归一到 -3 dBFS → 16-bit WAV；返回时长（秒）。"""
    from scipy.io import wavfile

    x, sr = decode_audio(mp3_path)
    x = resample(x, sr, SAMPLE_RATE)
    x = trim_silence(x, SAMPLE_RATE, trim_db)
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if peak > 0:
        x = x * (0.708 / peak)
    wavfile.write(str(wav_path), SAMPLE_RATE, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    return len(x) / SAMPLE_RATE


# ---------------------------------------------------------------- 入口
def read_manifest(path: Path) -> list[ClipRecord]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [ClipRecord(**json.loads(line)) for line in f if line.strip()]


def write_manifest(path: Path, records: list[ClipRecord]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def run_synthesis(cfg: TrainingConfig) -> list[ClipRecord]:
    out_dir = cfg.clips_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = plan_jobs(cfg)

    existing = {r.wav: r for r in read_manifest(cfg.manifest_path)}
    todo = [j for j in jobs if f"{j.stem}.wav" not in existing or not (out_dir / f"{j.stem}.wav").exists()]
    log.info("%d clips already exist, %d to synthesize", len(jobs) - len(todo), len(todo))

    failed: list[SynthJob] = []
    if todo:
        failed = asyncio.run(_run_jobs(todo, out_dir, cfg.tts.concurrency, cfg.tts.retries))
        failed_keys = {j.key for j in failed}
        for job in todo:
            if job.key in failed_keys:
                continue
            mp3 = out_dir / f"{job.stem}.mp3"
            wav = out_dir / f"{job.stem}.wav"
            try:
                duration = mp3_to_wav16k(mp3, wav, cfg.tts.trim_db)
            except Exception as e:  # noqa: BLE001
                log.error("decode failed for %s: %s", mp3, e)
                continue
            finally:
                if mp3.exists():
                    mp3.unlink()
            existing[wav.name] = ClipRecord(
                text=job.text,
                voice=job.voice,
                rate=job.rate,
                pitch=job.pitch,
                label=job.label,
                lang=job.lang,
                wav=wav.name,
                duration=duration,
                split=split_for(job.key, cfg.data.val_fraction),
            )

    # 已有片段的文本 / 标签以当前计划为准（同一句话可能从 adversarial 挪到了 homophone）
    by_name = {f"{j.stem}.wav": j for j in jobs}
    records = [
        replace(r, text=j.text, voice=j.voice, rate=j.rate, pitch=j.pitch, label=j.label, lang=j.lang)
        for name, r in existing.items()
        if (j := by_name.get(name)) is not None
    ]
    write_manifest(cfg.manifest_path, records)
    n_pos = sum(r.is_positive for r in records)
    log.info(
        "manifest: %d clips (%d positive, %d negative), %d failed",
        len(records),
        n_pos,
        len(records) - n_pos,
        len(failed),
    )
    if failed:
        log.warning("re-run `synth` later to retry the %d failed clips", len(failed))
    return records
