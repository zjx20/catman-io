"""edge-tts 粤语合成：按句请求，整句 MP3 解码成 16 kHz PCM，磁盘缓存固定短语。

edge-tts 输出固定是 24 kHz MP3，soundfile 只能解整段，所以做不到句内流式；按句合成、
一句到了就播，首包延迟等于第一句的合成时间。证书：edge-tts 写死用 certifi，这里尊重
``SSL_CERT_FILE``（公司代理 / 自签 CA 环境）。
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import ssl
import time
import wave
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from catman_io.audio.frames import SAMPLE_RATE, read_wav, resample, to_int16

log = logging.getLogger(__name__)

_ssl_done = False


def _setup_ssl() -> None:
    global _ssl_done
    if _ssl_done:
        return
    _ssl_done = True
    cafile = os.environ.get("SSL_CERT_FILE")
    if not cafile:
        return
    try:
        import edge_tts.communicate as ec

        if hasattr(ec, "_SSL_CTX"):
            ec._SSL_CTX = ssl.create_default_context(cafile=cafile)
    except Exception:  # noqa: BLE001
        log.exception("could not install SSL_CERT_FILE into edge-tts")


def decode_mp3(data: bytes) -> np.ndarray:
    """MP3 字节 → 16 kHz int16 单声道。"""
    import soundfile as sf

    x, sr = sf.read(io.BytesIO(data), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        x = resample(x, sr, SAMPLE_RATE)
    return to_int16(x)


class EdgeTTS:
    def __init__(
        self,
        voice: str = "zh-HK-HiuMaanNeural",
        *,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        cache_dir: str | Path | None = None,
        timeout: float = 10.0,
        retries: int = 2,
    ):
        self.voice, self.rate, self.pitch = voice, rate, pitch
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout = timeout
        self.retries = retries
        self.last_fetch_seconds = 0.0

    def _cache_path(self, text: str) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha1(f"{self.voice}|{self.rate}|{self.pitch}|{text}".encode()).hexdigest()
        return self.cache_dir / f"{key}.wav"

    def synthesize(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.int16)
        path = self._cache_path(text)
        if path is not None and path.exists():
            self.last_fetch_seconds = 0.0
            return read_wav(path)
        t0 = time.perf_counter()
        pcm = decode_mp3(self._fetch(text))
        self.last_fetch_seconds = time.perf_counter() - t0
        if path is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with wave.open(str(tmp), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes(pcm.tobytes())
            tmp.replace(path)
        return pcm

    def _fetch(self, text: str) -> bytes:
        try:
            import edge_tts
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("edge-tts not installed: pip install 'catman-io[tts]'") from e
        _setup_ssl()
        err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                c = edge_tts.Communicate(
                    text,
                    self.voice,
                    rate=self.rate,
                    pitch=self.pitch,
                    connect_timeout=int(self.timeout),
                    receive_timeout=int(self.timeout * 3),
                )
                data = b"".join(ch["data"] for ch in c.stream_sync() if ch["type"] == "audio")
                if data:
                    return data
                err = RuntimeError("edge-tts returned no audio")
            except Exception as e:  # noqa: BLE001
                err = e
                log.warning("edge-tts attempt %d failed: %s", attempt + 1, e)
        raise RuntimeError(f"edge-tts failed for {text[:20]!r}: {err}") from err

    def prewarm(self, texts: Iterable[str]) -> int:
        """把固定短语合成进缓存（启动时后台调用）。返回新合成的条数。"""
        n = 0
        for t in texts:
            path = self._cache_path(t)
            if path is not None and path.exists():
                continue
            try:
                self.synthesize(t)
                n += 1
            except Exception as e:  # noqa: BLE001
                log.warning("prewarm failed for %r: %s", t, e)
        return n
