"""扬声器：一条常开的输出流 + 写线程 + 可打断的播放队列。

- ``Output`` 是可注入的底层输出：真声卡（sounddevice）、写 WAV（``run --out``）、内存列表（测试）。
- ``say(pcm, gen)`` 把音频按 20 ms 切块排队；``gen`` 是回合代号，过期回合的音频会被丢掉，
  这样被打断的回合即使 worker 线程还在合成，也插不进来。提示音用 ``gen=None`` 总是接受。
- ``stop()`` 清空队列，正在写的那一块（20 ms）写完就停，加上设备延迟大约几十毫秒。
"""

from __future__ import annotations

import logging
import threading
import time
import wave
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

from catman_io.audio.frames import SAMPLE_RATE, to_int16

log = logging.getLogger(__name__)


class Output(Protocol):
    keepalive: bool  # 空闲时是否持续写静音（真声卡要，文件不要）

    def write(self, pcm: np.ndarray) -> None:
        """阻塞写一块 16 kHz int16 单声道。"""

    def close(self) -> None: ...


class SoundDeviceOutput:
    keepalive = True

    def __init__(self, device: int | str | None = None, *, blocksize: int = 320, latency: str = "low"):
        from catman_io.audio.capture import _sounddevice

        sd = _sounddevice()
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            device=device,
            blocksize=blocksize,
            latency=latency,
        )
        self._stream.start()
        log.info("speaker: device=%r latency=%.0fms", device, self._stream.latency * 1000)

    def write(self, pcm: np.ndarray) -> None:
        self._stream.write(pcm)

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


class WavOutput:
    keepalive = False

    def __init__(self, path: str | Path):
        self._w = wave.open(str(path), "wb")
        self._w.setnchannels(1)
        self._w.setsampwidth(2)
        self._w.setframerate(SAMPLE_RATE)

    def write(self, pcm: np.ndarray) -> None:
        self._w.writeframes(to_int16(pcm).tobytes())

    def close(self) -> None:
        self._w.close()


class ListOutput:
    keepalive = False

    def __init__(self, delay: float = 0.0):
        self.chunks: list[np.ndarray] = []
        self.delay = delay  # 模拟真声卡的阻塞写

    def write(self, pcm: np.ndarray) -> None:
        self.chunks.append(np.array(pcm, copy=True))
        if self.delay:
            time.sleep(self.delay)

    def close(self) -> None:
        pass

    @property
    def audio(self) -> np.ndarray:
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, dtype=np.int16)


class Speaker:
    def __init__(
        self,
        output: Output,
        *,
        volume: float = 0.8,
        blocksize: int = 320,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.output = output
        self.volume = float(volume)
        self.blocksize = blocksize
        self.clock = clock
        self.current_gen: int | None = None
        self.first_audio: dict[int, float] = {}
        self._q: deque[tuple[int | None, np.ndarray]] = deque()
        self._cv = threading.Condition()
        self._busy = False
        self._closing = False
        self._thread = threading.Thread(target=self._run, name="speaker", daemon=True)
        self._thread.start()

    # ---- 主线程 / worker 线程调用 ----

    def set_gen(self, gen: int | None) -> None:
        self.current_gen = gen

    def say(self, pcm: np.ndarray, gen: int | None = None) -> bool:
        """排队播放；gen 不是当前回合就丢掉（返回 False）。"""
        if gen is not None and gen != self.current_gen:
            return False
        pcm = to_int16(pcm)
        with self._cv:
            for i in range(0, len(pcm), self.blocksize):
                self._q.append((gen, pcm[i : i + self.blocksize]))
            self._cv.notify_all()
        return True

    def stop(self) -> None:
        with self._cv:
            self._q.clear()
            self._cv.notify_all()

    @property
    def is_busy(self) -> bool:
        with self._cv:
            return self._busy or bool(self._q)

    def wait(self, timeout: float | None = None) -> bool:
        """等队列播空。返回是否真的播空了（False = 超时）。"""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            while self._q or self._busy:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._cv.wait(timeout=remaining if remaining is not None else 0.2)
        return True

    def close(self) -> None:
        with self._cv:
            self._closing = True
            self._q.clear()
            self._cv.notify_all()
        self._thread.join(timeout=2.0)
        self.output.close()

    # ---- 写线程 ----

    def _run(self) -> None:
        idle = np.zeros(self.blocksize, dtype=np.int16)
        while True:
            with self._cv:
                while not self._q and not self._closing:
                    if self.output.keepalive:
                        break
                    self._cv.wait(timeout=0.2)
                if self._closing:
                    return
                item = self._q.popleft() if self._q else None
                self._busy = item is not None
            if item is None:
                try:
                    self.output.write(idle)
                except Exception:  # noqa: BLE001
                    log.exception("speaker keepalive write failed")
                    time.sleep(0.02)
                continue
            gen, block = item
            if gen is not None and gen not in self.first_audio:
                self.first_audio[gen] = self.clock()
            if self.volume != 1.0:
                block = np.clip(block.astype(np.float32) * self.volume, -32768, 32767).astype(np.int16)
            try:
                self.output.write(block)
            except Exception:  # noqa: BLE001
                log.exception("speaker write failed")
            with self._cv:
                if not self._q:
                    self._busy = False
                    self._cv.notify_all()
