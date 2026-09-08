"""麦克风采集：任何 ALSA / PortAudio 能看到的输入设备 → 16 kHz 单声道 int16 的 80 ms 帧。

设备通常已经做了远场拾音和降噪，这里只负责取到干净的单声道数据并统一格式；
多声道设备可以用 channel 选一路。需要 `pip install catman-io[audio]`（sounddevice / PortAudio）。
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Iterator

import numpy as np

from .frames import FRAME_SAMPLES, SAMPLE_RATE, resample

log = logging.getLogger(__name__)


def _sounddevice():
    try:
        import sounddevice as sd
    except OSError as e:  # PortAudio 动态库不在
        raise RuntimeError(
            "PortAudio not found: install it (Debian/Ubuntu: apt install libportaudio2) and "
            "`pip install catman-io[audio]`"
        ) from e
    except ImportError as e:
        raise RuntimeError("sounddevice not installed: `pip install catman-io[audio]`") from e
    return sd


def list_devices() -> str:
    """返回 sounddevice 的设备列表文本（`catman-io devices`）。"""
    return str(_sounddevice().query_devices())


class MicCapture:
    """后台线程持续采集，`frames()` 按帧产出；队列满了丢最旧的帧并计数，绝不阻塞音频回调。"""

    def __init__(
        self,
        device: int | str | None = None,
        channels: int | None = None,
        channel: int = 0,
        sample_rate: int | None = None,
        frame_samples: int = FRAME_SAMPLES,
        queue_frames: int = 128,
    ):
        self.device = device
        self.channels = channels
        self.channel = channel
        self.sample_rate = sample_rate  # None = 优先 16 kHz，不支持则用设备默认采样率再重采样
        self.frame_samples = frame_samples
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_frames)
        self._stream = None
        self._buffer = np.zeros(0, dtype=np.int16)
        self._lock = threading.Lock()
        self.dropped = 0
        self.native_rate = SAMPLE_RATE

    # ------------------------------------------------------------ 生命周期
    def start(self) -> MicCapture:
        sd = _sounddevice()
        info = sd.query_devices(self.device, "input")
        channels = self.channels or max(1, min(int(info["max_input_channels"]), self.channel + 1))
        rate = self._pick_rate(sd, channels, info)
        self.native_rate = rate
        block = int(round(self.frame_samples * rate / SAMPLE_RATE))
        log.info(
            "capture: device=%r channels=%d rate=%d block=%d channel=%d",
            info["name"],
            channels,
            rate,
            block,
            self.channel,
        )
        self._stream = sd.InputStream(
            device=self.device,
            channels=channels,
            samplerate=rate,
            dtype="int16",
            blocksize=block,
            callback=self._callback,
        )
        self._stream.start()
        return self

    def _pick_rate(self, sd, channels: int, info) -> int:
        if self.sample_rate:
            return int(self.sample_rate)
        try:
            sd.check_input_settings(
                device=self.device, channels=channels, samplerate=SAMPLE_RATE, dtype="int16"
            )
            return SAMPLE_RATE
        except Exception as e:  # noqa: BLE001
            rate = int(info["default_samplerate"])
            log.info(
                "device does not accept %d Hz (%s); capturing at %d Hz and resampling", SAMPLE_RATE, e, rate
            )
            return rate

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self) -> MicCapture:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------ 数据
    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ARG002
        if status:
            log.debug("capture status: %s", status)
        mono = np.asarray(indata)[:, min(self.channel, indata.shape[1] - 1)].copy()
        if self.native_rate != SAMPLE_RATE:
            mono = resample(mono, self.native_rate, SAMPLE_RATE)
        with self._lock:
            self._buffer = np.concatenate([self._buffer, mono])
            while len(self._buffer) >= self.frame_samples:
                frame, self._buffer = self._buffer[: self.frame_samples], self._buffer[self.frame_samples :]
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    self.dropped += 1
                    try:
                        self._queue.get_nowait()
                        self._queue.put_nowait(frame)
                    except (queue.Empty, queue.Full):
                        pass

    def read(self, timeout: float | None = None) -> np.ndarray:
        return self._queue.get(timeout=timeout)

    def frames(self) -> Iterator[np.ndarray]:
        while self._stream is not None:
            try:
                yield self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
