"""流式唤醒词检测器。

每喂一帧（80 ms、1280 个 int16 采样点）返回这一帧触发的检测结果（通常为空）。
底层是 openWakeWord：共享的 melspectrogram + 语音嵌入模型（约 2.4 MB，首次运行自动下载）
产出每 80 ms 一帧的 96 维特征，再由我们训练的小分类器在最近 16 帧（1.28 s）上判断。

触发逻辑（都可配）：
- threshold：分数阈值；
- patience：连续多少帧高于阈值才算触发，抗偶发尖峰；
- cooldown：触发后多少秒内不再触发，避免同一句话连续触发。
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from catman_io.audio.frames import FRAME_SECONDS, to_int16

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"


@dataclass(frozen=True)
class Detection:
    """一次唤醒。stream_time 是从检测器启动起累计的音频时间（秒），与实际时钟无关。"""

    model: str
    score: float
    stream_time: float
    wall_time: float


def bundled_models() -> list[Path]:
    """随包附带的唤醒词模型（ONNX）。"""
    return sorted(MODELS_DIR.glob("*.onnx"))


def default_model_path() -> Path:
    models = bundled_models()
    if not models:
        raise FileNotFoundError(
            f"no wake-word model found in {MODELS_DIR}; fetch one with "
            "`python scripts/wakeword_model.py pull` (models live in the models/wakeword/<version> "
            "branches), train one with "
            "`python -m training.wakeword all ...`, or pass model_paths explicitly"
        )
    return models[0]


def ensure_base_models(quiet: bool = False) -> None:
    """确保 openWakeWord 的特征模型（melspectrogram / embedding / silero VAD）已下载。"""
    import openwakeword
    from openwakeword.utils import download_file

    target = Path(openwakeword.__file__).parent / "resources" / "models"
    target.mkdir(parents=True, exist_ok=True)
    urls = []
    for m in openwakeword.FEATURE_MODELS.values():
        urls += [m["download_url"], m["download_url"].replace(".tflite", ".onnx")]
    urls += [m["download_url"] for m in openwakeword.VAD_MODELS.values()]
    for url in urls:
        if not (target / url.rsplit("/", 1)[-1]).exists():
            if not quiet:
                log.info("downloading %s", url.rsplit("/", 1)[-1])
            download_file(url, str(target))


class WakeWordDetector:
    def __init__(
        self,
        model_paths: Sequence[str | os.PathLike] | None = None,
        *,
        threshold: float = 0.5,
        patience: int = 1,
        cooldown: float = 2.0,
        vad_threshold: float = 0.0,
        inference_framework: str = "onnx",
        ncpu: int = 1,
    ):
        ensure_base_models(quiet=True)
        from openwakeword import Model

        paths = [str(p) for p in (model_paths or [default_model_path()])]
        for p in paths:
            if not os.path.exists(p):
                raise FileNotFoundError(p)
        self._oww = Model(
            wakeword_models=paths,
            inference_framework=inference_framework,
            vad_threshold=vad_threshold,
            ncpu=ncpu,
        )
        self.names: list[str] = list(self._oww.models.keys())
        self.threshold = threshold
        self.patience = max(1, patience)
        self.cooldown = cooldown
        self._history: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.patience))
        self._last_fired: dict[str, float] = {}
        self._frames = 0
        self.last_scores: dict[str, float] = dict.fromkeys(self.names, 0.0)

    @property
    def stream_time(self) -> float:
        return self._frames * FRAME_SECONDS

    def reset(self) -> None:
        self._oww.reset()
        self._history.clear()
        self._last_fired.clear()
        self._frames = 0

    def configure(
        self, threshold: float | None = None, patience: int | None = None, cooldown: float | None = None
    ) -> None:
        """运行中调整触发参数（网页 demo 的滑块用）。"""
        if threshold is not None:
            self.threshold = float(threshold)
        if cooldown is not None:
            self.cooldown = float(cooldown)
        if patience is not None and max(1, int(patience)) != self.patience:
            self.patience = max(1, int(patience))
            self._history = defaultdict(lambda: deque(maxlen=self.patience))

    def process(self, frame: np.ndarray) -> list[Detection]:
        """喂入一帧（1280 个采样点；float 会自动转 int16），返回本帧触发的检测。"""
        frame = to_int16(np.asarray(frame).reshape(-1))
        scores = self._oww.predict(frame)
        self._frames += 1
        self.last_scores = {k: float(v) for k, v in scores.items()}
        now = self.stream_time
        fired: list[Detection] = []
        for name, score in self.last_scores.items():
            hist = self._history[name]
            hist.append(score)
            if len(hist) < self.patience or min(hist) < self.threshold:
                continue
            if now - self._last_fired.get(name, -1e9) < self.cooldown:
                continue
            self._last_fired[name] = now
            fired.append(Detection(model=name, score=score, stream_time=now, wall_time=time.time()))
        return fired

    def process_audio(self, audio: np.ndarray) -> list[Detection]:
        """离线跑一整段音频（按 80 ms 分帧），用于测试录音。"""
        from catman_io.audio.frames import iter_frames

        out: list[Detection] = []
        for frame in iter_frames(to_int16(audio)):
            out += self.process(frame)
        return out
