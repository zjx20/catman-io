"""粤语语音转文字（ASR）。

默认走 sherpa-onnx 的离线模型（SenseVoice 或 WenetSpeech-Yue，见 ``models.py``），CPU 上一句话
零点几秒。识别结果是简体（粤语字保留），转繁与匹配在 ``catman_io.intent.normalize`` 做。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from catman_io.config import Config


@dataclass
class Transcript:
    text: str
    raw: str = ""  # 识别器原样输出（未归一化）
    language: str = "yue"
    confidence: float | None = None
    elapsed: float = 0.0  # 识别耗时（秒）
    duration: float = 0.0  # 音频时长（秒）

    @property
    def rtf(self) -> float:
        return self.elapsed / self.duration if self.duration else 0.0


class SpeechRecognizer(Protocol):
    def transcribe(self, audio: np.ndarray) -> Transcript:
        """整段识别：16 kHz int16 单声道 → 文本。"""


def create_recognizer(cfg: Config) -> SpeechRecognizer:
    """按配置建识别器。模型没下载时报错提示 ``catman-io setup --asr``，不会静默下载几百 MB。"""
    from .models import is_installed, model_path, resolve_model

    a = cfg.asr
    if a.backend == "none":
        raise RuntimeError("asr.backend is 'none'")
    model = resolve_model(a.backend, a.model)
    root = cfg.asr_model_dir
    if not is_installed(root, model):
        raise RuntimeError(f"ASR model {model.name} not found under {root}: run `catman-io setup --asr`")
    from .sherpa import SherpaRecognizer

    return SherpaRecognizer(
        model_path(root, model),
        kind=model.kind,
        language=a.language,
        threads=a.threads,
        use_itn=a.use_itn,
        model_file=model.model_file,
        tokens_file=model.tokens_file,
    )


__all__ = ["SpeechRecognizer", "Transcript", "create_recognizer"]
