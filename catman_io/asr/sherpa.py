"""sherpa-onnx 离线识别：SenseVoice（language=yue）或 WenetSpeech-Yue CTC。整句进、文本出。"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from catman_io.audio.frames import SAMPLE_RATE, to_float32

from . import Transcript


class SherpaRecognizer:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        kind: str = "sensevoice",
        language: str = "yue",
        threads: int = 2,
        use_itn: bool = True,
        model_file: str = "model.int8.onnx",
        tokens_file: str = "tokens.txt",
    ):
        try:
            import sherpa_onnx
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("sherpa-onnx not installed: pip install 'catman-io[asr]'") from e
        d = Path(model_dir)
        model, tokens = str(d / model_file), str(d / tokens_file)
        self.kind = kind
        self.language = language
        if kind == "sensevoice":
            self._rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=model, tokens=tokens, num_threads=threads, language=language, use_itn=use_itn
            )
        elif kind == "wenet_ctc":
            self._rec = sherpa_onnx.OfflineRecognizer.from_wenet_ctc(
                model=model, tokens=tokens, num_threads=threads
            )
        else:
            raise ValueError(f"unknown sherpa model kind {kind!r}")

    def transcribe(self, audio: np.ndarray) -> Transcript:
        x = to_float32(audio)
        t0 = time.perf_counter()
        s = self._rec.create_stream()
        s.accept_waveform(SAMPLE_RATE, x)
        self._rec.decode_stream(s)
        r = s.result
        text = (r.text or "").strip()
        lang = (getattr(r, "lang", "") or "").strip("<|>") or self.language
        return Transcript(
            text=text,
            raw=text,
            language=lang,
            elapsed=time.perf_counter() - t0,
            duration=len(x) / SAMPLE_RATE,
        )

    def warmup(self) -> None:
        self.transcribe(np.zeros(SAMPLE_RATE // 2, dtype=np.int16))
