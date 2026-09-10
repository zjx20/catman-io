"""silero VAD（v4 ONNX）——用 openWakeWord 随包下载的那份模型，onnxruntime 直接跑。

模型输入 ``input[1, N]``（float32，-1 到 1）、``sr``（int64 标量 16000）、``h/c``（[2,1,64] 状态），
输出 ``output/hn/cn``。图是变长的，但 silero 只在 512 到 1536 点上调过，所以一帧 1280 点切成
两段 640 点送两次、取最大值，状态跨帧连续。每段耗时不到 1 ms。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from catman_io.audio.frames import SAMPLE_RATE, to_float32


def default_model_path() -> Path:
    import openwakeword

    return Path(openwakeword.__file__).parent / "resources" / "models" / "silero_vad.onnx"


class SileroVAD:
    def __init__(self, model_path: str | os.PathLike | None = None, *, threads: int = 1, chunk: int = 640):
        import onnxruntime as ort

        path = Path(model_path) if model_path else default_model_path()
        if not path.exists():
            from catman_io.wakeword import ensure_base_models

            ensure_base_models(quiet=True)
        if not path.exists():
            raise FileNotFoundError(f"silero VAD model not found: {path} (run `catman-io setup`)")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = threads
        self._sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.chunk = chunk
        self.last = 0.0
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self.last = 0.0

    def process(self, frame: np.ndarray) -> float:
        """一帧 16 kHz 音频（int16 或 float）→ 这一帧含人声的概率（各段的最大值）。"""
        x = to_float32(frame)
        best = 0.0
        n = len(x)
        step = self.chunk if n >= self.chunk else n
        for i in range(0, n - step + 1, step):
            out, self._h, self._c = self._sess.run(
                None, {"input": x[None, i : i + step], "sr": self._sr, "h": self._h, "c": self._c}
            )
            best = max(best, float(out[0, 0]))
        self.last = best
        return best
