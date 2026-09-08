"""音频流推送——规划中。

唤醒后把 80 ms 帧持续推给后端（例如支持实时语音的 live 模型），并接收回传的音频/文本。
接口按"开始会话 → 逐帧发送 → 结束"设计，具体传输（WebSocket / gRPC）由实现决定。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class AudioStreamer(Protocol):
    def start(self) -> None: ...

    def send(self, frame: np.ndarray) -> None:
        """推送一帧 16 kHz int16。"""

    def stop(self) -> None: ...
