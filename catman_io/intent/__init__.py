"""意图识别——规划中。

把识别出来的粤语文本归类：本地就能处理的简单指令（音量、停止、取消）走快速通道，
其余交给后端（catman）的大模型。也可能直接把音频推给后端的 live 模型，由它一并完成。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Intent:
    name: str
    slots: dict[str, str] = field(default_factory=dict)
    confidence: float = 1.0


class IntentRecognizer(Protocol):
    def recognize(self, text: str) -> Intent: ...
