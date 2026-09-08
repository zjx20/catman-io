"""运行时配置（YAML）。示例见仓库根目录 config.example.yaml。"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AudioConfig:
    device: int | str | None = None  # sounddevice 设备编号或名字子串；None = 系统默认
    channel: int = 0  # 多声道设备取哪一路
    sample_rate: int | None = None  # None = 优先 16 kHz，不支持则自动重采样
    output_device: int | str | None = None


@dataclass
class WakeWordConfig:
    models: list[str] = field(default_factory=list)  # 空 = 用随包附带的模型
    threshold: float = 0.5
    patience: int = 1
    cooldown: float = 2.0
    vad_threshold: float = 0.0  # >0 时用 silero VAD 过滤非人声段的触发


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    wakeword: WakeWordConfig = field(default_factory=WakeWordConfig)

    @classmethod
    def load(cls, path: str | Path | None) -> Config:
        if path is None:
            return cls()
        with open(path, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config sections {sorted(unknown)}; allowed: {sorted(known)}")
        return cls(
            audio=AudioConfig(**(raw.get("audio") or {})),
            wakeword=WakeWordConfig(**(raw.get("wakeword") or {})),
        )
