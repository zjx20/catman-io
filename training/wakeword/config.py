"""训练配置：YAML → 嵌套 dataclass，未知字段直接报错，避免写错键名后静默用默认值。"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import phrases


@dataclass
class TTSConfig:
    # 粤语音色（edge-tts 目前只有这三个 zh-HK 音色，多样性主要靠语速/音高和后续增强补）
    voices: list[str] = field(
        default_factory=lambda: ["zh-HK-HiuGaaiNeural", "zh-HK-HiuMaanNeural", "zh-HK-WanLungNeural"]
    )
    # 正样本展开：每个文本 × 音色 × 语速 × 音高
    rates: list[str] = field(default_factory=lambda: ["-20%", "-10%", "+0%", "+10%", "+20%"])
    pitches: list[str] = field(default_factory=lambda: ["-30Hz", "-15Hz", "+0Hz", "+15Hz", "+30Hz"])
    # 负样本文本多，展开少一点
    negative_rates: list[str] = field(default_factory=lambda: ["-10%", "+0%", "+15%"])
    negative_pitches: list[str] = field(default_factory=lambda: ["+0Hz"])
    # 其他语言的负样本音色（键是语言，值是音色列表）
    other_language_voices: dict[str, list[str]] = field(
        default_factory=lambda: {
            "zh-CN": ["zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural"],
            "en-US": ["en-US-AriaNeural", "en-US-GuyNeural"],
        }
    )
    concurrency: int = 4
    retries: int = 5
    # 合成后按能量裁掉首尾静音的阈值（dBFS）
    trim_db: float = -45.0
    # 0 = 全部组合；>0 = 随机抽这么多个（快速试跑用）
    max_positive_clips: int = 0
    max_negative_clips: int = 0


@dataclass
class DataConfig:
    # 留空则用 phrases.py 里的默认清单
    positive_phrases: list[str] = field(default_factory=list)
    adversarial_phrases: list[str] = field(default_factory=list)
    general_phrases: list[str] = field(default_factory=list)
    mandarin_phrases: list[str] = field(default_factory=list)
    english_phrases: list[str] = field(default_factory=list)
    # 真人录音目录（16 kHz 单声道 wav；其他采样率会自动重采样）。强烈建议录几十条正样本。
    extra_positive_dirs: list[str] = field(default_factory=list)
    extra_negative_dirs: list[str] = field(default_factory=list)
    val_fraction: float = 0.15
    # openWakeWord 预计算的通用负样本特征（ACAV100M，共 2000 小时 / 17 GB）只下载前 N 小时
    precomputed_negative_hours: float = 20.0
    # openWakeWord 的 11 小时验证集特征（约 180 MB），用于估计每小时误唤醒次数
    fp_validation: bool = True

    def resolved_positive_phrases(self) -> list[str]:
        return self.positive_phrases or list(phrases.POSITIVE_VARIANTS)

    def resolved_adversarial_phrases(self) -> list[str]:
        return self.adversarial_phrases or list(phrases.ADVERSARIAL)

    def resolved_general_phrases(self) -> list[str]:
        return self.general_phrases or list(phrases.GENERAL_CANTONESE)

    def resolved_other_language_phrases(self) -> dict[str, list[str]]:
        return {
            "zh-CN": self.mandarin_phrases or list(phrases.GENERAL_MANDARIN),
            "en-US": self.english_phrases or list(phrases.GENERAL_ENGLISH),
        }


@dataclass
class AugmentConfig:
    clip_seconds: float = 2.0
    rounds_positive: int = 4
    rounds_negative: int = 2
    # 环境噪声目录（任意 wav/flac/mp3；推荐用目标设备在真实环境录的房间噪声、电视声）
    background_dirs: list[str] = field(default_factory=list)
    # 房间冲激响应目录；留空且 download_rirs=true 时自动下载 MIT 的 270 条 RIR
    rir_dirs: list[str] = field(default_factory=list)
    download_rirs: bool = True
    speed_factors: list[float] = field(default_factory=lambda: [0.9, 1.0, 1.1])
    p_speed: float = 0.5
    snr_db: list[float] = field(default_factory=lambda: [-5.0, 20.0])
    p_background: float = 0.75
    p_colored_noise: float = 0.3
    colored_snr_db: list[float] = field(default_factory=lambda: [5.0, 30.0])
    p_babble: float = 0.2
    babble_snr_db: list[float] = field(default_factory=lambda: [0.0, 15.0])
    p_rir: float = 0.5
    p_bandstop: float = 0.2
    p_lowpass: float = 0.15
    p_distortion: float = 0.15
    peak_range: list[float] = field(default_factory=lambda: [0.05, 1.0])
    # 唤醒词结束点距离窗口末尾的随机抖动（秒）
    end_jitter: float = 0.2
    batch_size: int = 64


@dataclass
class TrainConfig:
    layer_size: int = 32
    n_blocks: int = 1
    dropout: float = 0.1
    steps: int = 20000
    batch_positive: int = 50
    batch_negative: int = 50
    batch_precomputed: int = 1024
    lr: float = 1e-4
    max_negative_weight: float = 1500.0
    target_fp_per_hour: float = 0.2
    seed: int = 0
    threads: int = 0  # 0 = 自动


@dataclass
class TrainingConfig:
    model_name: str = "siu_maau_jan"
    wake_phrase: str = phrases.WAKE_PHRASE
    workdir: str = "training/wakeword/work/siu_maau_jan"
    tts: TTSConfig = field(default_factory=TTSConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ---- 派生路径 ----
    @property
    def work(self) -> Path:
        return Path(self.workdir)

    @property
    def clips_dir(self) -> Path:
        return self.work / "clips"

    @property
    def resources_dir(self) -> Path:
        return self.work / "resources"

    @property
    def features_dir(self) -> Path:
        return self.work / "features"

    @property
    def export_dir(self) -> Path:
        return self.work / "export"

    @property
    def manifest_path(self) -> Path:
        return self.clips_dir / "manifest.jsonl"

    @classmethod
    def load(cls, path: str | Path) -> TrainingConfig:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return _from_dict(cls, raw, where="<root>")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _from_dict(cls: type, data: dict[str, Any], where: str):
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(data).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ValueError(f"{where}: unknown keys {unknown}; allowed: {sorted(fields)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        f = fields[name]
        if dataclasses.is_dataclass(f.type if isinstance(f.type, type) else _resolve(f.type)):
            sub = f.type if isinstance(f.type, type) else _resolve(f.type)
            kwargs[name] = _from_dict(sub, value or {}, where=f"{where}.{name}")
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _resolve(annotation: Any) -> Any:
    """`from __future__ import annotations` 让类型变成字符串；在本模块的命名空间里解析回类。"""
    if isinstance(annotation, str):
        return globals().get(annotation, annotation)
    return annotation
