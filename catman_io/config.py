"""运行时配置（YAML）。示例见仓库根目录 config.example.yaml。

所有 section 都有默认值：不写就是"用默认 / 关掉该功能"。密钥一律不进 YAML，
而是用 ``*_env`` 字段指向环境变量，用到时才读（见 :func:`secret`）。
"""

from __future__ import annotations

import dataclasses
import os
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


@dataclass
class AudioConfig:
    device: int | str | None = None  # sounddevice 设备编号或名字子串；None = 系统默认
    channel: int = 0  # 多声道设备取哪一路
    sample_rate: int | None = None  # None = 优先 16 kHz，不支持则自动重采样
    output_device: int | str | None = None  # 扬声器；None = 系统默认


@dataclass
class WakeWordConfig:
    models: list[str] = field(default_factory=list)  # 空 = 用随包附带的模型
    threshold: float = 0.5
    patience: int = 1
    cooldown: float = 2.0
    vad_threshold: float = 0.0  # >0 时用 silero VAD 过滤非人声段的触发


@dataclass
class VadConfig:
    """端点检测（唤醒之后判断一句话何时开始、何时说完）。"""

    start_threshold: float = 0.6  # 人声概率高于它算"开始说"
    end_threshold: float = 0.35  # 低于它算"停了"（滞回，避免抖动）
    start_frames: int = 2  # 连续多少帧高于 start_threshold 才算开始
    pre_roll_ms: int = 300  # 开始说之前多保留一点，免得吃掉第一个字
    trailing_silence_ms: int = 600  # 停多久算说完
    min_speech_ms: int = 200  # 短于这个的当噪声丢掉
    max_utterance_ms: int = 12000  # 最长一句，超过强制切断
    no_speech_timeout_ms: int = 5000  # 唤醒后多久没人说话就放弃
    tail_pad_ms: int = 200  # 切句时在最后一帧人声后多留一点
    followup_start_threshold: float = 0.8  # 免唤醒跟进时的更严格阈值
    followup_start_frames: int = 3
    followup_guard_ms: int = 400  # 播完之后先不听一会，免得把自己的尾音当人声
    threads: int = 1


@dataclass
class AsrConfig:
    backend: str = "sensevoice"  # sensevoice | wenet_yue | none
    model: str = ""  # asr/models.py 里的模型名；空 = 该 backend 的默认模型
    model_dir: str = ""  # 空 = <data_dir>/models/asr
    language: str = "yue"
    threads: int = 2
    use_itn: bool = True  # 逆文本正则化（数字、标点），SenseVoice 支持


@dataclass
class LLMConfig:
    """OpenAI 兼容的 chat completions 端点（Gemini 兼容端点、OpenAI、DeepSeek、本地 vLLM 都行）。"""

    enabled: bool = False
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    model: str = ""
    api_key_env: str = "CATMAN_IO_LLM_API_KEY"
    timeout: float = 4.0
    temperature: float = 0.0
    max_tokens: int = 256


@dataclass
class IntentLLMConfig(LLMConfig):
    tool_choice: str = "auto"  # auto | required（有的端点不支持 required）
    answer_chat: bool = False  # 模型直接回了文本就当回复念出来，不再进大脑
    shadow_rate: float = 0.0  # 规则命中后抽样再问一次 LLM 做对照（后台，不影响延迟）


@dataclass
class IntentContextConfig:
    rooms: list[str] = field(default_factory=list)  # 房间名，给 room 槽位和 LLM 提示
    home: str = ""  # 一句话描述这个家 / 设备，放进 LLM 提示
    extra: str = ""  # 额外提示词


@dataclass
class IntentConfig:
    rules_dir: str = ""  # 现场规则目录（catman 改的那份）；空 = <data_dir>/intent/rules
    cases_file: str = ""  # 回归用例；空 = <data_dir>/intent/cases.jsonl
    llm: IntentLLMConfig = field(default_factory=IntentLLMConfig)
    context: IntentContextConfig = field(default_factory=IntentContextConfig)


@dataclass
class BrainLLMConfig(LLMConfig):
    timeout: float = 30.0
    max_tokens: int = 400
    system_prompt: str = ""  # 空 = 内置的"简短粤语口语"提示
    memory_minutes: float = 10.0  # 短期记忆：最近几分钟的对话带进上下文
    memory_turns: int = 6


@dataclass
class CatmanConfig:
    base_url: str = ""  # 例如 http://192.168.1.1:8787；空 = 不接 catman
    token_env: str = "CATMAN_ADMIN_TOKEN"
    timeout: float = 5.0


@dataclass
class BrainConfig:
    llm: BrainLLMConfig = field(default_factory=BrainLLMConfig)
    catman: CatmanConfig = field(default_factory=CatmanConfig)


@dataclass
class TtsConfig:
    backend: str = "edge"  # edge | none
    voice: str = "zh-HK-HiuMaanNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"
    cache_dir: str = ""  # 空 = <data_dir>/cache/tts
    timeout: float = 10.0


@dataclass
class SpeakerConfig:
    volume: float = 0.8  # 0 到 1 的增益（volume.* 意图改的就是它）
    blocksize: int = 320  # 20 ms
    latency: str = "low"


@dataclass
class DialogConfig:
    followup_seconds: float = 6.0  # 回答完后几秒内不用叫唤醒词；0 = 关
    followup_question_seconds: float = 10.0  # 动作明确要追问时的窗口
    earcons: bool = True
    thinking_beep_after: float = 1.5  # 思考超过几秒还没开口就提示一声
    turn_timeout: float = 120.0  # 思考多久没结果就放弃这一轮
    reprompt_on_empty: bool = True  # 没听清时再听一次
    slow_threshold: float = 3.0  # 说完到首包音频超过几秒算慢（打 slow 标记）
    wake_context_seconds: float = 1.5  # 唤醒时往前保存多少音频（做训练样本）


@dataclass
class JournalConfig:
    enabled: bool = True
    save_audio: bool = True
    keep_days: int = 30


@dataclass
class ApiConfig:
    enabled: bool = True
    host: str = "127.0.0.1"  # 给 catman（Docker 里）访问要改成 0.0.0.0
    port: int = 8766
    token_env: str = "CATMAN_IO_API_TOKEN"
    token_file: str = ""  # 空 = <data_dir>/api_token（首次运行自动生成）


@dataclass
class WeatherConfig:
    latitude: float | None = None
    longitude: float | None = None
    name: str = ""  # 报天气时念的地名


@dataclass
class ActionsConfig:
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    http_timeout: float = 5.0


@dataclass
class Config:
    data_dir: str = "data"  # 日志、现场规则、缓存、模型都放这里
    audio: AudioConfig = field(default_factory=AudioConfig)
    wakeword: WakeWordConfig = field(default_factory=WakeWordConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    intent: IntentConfig = field(default_factory=IntentConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    dialog: DialogConfig = field(default_factory=DialogConfig)
    journal: JournalConfig = field(default_factory=JournalConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    actions: ActionsConfig = field(default_factory=ActionsConfig)

    @classmethod
    def load(cls, path: str | Path | None) -> Config:
        if path is None:
            return cls()
        with open(path, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return from_dict(cls, raw, where="<root>")

    # ---- 派生路径（都从 data_dir 出发，除非单独配置了）----
    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def rules_dir(self) -> Path:
        return Path(self.intent.rules_dir) if self.intent.rules_dir else self.data_path / "intent" / "rules"

    @property
    def cases_path(self) -> Path:
        if self.intent.cases_file:
            return Path(self.intent.cases_file)
        return self.data_path / "intent" / "cases.jsonl"

    @property
    def asr_model_dir(self) -> Path:
        return Path(self.asr.model_dir) if self.asr.model_dir else self.data_path / "models" / "asr"

    @property
    def tts_cache_dir(self) -> Path:
        return Path(self.tts.cache_dir) if self.tts.cache_dir else self.data_path / "cache" / "tts"

    @property
    def journal_dir(self) -> Path:
        return self.data_path / "journal"

    @property
    def api_token_path(self) -> Path:
        return Path(self.api.token_file) if self.api.token_file else self.data_path / "api_token"


def secret(env_name: str, *, required: bool = True, what: str = "") -> str | None:
    """从环境变量取密钥。缺了就报一句能看懂的错，而不是在 HTTP 401 里猜。"""
    value = os.environ.get(env_name)
    if value:
        return value
    if required:
        raise ConfigError(f"environment variable {env_name} is not set{f' ({what})' if what else ''}")
    return None


# ---- 递归解析：嵌套 dataclass、未知键报错、标量类型检查 ----

_NoneType = type(None)


def from_dict(cls: type, data: Any, where: str):
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(f"{where}: unknown keys {unknown}; allowed: {sorted(fields)}")
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        hint = hints[name]
        sub_where = f"{where}.{name}"
        if dataclasses.is_dataclass(hint) and isinstance(hint, type):
            kwargs[name] = from_dict(hint, value, where=sub_where)
        else:
            kwargs[name] = _check_value(value, hint, sub_where)
    return cls(**kwargs)


def _check_value(value: Any, hint: Any, where: str) -> Any:
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        members = typing.get_args(hint)
    else:
        members = (hint,)
    for m in members:
        if m is _NoneType and value is None:
            return None
        if m is bool and isinstance(value, bool):
            return value
        if m is int and isinstance(value, int) and not isinstance(value, bool):
            return value
        if m is float and isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if m is str and isinstance(value, str):
            return value
        if m is Any:
            return value
        m_origin = typing.get_origin(m)
        if (m is list or m_origin is list) and isinstance(value, list):
            return list(value)
        if (m is dict or m_origin is dict) and isinstance(value, dict):
            return dict(value)
    expected = " | ".join(getattr(m, "__name__", str(m)) for m in members)
    raise ConfigError(f"{where}: expected {expected}, got {type(value).__name__} ({value!r})")
