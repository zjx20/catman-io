"""粤语语音合成（TTS）与念稿前的文本处理。

- :class:`EdgeTTS`（``edge.py``）：edge-tts 的 zh-HK 音色，联网、免费；按句合成、缓存固定短语。
- :func:`clean_for_speech`：把 markdown / 链接 / emoji 去掉、限长，并改写唤醒词避免自己把自己叫醒。
- :func:`split_sentences`：按句切开给 TTS，第一句到了就能开口。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from catman_io.config import Config

SENTENCE_END = "。！？!?；;\n"
SOFT_BREAK = "，,、：:"
WAKE_PHRASES = ("小貓人", "小猫人")
WAKE_REPLACEMENT = "貓人"

_CODE_BLOCK = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`([^`\n]*)`")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMOJI = re.compile(
    "[\U0001f000-\U0001faff☀-➿⬀-⯿️‍⌀-⏿←-⇿■-◿]"
)
_MD_LINE = re.compile(r"^\s*(?:#{1,6}\s*|>\s?|[-*+]\s+|\d+[.)]\s+)", re.M)
_MD_RULE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", re.M)
_EMPHASIS = re.compile(r"\*\*|__|(?<!\w)[*_](?=\S)|(?<=\S)[*_](?!\w)")


class Synthesizer(Protocol):
    def synthesize(self, text: str) -> np.ndarray:
        """文本 → 16 kHz int16 单声道 PCM。"""


def clean_for_speech(text: str, *, max_chars: int = 300) -> str:
    t = _CODE_BLOCK.sub("（略過一段代碼）", text)
    t = _INLINE_CODE.sub(r"\1", t)
    t = _LINK.sub(r"\1", t)
    t = _URL.sub("一個連結", t)
    t = _MD_RULE.sub("", t)
    t = _MD_LINE.sub("", t)
    t = _EMPHASIS.sub("", t)
    t = t.replace("|", "，")
    t = _EMOJI.sub("", t)
    for w in WAKE_PHRASES:
        t = t.replace(w, WAKE_REPLACEMENT)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s*\n\s*", "\n", t).strip()
    if len(t) > max_chars:
        cut = max((t.rfind(ch, 0, max_chars) for ch in SENTENCE_END), default=-1)
        t = t[: cut + 1] if cut > max_chars // 2 else t[:max_chars]
        t = t.rstrip() + "……"
    return t


def split_sentences(text: str, *, max_len: int = 60) -> list[str]:
    """按句号 / 问号 / 感叹号 / 换行切；太长的句子在逗号处再切。标点留在句尾。"""
    out: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        if ch in SENTENCE_END or (ch in SOFT_BREAK and len(buf) >= max_len):
            _flush(buf, out)
    _flush(buf, out)
    return out


def _flush(buf: list[str], out: list[str]) -> None:
    s = "".join(buf).strip()
    buf.clear()
    if s and re.search(r"\w", s):
        out.append(s)


def create_synthesizer(cfg: Config) -> Synthesizer | None:
    if cfg.tts.backend == "none":
        return None
    if cfg.tts.backend != "edge":
        raise ValueError(f"unknown tts backend {cfg.tts.backend!r}")
    from .edge import EdgeTTS

    return EdgeTTS(
        cfg.tts.voice,
        rate=cfg.tts.rate,
        pitch=cfg.tts.pitch,
        cache_dir=cfg.tts_cache_dir,
        timeout=cfg.tts.timeout,
    )


__all__ = ["Synthesizer", "clean_for_speech", "create_synthesizer", "split_sentences"]
