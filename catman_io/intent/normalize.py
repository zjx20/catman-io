"""识别文本归一化：简体 → 香港繁体、全角 → 半角、小写、去标点空白、常见口语异写统一。

规则里的模式与例句也走同一套（除了不去标点，正则需要它），所以 catman 用简体写规则也能匹配。
"""

from __future__ import annotations

import re
import unicodedata

import zhconv

_PUNCT = re.compile(r"[^\w%]|_")
_TAG = re.compile(r"<\|[^|]*\|>")
_WAKE = re.compile(r"^(?:小貓人|貓人)+")
VARIANTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"宜家|依家|依加|而加|現在"), "而家"),
    (re.compile(r"噉樣|咁樣|噉"), "咁"),
    (re.compile(r"聽朝"), "聽日朝早"),
    (re.compile(r"(?<=[一-鿿])d(?![a-z])"), "啲"),
    (re.compile(r"唔駛"), "唔使"),
    (re.compile(r"係咪"), "係唔係"),
)


def to_hk(text: str) -> str:
    """转香港繁体 + 半角 + 小写，不动标点。规则模式用这个。"""
    return zhconv.convert(unicodedata.normalize("NFKC", text), "zh-hk").lower()


def normalize(text: str, *, strip_wake: bool = True) -> str:
    """识别结果用这个：再去掉标点空白与识别器标签，统一异写，可选去掉开头的唤醒词。"""
    t = to_hk(_TAG.sub("", text))
    t = _PUNCT.sub("", t)
    for pat, rep in VARIANTS:
        t = pat.sub(rep, t)
    if strip_wake:
        t = _WAKE.sub("", t)
    return t
