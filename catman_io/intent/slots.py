"""槽位类型：模式片段 + 解析器，把粤语口语里的数字、时长、时间、日期、百分比、房间变成规范值。

约定（三個字 = 15 分鐘、三點三 = 3:15 这类香港说法都照顾到）：
- number：int / float
- duration：秒数（int）
- time：{"hour", "minute", "period": "am" | "pm" | None}；period 已知时 hour 是 24 小时制，
  未知时是口语里的 1 到 12
- date：ISO 日期字符串，相对"今日"解析
- percent：0..100 的 int
- room：房间的规范名（配置里可改）
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "兩": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
UNITS = {"十": 10, "百": 100, "千": 1000, "萬": 10000, "万": 10000}
NUM_INT = r"(?:\d+|[零〇一二兩两三四五六七八九十百千萬万廿卅]+)"
NUM = r"(?:\d+(?:\.\d+)?|[零〇一二兩两三四五六七八九十百千萬万廿卅]+(?:點[零〇一二三四五六七八九]+)?|半)"
WEEKDAYS = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
DEFAULT_ROOMS = [
    "客廳|廳|大廳",
    "睡房|房|房間|臥室",
    "廚房",
    "廁所|浴室|洗手間",
    "書房",
    "飯廳",
    "露台|陽台",
    "走廊",
]

_today: Callable[[], dt.date] = dt.date.today


def set_today(fn: Callable[[], dt.date]) -> None:
    """测试用：固定"今日"。"""
    global _today
    _today = fn


def parse_number(s: str) -> int | float | None:
    s = s.strip()
    if not s:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        return float(s) if "." in s else int(s)
    if s == "半":
        return 0.5
    s = s.replace("廿", "二十").replace("卅", "三十")
    frac = 0.0
    if "點" in s:
        s, tail = s.split("點", 1)
        if not tail or any(ch not in DIGITS for ch in tail):
            return None
        frac = float("0." + "".join(str(DIGITS[ch]) for ch in tail))
        if not s:
            return None
    total = 0
    section = 0
    num: int | None = None
    last_unit = 0
    for ch in s:
        if ch in DIGITS:
            if DIGITS[ch] == 0:  # 「一百零五」的零：明确的占位，后面不再按口语简写算
                num, last_unit = None, 0
                continue
            num = DIGITS[ch]
        elif ch in UNITS:
            u = UNITS[ch]
            if u == 10000:
                section += num if num is not None else (1 if section == 0 else 0)
                total += section * u
                section, num, last_unit = 0, None, 0
            else:
                section += (num if num is not None else 1) * u
                num, last_unit = None, u
        else:
            return None
    if num is not None:
        # "一百二" = 120、"三千五" = 3500：单位后面跟着的一个数字按下一级单位算
        section += num * (last_unit // 10) if last_unit >= 100 else num
    value: int | float = total + section
    if frac:
        value = value + frac
    return value


_DURATION = re.compile(
    rf"^(?P<n>{NUM})?(?P<half1>半)?(?P<ge>個)?(?P<half2>半)?(?P<unit>鐘頭|鐘|小時|分鐘|分|秒鐘|秒|字)$"
)
UNIT_SECONDS = {"鐘頭": 3600, "鐘": 3600, "小時": 3600, "分鐘": 60, "分": 60, "秒鐘": 1, "秒": 1, "字": 300}


def parse_duration(s: str) -> int | None:
    m = _DURATION.match(s.strip())
    if not m:
        return None
    n = parse_number(m.group("n")) if m.group("n") else (0 if m.group("half1") else 1)
    if n is None:
        return None
    if m.group("half1") or m.group("half2"):
        n = n + 0.5
    return int(round(n * UNIT_SECONDS[m.group("unit")]))


PERIOD_AM = ("朝早", "朝頭早", "早上", "早晨", "上晝", "上午", "凌晨", "半夜")
PERIOD_PM = ("晏晝", "下晝", "下午", "夜晚", "晚上", "今晚", "聽晚", "夜")
PERIOD_NOON = ("中午",)
PERIOD = "(?:" + "|".join(sorted(PERIOD_AM + PERIOD_PM + PERIOD_NOON, key=len, reverse=True)) + ")"
_TIME = re.compile(
    rf"^(?P<period>{PERIOD})?(?P<h>{NUM_INT})點(?:(?P<half>半)|(?P<zi>{NUM_INT})個字|(?P<min>{NUM_INT})分|(?P<zheng>正)|(?P<bare>{NUM_INT}))?$"
)


def parse_time(s: str) -> dict[str, Any] | None:
    m = _TIME.match(s.strip())
    if not m:
        return None
    h = parse_number(m.group("h"))
    if h is None or not float(h).is_integer() or not 0 <= h <= 24:
        return None
    h = int(h)
    minute = 0
    if m.group("half"):
        minute = 30
    elif m.group("zi"):
        z = parse_number(m.group("zi"))
        minute = int(z) * 5 if z is not None else 0
    elif m.group("min"):
        mm = parse_number(m.group("min"))
        minute = int(mm) if mm is not None else 0
    elif m.group("bare"):
        b = parse_number(m.group("bare"))
        if b is None:
            return None
        minute = int(b) * 5 if b <= 11 else int(b)
    if minute > 59:
        return None
    p = m.group("period")
    period: str | None = None
    if p in PERIOD_AM:
        period = "am"
        if h == 12 and p in ("凌晨", "半夜"):
            h = 0
    elif p in PERIOD_PM:
        period = "pm"
        if h < 12:
            h += 12
    elif p in PERIOD_NOON:
        period = "pm"
        if 1 <= h < 12:
            h += 12
    elif h >= 13 or h == 0:
        period = "am" if h < 12 else "pm"
    if h == 24:
        h, period = 0, "am"
    return {"hour": h, "minute": minute, "period": period}


def resolve_clock(value: dict[str, Any], now: dt.datetime) -> dt.datetime:
    """把 time 槽位变成下一个对应时刻：period 未知就取接下来最近的一个（上午或下午）。"""
    h, m, period = int(value["hour"]), int(value["minute"]), value.get("period")
    if period is not None:
        cands = [h]
    else:
        cands = sorted({h % 12, h % 12 + 12})
    for day in (0, 1):
        base = (now + dt.timedelta(days=day)).replace(hour=0, minute=0, second=0, microsecond=0)
        for hh in cands:
            t = base + dt.timedelta(hours=hh, minutes=m)
            if t > now:
                return t
    return now


_RELATIVE = {
    "今日": 0,
    "今天": 0,
    "今晚": 0,
    "聽日": 1,
    "明日": 1,
    "明天": 1,
    "聽晚": 1,
    "後日": 2,
    "後天": 2,
    "大後日": 3,
    "大後天": 3,
}
_WEEK = re.compile(r"^(?P<next>下個?|下下|今個?|呢個?)?(?:星期|禮拜|周|週)(?P<d>[一二三四五六日天])$")
_DAY = re.compile(rf"^(?:(?P<mon>{NUM_INT})月)?(?P<day>{NUM_INT})[號日]$")
DATE = (
    r"(?:大後日|大後天|後日|後天|聽日|明日|明天|聽晚|今晚|今日|今天|"
    r"(?:下個?|下下|今個?|呢個?)?(?:星期|禮拜|周|週)[一二三四五六日天]|"
    rf"(?:{NUM_INT}月)?{NUM_INT}[號日])"
)


def parse_date(s: str) -> str | None:
    s = s.strip()
    today = _today()
    if s in _RELATIVE:
        return (today + dt.timedelta(days=_RELATIVE[s])).isoformat()
    m = _WEEK.match(s)
    if m:
        target = WEEKDAYS[m.group("d")]
        delta = (target - today.weekday()) % 7
        nxt = m.group("next") or ""
        if nxt.startswith("下下"):
            delta += 14 if delta else 14
        elif nxt.startswith("下"):
            delta += 7
        return (today + dt.timedelta(days=delta)).isoformat()
    m = _DAY.match(s)
    if m:
        day = parse_number(m.group("day"))
        if day is None or not 1 <= day <= 31:
            return None
        day = int(day)
        if m.group("mon"):
            mon = parse_number(m.group("mon"))
            if mon is None or not 1 <= mon <= 12:
                return None
            mon = int(mon)
            year = today.year if (mon, day) >= (today.month, today.day) else today.year + 1
            try:
                return dt.date(year, mon, day).isoformat()
            except ValueError:
                return None
        year, mon = today.year, today.month
        if day < today.day:
            mon += 1
            if mon > 12:
                mon, year = 1, year + 1
        try:
            return dt.date(year, mon, day).isoformat()
        except ValueError:
            return None
    return None


PERCENT = rf"(?:{NUM}(?:%|成|個巴仙|percent|pa)|一半|最大|最細|最小|最高|最低)"


def parse_percent(s: str) -> int | None:
    s = s.strip()
    if s == "一半":
        return 50
    if s in ("最大", "最高"):
        return 100
    if s in ("最細", "最小", "最低"):
        return 10
    m = re.fullmatch(rf"({NUM})(%|成|個巴仙|percent|pa)", s)
    if not m:
        return None
    n = parse_number(m.group(1))
    if n is None:
        return None
    if m.group(2) == "成":
        n = n * 10
    return int(max(0, min(100, round(n))))


@dataclass(frozen=True)
class SlotType:
    name: str
    pattern: str  # 正则片段，只能用非捕获组
    parse: Callable[[str], Any]
    description: str  # 给 LLM 的说明
    json_type: str = "string"


def room_slot(rooms: list[str] | None = None) -> SlotType:
    entries = rooms or DEFAULT_ROOMS
    alias_to_name: dict[str, str] = {}
    for entry in entries:
        parts = [p.strip() for p in entry.split("|") if p.strip()]
        if not parts:
            continue
        for p in parts:
            alias_to_name[p] = parts[0]
    aliases = sorted(alias_to_name, key=len, reverse=True)
    pattern = "(?:" + "|".join(re.escape(a) for a in aliases) + ")" if aliases else "(?:客廳)"
    return SlotType("room", pattern, lambda s: alias_to_name.get(s.strip()), "房間名", "string")


def slot_types(rooms: list[str] | None = None) -> dict[str, SlotType]:
    types = [
        SlotType("number", NUM, parse_number, "數字", "number"),
        SlotType(
            "duration",
            rf"(?:{NUM}(?:個半|個)?|半個?)(?:鐘頭|鐘|小時|分鐘|分|秒鐘|秒|字)",
            parse_duration,
            "時長，秒數",
            "integer",
        ),
        SlotType(
            "time",
            rf"{PERIOD}?{NUM_INT}點(?:半|{NUM_INT}個字|{NUM_INT}分|正|{NUM_INT})?",
            parse_time,
            "時間，24 小時制 HH:MM",
            "string",
        ),
        SlotType("date", DATE, parse_date, "日期，YYYY-MM-DD", "string"),
        SlotType("percent", PERCENT, parse_percent, "百分比 0 到 100", "integer"),
        SlotType("text", r".+?", lambda s: s.strip(), "原話", "string"),
        room_slot(rooms),
    ]
    return {t.name: t for t in types}
