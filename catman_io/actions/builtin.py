"""内置动作：報時、日期、計時、音量、停、再講、幫助、天氣（Open-Meteo，免 key）。"""

from __future__ import annotations

import datetime as dt
import json
import logging
import urllib.parse
from typing import Any

from catman_io.intent import Intent

from . import ActionContext, ActionRegistry, ActionResult

log = logging.getLogger(__name__)

WEEKDAYS = "一二三四五六日"
_CN = "零一二三四五六七八九"


def num_cn(n: int) -> str:
    """0 到 99 的粤语读法（2 讀「兩」只在整點用，這裡按數字讀）。"""
    n = int(n)
    if n < 10:
        return _CN[n]
    tens, ones = divmod(n, 10)
    if tens == 1:
        return "十" + (_CN[ones] if ones else "")
    return _CN[tens] + "十" + (_CN[ones] if ones else "")


def cantonese_time(now: dt.datetime) -> str:
    h, m = now.hour, now.minute
    if h < 5:
        period = "凌晨"
    elif h < 12:
        period = "朝早"
    elif h == 12:
        period = "中午"
    elif h < 18:
        period = "下晝"
    else:
        period = "夜晚"
    h12 = h % 12 or 12
    hour = "兩" if h12 == 2 else num_cn(h12)
    if m == 0:
        minute = "正"
    elif m == 30:
        minute = "半"
    elif m < 10:
        minute = f"零{num_cn(m)}分"
    else:
        minute = f"{num_cn(m)}分"
    return f"{period}{hour}點{minute}"


def cantonese_date(d: dt.date) -> str:
    return f"{d.year}年{d.month}月{d.day}號，星期{WEEKDAYS[d.weekday()]}"


def duration_label(seconds: int) -> str:
    seconds = int(seconds)
    if seconds % 3600 == 0:
        h = seconds // 3600
        return "一個鐘" if h == 1 else f"{'兩' if h == 2 else num_cn(h)}個鐘"
    if seconds % 1800 == 0:
        h = seconds // 3600
        return "半個鐘" if h == 0 else f"{'一' if h == 1 else ('兩' if h == 2 else num_cn(h))}個半鐘"
    if seconds % 60 == 0:
        m = seconds // 60
        return f"{num_cn(m) if m < 100 else m}分鐘"
    return f"{num_cn(seconds) if seconds < 100 else seconds}秒"


# ---- 动作 ----


def act_time(intent: Intent, ctx: ActionContext) -> ActionResult:
    return ActionResult(say=f"而家{cantonese_time(ctx.now())}。")


def act_date(intent: Intent, ctx: ActionContext) -> ActionResult:
    return ActionResult(say=f"今日係{cantonese_date(ctx.now().date())}。")


def act_timer(intent: Intent, ctx: ActionContext) -> ActionResult:
    seconds = intent.slots.get("duration")
    if not seconds:
        return ActionResult(say="要計幾耐呀？", followup=True)
    seconds = int(seconds)
    if seconds > 24 * 3600:
        return ActionResult(say="太耐喇，最多計二十四個鐘。", ok=False, error="duration too long")
    label = duration_label(seconds)
    t = ctx.timers.add(ctx.clock(), seconds, label)
    return ActionResult(say=f"好，{label}後叫你。", data={"timer_id": t.id, "seconds": seconds})


def act_timer_cancel(intent: Intent, ctx: ActionContext) -> ActionResult:
    n = ctx.timers.cancel_all()
    return ActionResult(say="取消咗。" if n else "而家冇計時緊。", data={"cancelled": n})


def _volume(ctx: ActionContext, value: float, say: str | None) -> ActionResult:
    if ctx.speaker is None:
        return ActionResult(say="呢度冇得調音量。", ok=False, error="no speaker")
    ctx.speaker.volume = max(0.0, min(1.0, value))
    return ActionResult(say=say, data={"volume": ctx.speaker.volume})


def act_volume_up(intent: Intent, ctx: ActionContext) -> ActionResult:
    cur = ctx.speaker.volume if ctx.speaker is not None else 0.0
    return _volume(ctx, max(cur, 0.1) + 0.15 if cur > 0 else 0.5, "大聲咗。")


def act_volume_down(intent: Intent, ctx: ActionContext) -> ActionResult:
    cur = ctx.speaker.volume if ctx.speaker is not None else 0.0
    return _volume(ctx, max(0.1, cur - 0.15), "細聲咗。")


def act_volume_set(intent: Intent, ctx: ActionContext) -> ActionResult:
    p = intent.slots.get("percent")
    if p is None:
        return ActionResult(say="調到幾多呀？", followup=True)
    p = int(p)
    return _volume(ctx, max(0.05, p / 100.0), f"音量{p}%。")


def act_mute(intent: Intent, ctx: ActionContext) -> ActionResult:
    if ctx.speaker is not None:
        ctx.state["volume_before_mute"] = ctx.speaker.volume
    return _volume(ctx, 0.0, None)


def act_stop(intent: Intent, ctx: ActionContext) -> ActionResult:
    if ctx.speaker is not None:
        ctx.speaker.stop()
    if ctx.brain is not None:
        ctx.brain.cancel()
    return ActionResult(say=None)


def act_repeat(intent: Intent, ctx: ActionContext) -> ActionResult:
    last = ctx.state.get("last_reply")
    return ActionResult(say=last or "頭先冇講嘢。")


def act_help(intent: Intent, ctx: ActionContext) -> ActionResult:
    return ActionResult(
        say="我識報時、報日期、計時、報天氣、調音量。其他嘢可以直接問我，要做嘢嘅話我會交俾 catman。"
    )


WMO = {
    0: "天晴",
    1: "大致天晴",
    2: "多雲",
    3: "陰天",
    45: "有霧",
    48: "有霧",
    51: "毛毛雨",
    53: "毛毛雨",
    55: "毛毛雨",
    56: "凍雨",
    57: "凍雨",
    61: "落雨",
    63: "落雨",
    65: "大雨",
    66: "凍雨",
    67: "凍雨",
    71: "落雪",
    73: "落雪",
    75: "大雪",
    77: "落雪",
    80: "驟雨",
    81: "驟雨",
    82: "大驟雨",
    85: "驟雪",
    86: "驟雪",
    95: "雷暴",
    96: "雷暴夾冰雹",
    99: "雷暴夾冰雹",
}
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
WEATHER_CACHE_SECONDS = 600


def fetch_weather(ctx: ActionContext, lat: float, lon: float) -> dict[str, Any]:
    cache = ctx.state.get("weather_cache")
    now = ctx.clock()
    if cache and now - cache[0] < WEATHER_CACHE_SECONDS:
        return cache[1]
    from .http import urllib_http

    q = urllib.parse.urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
            "timezone": "auto",
            "forecast_days": 4,
        }
    )
    http = ctx.http or urllib_http
    status, raw = http("GET", f"{OPEN_METEO}?{q}", {}, None, ctx.cfg.actions.http_timeout)
    if status != 200:
        raise RuntimeError(f"open-meteo HTTP {status}")
    data = json.loads(raw)
    ctx.state["weather_cache"] = (now, data)
    return data


def act_weather(intent: Intent, ctx: ActionContext) -> ActionResult:
    w = ctx.cfg.actions.weather
    if w.latitude is None or w.longitude is None:
        return ActionResult(
            say="未設定地點，要喺配置填經緯度先。", ok=False, error="weather location not configured"
        )
    try:
        data = fetch_weather(ctx, w.latitude, w.longitude)
    except Exception as e:  # noqa: BLE001
        log.warning("weather failed: %s", e)
        return ActionResult(say="查唔到天氣，遲啲再試。", ok=False, error=str(e))
    today = ctx.now().date()
    target = intent.slots.get("date") or today.isoformat()
    daily = data.get("daily") or {}
    days = list(daily.get("time") or [])
    if target not in days:
        return ActionResult(say="只有未來三日嘅天氣。", ok=False, error=f"no forecast for {target}")
    i = days.index(target)
    desc = WMO.get(int(daily["weather_code"][i]), "")
    tmax = round(daily["temperature_2m_max"][i])
    tmin = round(daily["temperature_2m_min"][i])
    rain = daily.get("precipitation_probability_max", [None] * len(days))[i]
    rain_text = f"，落雨機會{int(rain)}%" if rain is not None else ""
    name = w.name
    if target == today.isoformat():
        cur = data.get("current") or {}
        temp = round(cur.get("temperature_2m", tmax))
        cur_desc = WMO.get(int(cur.get("weather_code", daily["weather_code"][i])), desc)
        say = f"{name}而家{temp}度，{cur_desc}。今日最高{tmax}度，最低{tmin}度{rain_text}。"
    else:
        delta = (dt.date.fromisoformat(target) - today).days
        label = {1: "聽日", 2: "後日", 3: "大後日"}.get(
            delta, f"{target[5:7].lstrip('0')}月{target[8:].lstrip('0')}號"
        )
        say = f"{name}{label}{desc}，最高{tmax}度，最低{tmin}度{rain_text}。"
    return ActionResult(say=say, data={"date": target, "desc": desc, "max": tmax, "min": tmin})


def register_all(reg: ActionRegistry) -> None:
    reg.register("builtin.time", act_time)
    reg.register("builtin.date", act_date)
    reg.register("builtin.timer", act_timer)
    reg.register("builtin.timer_cancel", act_timer_cancel)
    reg.register("builtin.volume_up", act_volume_up)
    reg.register("builtin.volume_down", act_volume_down)
    reg.register("builtin.volume_set", act_volume_set)
    reg.register("builtin.mute", act_mute)
    reg.register("builtin.stop", act_stop)
    reg.register("builtin.repeat", act_repeat)
    reg.register("builtin.help", act_help)
    reg.register("builtin.weather", act_weather)
