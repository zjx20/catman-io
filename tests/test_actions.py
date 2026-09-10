import datetime as dt
import json

from catman_io.actions import ActionContext, ActionResult, default_registry
from catman_io.actions.builtin import cantonese_date, cantonese_time, duration_label, num_cn
from catman_io.actions.http import http_action, render
from catman_io.actions.timers import TimerService
from catman_io.config import Config
from catman_io.intent import BUILTIN_RULES, Intent
from catman_io.intent.rules import IntentSpec, RuleSet


class FakeSpeaker:
    def __init__(self):
        self.volume = 0.8
        self.stopped = 0

    def stop(self):
        self.stopped += 1


class FakeBrain:
    def __init__(self):
        self.cancelled = 0

    def cancel(self):
        self.cancelled += 1


def ctx(**kw):
    cfg = Config.load(None)
    clock = {"t": 100.0}
    return ActionContext(
        cfg=cfg,
        timers=TimerService(),
        speaker=FakeSpeaker(),
        brain=FakeBrain(),
        now=lambda: dt.datetime(2026, 9, 10, 14, 5),
        clock=lambda: clock["t"],
        **kw,
    )


def test_cantonese_readouts():
    assert cantonese_time(dt.datetime(2026, 1, 1, 14, 5)) == "下晝兩點零五分"
    assert cantonese_time(dt.datetime(2026, 1, 1, 9, 30)) == "朝早九點半"
    assert cantonese_time(dt.datetime(2026, 1, 1, 0, 0)) == "凌晨十二點正"
    assert cantonese_time(dt.datetime(2026, 1, 1, 12, 45)) == "中午十二點四十五分"
    assert cantonese_time(dt.datetime(2026, 1, 1, 23, 12)) == "夜晚十一點十二分"
    assert cantonese_date(dt.date(2026, 9, 10)) == "2026年9月10號，星期四"
    assert [num_cn(n) for n in (0, 7, 10, 11, 20, 21, 59)] == [
        "零",
        "七",
        "十",
        "十一",
        "二十",
        "二十一",
        "五十九",
    ]
    assert (
        duration_label(600) == "十分鐘"
        and duration_label(3600) == "一個鐘"
        and duration_label(1800) == "半個鐘"
    )
    assert (
        duration_label(5400) == "一個半鐘"
        and duration_label(7200) == "兩個鐘"
        and duration_label(30) == "三十秒"
    )
    assert duration_label(900) == "十五分鐘"


def test_builtin_time_date_help_repeat():
    reg = default_registry()
    rules = RuleSet.load([BUILTIN_RULES])
    c = ctx()
    r = reg.run(Intent("time.now"), rules.intents["time.now"], c)
    assert r.say == "而家下晝兩點零五分。" and r.ok
    assert reg.run(Intent("date.today"), rules.intents["date.today"], c).say.startswith("今日係2026年9月10號")
    assert "報時" in reg.run(Intent("help"), rules.intents["help"], c).say
    assert reg.run(Intent("repeat"), rules.intents["repeat"], c).say == "頭先冇講嘢。"
    c.state["last_reply"] = "而家三點。"
    assert reg.run(Intent("repeat"), rules.intents["repeat"], c).say == "而家三點。"


def test_timer_set_and_cancel_and_poll():
    reg = default_registry()
    rules = RuleSet.load([BUILTIN_RULES])
    c = ctx()
    r = reg.run(Intent("timer.set", {"duration": 600}), rules.intents["timer.set"], c)
    assert r.say == "好，十分鐘後叫你。" and r.data["seconds"] == 600
    assert c.timers.poll(699.0) == []
    fired = c.timers.poll(700.0)
    assert len(fired) == 1 and fired[0].say == "十分鐘到喇" and fired[0].seconds == 600
    r = reg.run(Intent("timer.set"), rules.intents["timer.set"], c)
    assert r.followup and "幾耐" in r.say
    reg.run(Intent("timer.set", {"duration": 60}), rules.intents["timer.set"], c)
    assert reg.run(Intent("timer.cancel"), rules.intents["timer.cancel"], c).data["cancelled"] == 1
    assert reg.run(Intent("timer.cancel"), rules.intents["timer.cancel"], c).say == "而家冇計時緊。"
    assert not reg.run(Intent("timer.set", {"duration": 90000}), rules.intents["timer.set"], c).ok


def test_volume_and_stop():
    reg = default_registry()
    rules = RuleSet.load([BUILTIN_RULES])
    c = ctx()
    assert reg.run(Intent("volume.up"), rules.intents["volume.up"], c).say == "大聲咗。"
    assert abs(c.speaker.volume - 0.95) < 1e-9
    reg.run(Intent("volume.down"), rules.intents["volume.down"], c)
    assert abs(c.speaker.volume - 0.8) < 1e-9
    r = reg.run(Intent("volume.set", {"percent": 30}), rules.intents["volume.set"], c)
    assert r.say == "音量30%。" and abs(c.speaker.volume - 0.3) < 1e-9
    assert (
        reg.run(Intent("volume.mute"), rules.intents["volume.mute"], c).say is None
        and c.speaker.volume == 0.0
    )
    r = reg.run(Intent("stop"), rules.intents["stop"], c)
    assert r.say is None and c.speaker.stopped == 1 and c.brain.cancelled == 1
    c.speaker = None
    assert not reg.run(Intent("volume.up"), rules.intents["volume.up"], c).ok


def test_render_templates(monkeypatch):
    monkeypatch.setenv("HASS_TOKEN", "abc")
    ctxd = {"room": "客廳", "r": {"current": {"temperature_2m": 28.0}, "list": [1, 2]}}
    assert render("開咗{room|}燈喇", ctxd) == "開咗客廳燈喇"
    assert render("開咗{room|}燈喇", {}) == "開咗燈喇"
    assert render("light.{room|living_room}", {}) == "light.living_room"
    assert render("{r.current.temperature_2m}度 {r.list.1} {r.nope|?}", ctxd) == "28度 2 ?"
    assert render("Bearer ${HASS_TOKEN} ${NOPE}", {}) == "Bearer abc "


def test_http_action_templating_and_say():
    calls = []

    def fake_http(method, url, headers, body, timeout):
        calls.append((method, url, headers, body, timeout))
        return 200, json.dumps({"state": "on", "attributes": {"brightness": 200}}).encode()

    c = ctx(http=fake_http)
    action = {
        "type": "http",
        "method": "POST",
        "url": "http://ha.local:8123/api/services/light/turn_{state}",
        "headers": {"Authorization": "Bearer ${HASS_TOKEN}"},
        "json": {"entity_id": "light.{room|living_room}", "brightness": 200},
        "query": {"a": "{room}"},
        "timeout": 3,
    }
    intent = Intent("light.on", {"room": "客廳", "state": "on"})
    r = http_action(action, intent, c)
    method, url, headers, body, timeout = calls[0]
    assert method == "POST" and url == "http://ha.local:8123/api/services/light/turn_on?a=%E5%AE%A2%E5%BB%B3"
    assert headers["Content-Type"] == "application/json" and timeout == 3
    assert json.loads(body) == {"entity_id": "light.客廳", "brightness": 200}
    assert r.ok and r.say == "搞掂" and r.data["state"] == "on"
    reg = default_registry()
    spec = IntentSpec("light.on", action=action, say="開咗{room|}燈，光度{r.attributes.brightness}")
    assert reg.run(intent, spec, c).say == "開咗客廳燈，光度200"

    def bad_http(*a):
        return 500, b"boom"

    c.http = bad_http
    r = http_action(action, intent, c)
    assert not r.ok and r.error == "HTTP 500" and "做唔到" in r.say

    def raising(*a):
        raise OSError("down")

    c.http = raising
    assert not http_action(action, intent, c).ok


def test_registry_rejects_bad_specs():
    reg = default_registry()
    c = ctx()
    assert not reg.run(Intent("x"), None, c).ok
    assert not reg.run(Intent("x"), IntentSpec("x", action="builtin.nope"), c).ok
    assert not reg.run(Intent("x"), IntentSpec("x", action={"type": "shell"}), c).ok
    assert isinstance(reg.run(Intent("time.now"), IntentSpec("t", action="builtin.time"), c), ActionResult)


OPEN_METEO_JSON = {
    "current": {"temperature_2m": 28.4, "relative_humidity_2m": 80, "weather_code": 2},
    "daily": {
        "time": ["2026-09-10", "2026-09-11", "2026-09-12", "2026-09-13"],
        "temperature_2m_max": [31.2, 30.1, 29.0, 28.0],
        "temperature_2m_min": [26.0, 25.5, 25.0, 24.0],
        "precipitation_probability_max": [20, 70, 10, 5],
        "weather_code": [2, 61, 0, 3],
    },
}


def test_weather_today_and_tomorrow_with_cache():
    calls = []

    def fake_http(method, url, headers, body, timeout):
        calls.append(url)
        return 200, json.dumps(OPEN_METEO_JSON).encode()

    c = ctx(http=fake_http)
    reg = default_registry()
    rules = RuleSet.load([BUILTIN_RULES])
    spec = rules.intents["weather.query"]
    r = reg.run(Intent("weather.query"), spec, c)
    assert not r.ok and "經緯度" in r.say
    c.cfg.actions.weather.latitude, c.cfg.actions.weather.longitude, c.cfg.actions.weather.name = (
        22.3,
        114.2,
        "香港",
    )
    r = reg.run(Intent("weather.query"), spec, c)
    assert r.say == "香港而家28度，多雲。今日最高31度，最低26度，落雨機會20%。" and r.ok
    assert "latitude=22.3" in calls[0]
    r = reg.run(Intent("weather.query", {"date": "2026-09-11"}), spec, c)
    assert r.say == "香港聽日落雨，最高30度，最低26度，落雨機會70%。"
    assert len(calls) == 1  # 十分钟内用缓存
    r = reg.run(Intent("weather.query", {"date": "2026-09-20"}), spec, c)
    assert not r.ok and "未來三日" in r.say
