import datetime as dt

import pytest

from catman_io.intent import slots
from catman_io.intent.slots import (
    parse_date,
    parse_duration,
    parse_number,
    parse_percent,
    parse_time,
    resolve_clock,
    slot_types,
)


@pytest.mark.parametrize(
    "text,value",
    [
        ("十五", 15),
        ("兩", 2),
        ("廿三", 23),
        ("卅", 30),
        ("一百二十", 120),
        ("一百二", 120),
        ("三百", 300),
        ("十", 10),
        ("十二", 12),
        ("二十", 20),
        ("三千五", 3500),
        ("一萬二千", 12000),
        ("一百零五", 105),
        ("3.5", 3.5),
        ("42", 42),
        ("半", 0.5),
        ("三點五", 3.5),
        ("零", 0),
        ("abc", None),
    ],
)
def test_parse_number(text, value):
    assert parse_number(text) == value


@pytest.mark.parametrize(
    "text,seconds",
    [
        ("十分鐘", 600),
        ("半個鐘", 1800),
        ("一個半鐘", 5400),
        ("三個字", 900),
        ("兩個鐘頭", 7200),
        ("三十秒", 30),
        ("1個鐘", 3600),
        ("廿分鐘", 1200),
        ("半分鐘", 30),
        ("兩個字", 600),
        ("十秒鐘", 10),
        ("一個鐘", 3600),
        ("五分", 300),
        ("十", None),
        ("個鐘", 3600),
    ],
)
def test_parse_duration(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize(
    "text,hour,minute,period",
    [
        ("三點半", 3, 30, None),
        ("下晝三點三", 15, 15, "pm"),
        ("夜晚八點", 20, 0, "pm"),
        ("朝早七點", 7, 0, "am"),
        ("十二點", 12, 0, None),
        ("三點十五分", 3, 15, None),
        ("三點正", 3, 0, None),
        ("十八點", 18, 0, "pm"),
        ("凌晨十二點", 0, 0, "am"),
        ("中午一點", 13, 0, "pm"),
        ("今晚十點半", 22, 30, "pm"),
        ("三點四個字", 3, 20, None),
        ("五點四十", 5, 40, None),
        ("下午六點半", 18, 30, "pm"),
    ],
)
def test_parse_time(text, hour, minute, period):
    assert parse_time(text) == {"hour": hour, "minute": minute, "period": period}


def test_parse_time_rejects_nonsense():
    assert parse_time("廿六點") is None and parse_time("三點九十分") is None


def test_resolve_clock_picks_next_occurrence():
    now = dt.datetime(2026, 9, 10, 14, 0)
    assert resolve_clock({"hour": 3, "minute": 30, "period": None}, now) == dt.datetime(2026, 9, 10, 15, 30)
    assert resolve_clock({"hour": 1, "minute": 0, "period": None}, now) == dt.datetime(2026, 9, 11, 1, 0)
    assert resolve_clock({"hour": 7, "minute": 0, "period": "am"}, now) == dt.datetime(2026, 9, 11, 7, 0)
    assert resolve_clock({"hour": 20, "minute": 0, "period": "pm"}, now) == dt.datetime(2026, 9, 10, 20, 0)


def test_parse_date_relative(monkeypatch):
    slots.set_today(lambda: dt.date(2026, 9, 10))  # 星期四
    try:
        assert parse_date("今日") == "2026-09-10"
        assert parse_date("聽日") == "2026-09-11"
        assert parse_date("後日") == "2026-09-12"
        assert parse_date("大後日") == "2026-09-13"
        assert parse_date("星期三") == "2026-09-16"
        assert parse_date("星期四") == "2026-09-10"
        assert parse_date("下星期三") == "2026-09-23"
        assert parse_date("禮拜日") == "2026-09-13"
        assert parse_date("十五號") == "2026-09-15"
        assert parse_date("五號") == "2026-10-05"
        assert parse_date("三月一號") == "2027-03-01"
        assert parse_date("十二月廿五號") == "2026-12-25"
        assert parse_date("卅二號") is None
    finally:
        slots.set_today(dt.date.today)


@pytest.mark.parametrize(
    "text,value",
    [("五十%", 50), ("三成", 30), ("一半", 50), ("最大", 100), ("最細", 10), ("120%", 100), ("50", None)],
)
def test_parse_percent(text, value):
    assert parse_percent(text) == value


def test_room_slot_uses_canonical_name():
    st = slot_types(["客廳|廳", "睡房|房"])["room"]
    import re

    assert re.fullmatch(st.pattern, "廳") and st.parse("廳") == "客廳"
    assert st.parse("房") == "睡房"
    assert slot_types()["room"].parse("浴室") == "廁所"
