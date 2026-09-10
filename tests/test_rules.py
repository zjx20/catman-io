import time

import pytest

from catman_io.config import Config
from catman_io.intent import BUILTIN_RULES, load_rules, rule_store
from catman_io.intent.cases import Case, append_cases, evaluate, read_cases
from catman_io.intent.rules import RuleError, RuleSet


@pytest.fixture(scope="module")
def builtin():
    return RuleSet.load([BUILTIN_RULES])


def test_builtin_lint_clean(builtin):
    errors, warnings = builtin.lint()
    assert errors == [], "\n".join(errors)
    assert warnings == []


@pytest.mark.parametrize(
    "text,intent,slots",
    [
        ("而家幾點呀", "time.now", {}),
        ("宜家幾多點啊", "time.now", {}),
        ("幫我set個十分鐘嘅timer", "timer.set", {"duration": 600}),
        ("三個字之後叫我", "timer.set", {"duration": 900}),
        ("帮我锡个十分钟嘅 TIMER", "timer.set", {"duration": 600}),
        ("計時三十秒", "timer.set", {"duration": 30}),
        ("音量調到一半", "volume.set", {"percent": 50}),
        ("聲量校到三成", "volume.set", {"percent": 30}),
        ("今日天气点啊", "weather.query", {}),
        ("聽日會唔會落雨", "weather.query", {"date": "__tomorrow__"}),
        ("大聲啲啦", "volume.up", {}),
        ("停", "stop", {}),
        ("唔該報時", "time.now", {}),
    ],
)
def test_builtin_matches(builtin, text, intent, slots):
    got = builtin.match(text)
    assert got is not None and got.name == intent, got
    assert got.tier == "rule" and got.rule_id.startswith(intent)
    for k, v in slots.items():
        if v == "__tomorrow__":
            import datetime as dt

            assert got.slots[k] == (dt.date.today() + dt.timedelta(days=1)).isoformat()
        else:
            assert got.slots[k] == v


@pytest.mark.parametrize(
    "text", ["我想飲杯咖啡", "幫我播首歌", "你叫咩名", "講個笑話嚟聽下", "幾點都得", "十分鐘前發生咗乜嘢"]
)
def test_builtin_does_not_match_chat(builtin, text):
    assert builtin.match(text) is None, builtin.match(text)


def test_slot_parse_failure_skips_rule(builtin):
    # 「卅二號」能被 date 的模式匹配到，但解析成日期失败，这条规则要跳过而不是报错
    assert builtin.match("十五號天氣點").name == "weather.query"
    assert builtin.match("卅二號天氣點") is None


def test_site_rules_override_and_extend(tmp_path):
    site = tmp_path / "site.yaml"
    site.write_text(
        "version: 1\nintents:\n"
        "  - name: time.now\n    patterns: ['^報時$']\n    examples: ['報時']\n    action: builtin.time\n"
        "  - name: light.on\n    slots: {room: room}\n"
        "    patterns: ['^{polite}開(?:埋)?({room})?(?:盞)?燈{tail}$']\n"
        "    examples: ['開燈', '幫我開廳燈']\n    action: {type: http, url: 'http://x'}\n",
        encoding="utf-8",
    )
    rs = RuleSet.load([BUILTIN_RULES, site])
    assert rs.match("而家幾點") is None  # 被 site 覆盖
    assert rs.match("報時").name == "time.now"
    got = rs.match("幫我開廳燈")
    assert got.name == "light.on" and got.slots == {"room": "客廳"}
    assert rs.match("開燈").slots == {}
    assert rs.intents["light.on"].action == {"type": "http", "url": "http://x"}
    assert rs.lint()[0] == []


def test_rule_errors_are_specific():
    with pytest.raises(RuleError, match="unknown slot type"):
        RuleSet.from_docs([("x", {"intents": [{"name": "a", "patterns": ["{bogus}"]}]})])
    with pytest.raises(RuleError, match="unknown keys"):
        RuleSet.from_docs([("x", {"intents": [{"name": "a", "pattern": ["x"]}]})])
    with pytest.raises(RuleError, match="bad pattern"):
        RuleSet.from_docs([("x", {"intents": [{"name": "a", "patterns": ["(("]}]})])
    rs = RuleSet.from_docs([("x", {"intents": [{"name": "a", "patterns": ["x*"], "examples": ["y"]}]})])
    errors, warnings = rs.lint()
    assert any("empty string" in e for e in errors) and any("not anchored" in w for w in warnings)


def test_tools_from_rules(builtin):
    tools = builtin.tools()
    names = {t["function"]["name"] for t in tools}
    assert "timer_set" in names and "weather_query" in names
    timer = next(t for t in tools if t["function"]["name"] == "timer_set")["function"]
    assert timer["parameters"]["required"] == ["duration"]
    assert timer["parameters"]["properties"]["duration"]["type"] == "integer"
    weather = next(t for t in tools if t["function"]["name"] == "weather_query")["function"]
    assert weather["parameters"]["required"] == []
    assert builtin.intent_for_tool("timer_set") == "timer.set"


def test_cases_roundtrip_and_evaluate(tmp_path, builtin):
    path = tmp_path / "cases.jsonl"
    cases = [
        Case("而家幾點", "time.now"),
        Case("三個字之後叫我", "timer.set", {"duration": 900}, source="llm", turn_id="t1"),
        Case("我想飲咖啡", "none"),
        Case("幫我set個十分鐘嘅timer", "timer.set", {"duration": 660}),  # 故意错
        Case("你叫咩名", "help"),  # 故意错
    ]
    assert append_cases(path, cases) == 5
    back = read_cases(path)
    assert [c.text for c in back] == [c.text for c in cases] and back[1].turn_id == "t1"
    report = evaluate(builtin, back)
    assert report.passed == 3 and report.failed == 2
    reasons = {r.case.text: r.reason for r in report.failures()}
    assert "slots differ" in reasons["幫我set個十分鐘嘅timer"] and reasons["你叫咩名"] == "no match"
    assert "2 failed" in report.format() and report.to_dict()["failed"] == 2


def test_rule_store_reloads_and_keeps_old_on_error(tmp_path):
    cfg = Config.load(None)
    cfg.data_dir = str(tmp_path)
    store = rule_store(cfg)
    assert store.current.match("而家幾點").name == "time.now"
    site = cfg.rules_dir / "site.yaml"
    site.parent.mkdir(parents=True)
    site.write_text(
        "version: 1\nintents:\n  - name: ping\n    patterns: ['^ping$']\n    examples: ['ping']\n",
        encoding="utf-8",
    )
    assert store.reload_if_changed() and store.current.match("ping").name == "ping"
    assert not store.reload_if_changed()
    time.sleep(0.01)
    site.write_text("version: 1\nintents:\n  - name: ping\n    patterns: ['((']\n", encoding="utf-8")
    assert not store.reload_if_changed()
    assert store.current.match("ping").name == "ping"  # 坏文件不影响旧规则
    with pytest.raises(RuleError):  # 直接加载则如实报错
        load_rules(cfg)
