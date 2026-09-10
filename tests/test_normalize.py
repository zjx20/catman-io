from catman_io.intent.normalize import normalize, to_hk


def test_normalize_converts_and_strips():
    assert normalize("今日天气点啊？") == "今日天氣點啊"
    assert normalize("幫我 set 個 Timer！") == "幫我set個timer"
    assert normalize("音量 ５０％") == "音量50%"
    assert normalize("<|yue|>宜家幾點") == "而家幾點"
    assert normalize("小貓人 開燈") == "開燈"
    assert normalize("小貓人 開燈", strip_wake=False) == "小貓人開燈"
    assert normalize("大聲D") == "大聲啲"
    assert normalize("噉樣得唔得") == "咁得唔得"


def test_to_hk_keeps_regex_punctuation():
    assert to_hk("^(开|開)灯$") == "^(開|開)燈$"
    assert to_hk("Timer|TIMER") == "timer|timer"
