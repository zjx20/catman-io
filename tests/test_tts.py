import os

import numpy as np
import pytest

from catman_io.tts import clean_for_speech, create_synthesizer, split_sentences


def test_split_sentences_keeps_punctuation_and_soft_breaks_long_ones():
    assert split_sentences("而家三點半。聽日落雨！你好嗎？") == ["而家三點半。", "聽日落雨！", "你好嗎？"]
    assert split_sentences("第一行\n第二行") == ["第一行", "第二行"]
    long = "，".join(["好長嘅一句話"] * 12)
    parts = split_sentences(long, max_len=20)
    assert len(parts) > 3 and all(len(p) <= 27 for p in parts)
    assert split_sentences("……。！") == []


def test_clean_for_speech_strips_markdown_links_emoji_and_wake_word():
    text = (
        "## 今日天氣\n\n- **晴**，28 度 😀\n- 詳情見 [天文台](https://hko.gov.hk) 或 https://x.y/z\n\n"
        "```py\nprint(1)\n```\n小貓人幫到你 🎉"
    )
    out = clean_for_speech(text)
    assert "**" not in out and "#" not in out and "http" not in out and "😀" not in out
    assert "天文台" in out and "一個連結" in out and "略過一段代碼" in out
    assert "小貓人" not in out and "貓人幫到你" in out
    assert out.startswith("今日天氣")


def test_clean_for_speech_truncates_at_sentence_boundary():
    text = "第一句好長好長好長。" * 40
    out = clean_for_speech(text, max_chars=50)
    assert out.endswith("。……") and len(out) <= 55


def test_create_synthesizer_none_backend():
    from catman_io.config import Config

    cfg = Config.load(None)
    cfg.tts.backend = "none"
    assert create_synthesizer(cfg) is None
    cfg.tts.backend = "bogus"
    with pytest.raises(ValueError):
        create_synthesizer(cfg)


@pytest.mark.network
def test_edge_tts_synthesizes_and_caches(tmp_path):
    pytest.importorskip("edge_tts")
    from catman_io.tts.edge import EdgeTTS

    tts = EdgeTTS("zh-HK-HiuMaanNeural", cache_dir=tmp_path)
    pcm = tts.synthesize("你好，我係貓人。")
    assert pcm.dtype == np.int16 and 0.5 < len(pcm) / 16000 < 5
    assert tts.last_fetch_seconds > 0
    assert len(list(tmp_path.glob("*.wav"))) == 1
    again = tts.synthesize("你好，我係貓人。")
    assert tts.last_fetch_seconds == 0.0 and len(again) == len(pcm)
    assert os.path.getsize(next(tmp_path.glob("*.wav"))) > 1000
