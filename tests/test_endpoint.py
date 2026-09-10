"""端点检测的纯逻辑测试：用人声概率序列驱动，不需要模型。"""

import numpy as np

from catman_io.config import VadConfig
from catman_io.vad.endpoint import DONE, WAIT, Endpointer, NoSpeech, SpeechStart, Utterance

FRAME = np.ones(1280, dtype=np.int16)


def run(ep: Endpointer, probs):
    events = []
    for i, p in enumerate(probs):
        events += ep.feed(FRAME * (i + 1), p)
    return events


def test_no_speech_times_out():
    ep = Endpointer(VadConfig(no_speech_timeout_ms=800))
    ev = run(ep, [0.0] * 20)
    assert len(ev) == 1 and isinstance(ev[0], NoSpeech)
    assert abs(ev[0].t - 0.8) < 1e-6
    assert ep.state == DONE
    assert ep.feed(FRAME, 0.9) == []  # DONE 之后不再吐事件


def test_utterance_with_pre_roll_and_tail_pad():
    cfg = VadConfig(pre_roll_ms=240, start_frames=2, trailing_silence_ms=400, tail_pad_ms=160)
    ep = Endpointer(cfg)
    probs = [0.0] * 5 + [0.9] * 10 + [0.0] * 10
    ev = run(ep, probs)
    assert isinstance(ev[0], SpeechStart) and abs(ev[0].t - 0.4) < 1e-6
    utt = ev[1]
    assert isinstance(utt, Utterance) and not utt.forced
    # 3 帧 pre-roll + 10 帧人声 + 2 帧尾部
    assert len(utt.audio) == (3 + 10 + 2) * 1280
    assert utt.audio[0] == 3  # 第一帧是 pre-roll 里的第 3 帧（帧值 = 序号+1）
    assert ep.state == DONE


def test_short_blip_is_discarded_then_timeout():
    cfg = VadConfig(min_speech_ms=300, trailing_silence_ms=320, no_speech_timeout_ms=2000, start_frames=1)
    ep = Endpointer(cfg)
    probs = [0.0] * 3 + [0.9] * 2 + [0.0] * 30
    ev = run(ep, probs)
    kinds = [type(e).__name__ for e in ev]
    assert kinds == ["SpeechStart", "NoSpeech"]


def test_pause_inside_sentence_does_not_end_it():
    cfg = VadConfig(trailing_silence_ms=480, start_frames=1)
    ep = Endpointer(cfg)
    probs = [0.9] * 5 + [0.1] * 4 + [0.9] * 5 + [0.0] * 8
    ev = run(ep, probs)
    utts = [e for e in ev if isinstance(e, Utterance)]
    assert len(utts) == 1
    assert len(utts[0].audio) == (5 + 4 + 5 + 2) * 1280  # 中间的停顿保留在句子里


def test_max_utterance_forces_end():
    cfg = VadConfig(max_utterance_ms=800, start_frames=1)
    ep = Endpointer(cfg)
    ev = run(ep, [0.9] * 30)
    utts = [e for e in ev if isinstance(e, Utterance)]
    assert len(utts) == 1 and utts[0].forced
    assert len(utts[0].audio) == 10 * 1280


def test_followup_mode_has_no_timeout_and_stricter_start():
    cfg = VadConfig(
        no_speech_timeout_ms=400, start_threshold=0.5, followup_start_threshold=0.8, followup_start_frames=2
    )
    ep = Endpointer(cfg)
    ep.reset(followup=True)
    assert run(ep, [0.6] * 20) == []  # 0.6 在跟进模式下不算开口，也不超时
    assert ep.state == WAIT
    ev = run(ep, [0.9, 0.9])
    assert isinstance(ev[0], SpeechStart)


def test_reset_clears_everything():
    ep = Endpointer(VadConfig(start_frames=1))
    run(ep, [0.9] * 3)
    ep.reset()
    assert ep.state == WAIT and ep.elapsed == 0.0
