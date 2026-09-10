"""真 silero 模型 + 端点器：用仓库里的样例录音检查能切出一句话。"""

import numpy as np
import pytest

from catman_io.audio.frames import iter_frames, read_wav
from catman_io.config import VadConfig
from catman_io.vad import Endpointer, NoSpeech, SileroVAD, Utterance
from catman_io.wakeword import ensure_base_models

SILENCE = np.zeros(16000, dtype=np.int16)


@pytest.fixture(scope="module")
def vad():
    try:
        ensure_base_models(quiet=True)
        return SileroVAD()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"silero model unavailable: {e}")


def test_speech_probability_high_on_speech_low_on_silence(vad):
    speech = read_wav("tests/data/positive_siu_maau_jan_hiugaai.wav")
    vad.reset()
    speech_probs = [vad.process(f) for f in iter_frames(speech)]
    vad.reset()
    silence_probs = [vad.process(f) for f in iter_frames(SILENCE)]
    assert max(speech_probs) > 0.8
    assert max(silence_probs) < 0.1


def test_endpointer_cuts_one_utterance(vad):
    speech = read_wav("tests/data/negative_weather_wanlung.wav")
    audio = np.concatenate([SILENCE, speech, SILENCE, SILENCE])
    ep = Endpointer(VadConfig())
    vad.reset()
    events = []
    for f in iter_frames(audio):
        events += ep.feed(f, vad.process(f))
    utts = [e for e in events if isinstance(e, Utterance)]
    assert len(utts) == 1 and not utts[0].forced
    assert not any(isinstance(e, NoSpeech) for e in events)
    dur = len(utts[0].audio) / 16000
    assert abs(dur - len(speech) / 16000) < 0.8
    assert 0.5 < utts[0].t_start < 1.1  # 1.0 s 处开口，减去 pre-roll 与起声帧


def test_endpointer_times_out_on_silence(vad):
    ep = Endpointer(VadConfig(no_speech_timeout_ms=1000))
    vad.reset()
    events = []
    for f in iter_frames(np.concatenate([SILENCE, SILENCE])):
        events += ep.feed(f, vad.process(f))
    assert [type(e).__name__ for e in events] == ["NoSpeech"]
