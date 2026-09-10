import time

import numpy as np

from catman_io.tts.earcons import KINDS, earcon
from catman_io.tts.speaker import ListOutput, Speaker, WavOutput


def pcm(n, value=1000):
    return np.full(n, value, dtype=np.int16)


def test_say_wait_and_volume():
    out = ListOutput()
    sp = Speaker(out, volume=0.5, blocksize=320)
    sp.set_gen(1)
    assert sp.say(pcm(1000), gen=1)
    assert sp.wait(timeout=2.0)
    sp.close()
    assert len(out.audio) == 1000 and out.audio[0] == 500
    assert 1 in sp.first_audio


def test_stale_generation_is_dropped():
    out = ListOutput()
    sp = Speaker(out, volume=1.0)
    sp.set_gen(2)
    assert not sp.say(pcm(640), gen=1)
    assert sp.say(pcm(640), gen=None)  # 提示音不看回合
    assert sp.wait(timeout=2.0)
    sp.close()
    assert len(out.audio) == 640


def test_stop_clears_queue_quickly():
    out = ListOutput(delay=0.01)
    sp = Speaker(out, volume=1.0, blocksize=320)
    sp.set_gen(1)
    sp.say(pcm(320 * 100), gen=1)  # 100 块 ≈ 1 s 的模拟播放
    time.sleep(0.05)
    sp.stop()
    assert sp.wait(timeout=1.0)
    sp.close()
    assert len(out.chunks) < 30


def test_wav_output(tmp_path):
    from catman_io.audio.frames import read_wav

    path = tmp_path / "o.wav"
    sp = Speaker(WavOutput(path), volume=1.0)
    sp.set_gen(1)
    sp.say(pcm(3200, 123), gen=1)
    sp.wait(timeout=2.0)
    sp.close()
    got = read_wav(path)
    assert len(got) == 3200 and got[0] == 123


def test_earcons_are_short_int16():
    for k in KINDS:
        x = earcon(k)
        assert x.dtype == np.int16 and 0.05 < len(x) / 16000 < 2.0
        assert np.abs(x).max() > 1000
    assert earcon("unknown").dtype == np.int16
