"""listen 命令离线跑一段"唤醒词 + 一句话"的 WAV，应切出并保存一句。"""

import wave

import numpy as np
import pytest

from catman_io.audio.frames import read_wav
from catman_io.cli import main
from catman_io.wakeword import bundled_models, ensure_base_models


@pytest.fixture(scope="module")
def base_models():
    try:
        ensure_base_models(quiet=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"openWakeWord base models unavailable: {e}")


@pytest.mark.skipif(not bundled_models(), reason="no bundled wake-word model")
def test_listen_wav_saves_one_utterance(tmp_path, base_models, capsys):
    wake = read_wav("tests/data/positive_siu_maau_jan_hiugaai.wav")
    sentence = read_wav("tests/data/negative_weather_wanlung.wav")
    gap = np.zeros(int(0.4 * 16000), dtype=np.int16)
    audio = np.concatenate([gap, wake, gap, sentence, gap])
    wav = tmp_path / "in.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(audio.tobytes())
    out = tmp_path / "utt"
    rc = main(["listen", "--wav", str(wav), "--out-dir", str(out), "--no-asr", "--max-seconds", "30"])
    assert rc == 0
    saved = sorted(out.glob("*.wav"))
    assert len(saved) == 1, capsys.readouterr().out
    got = read_wav(saved[0])
    assert abs(len(got) - len(sentence)) < 0.8 * 16000
