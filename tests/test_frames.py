import wave

import numpy as np

from catman_io.audio.frames import FRAME_SAMPLES, iter_frames, read_wav, resample, to_float32, to_int16


def test_int16_float_roundtrip():
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    i = to_int16(x)
    assert i.dtype == np.int16 and i[0] == -32767 and i[-1] == 32767
    assert np.allclose(to_float32(i), x, atol=1e-3)
    assert to_int16(i) is i  # 已经是 int16 就原样返回


def test_iter_frames_pads_last():
    x = np.arange(FRAME_SAMPLES * 2 + 100, dtype=np.int16)
    frames = list(iter_frames(x))
    assert len(frames) == 3
    assert all(len(f) == FRAME_SAMPLES for f in frames)
    assert frames[-1][100] == 0 and frames[-1][99] == x[-1]
    assert len(list(iter_frames(x, pad=False))) == 2


def test_resample_changes_length_and_keeps_dtype():
    x = (np.sin(np.linspace(0, 100, 48000)) * 20000).astype(np.int16)
    y = resample(x, 48000, 16000)
    assert y.dtype == np.int16 and len(y) == 16000
    assert resample(x, 16000, 16000) is x


def test_read_wav_stereo_48k_to_mono_16k(tmp_path):
    sr = 48000
    t = np.arange(sr) / sr
    left = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    right = np.zeros(sr, dtype=np.int16)
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.stack([left, right], axis=1).tobytes())
    mono = read_wav(path, channel=0)
    assert mono.dtype == np.int16 and len(mono) == 16000
    assert np.abs(mono).max() > 5000
    assert np.abs(read_wav(path, channel=1)).max() < 200
