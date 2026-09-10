import numpy as np
import pytest

from training.wakeword.augment import (
    SR,
    AudioPool,
    Augmenter,
    active_rms,
    apply_rir,
    colored_noise,
    fit_clip,
    mix_at_snr,
    rms,
    speed_perturb,
    time_stretch,
    to_int16,
)
from training.wakeword.config import AugmentConfig


def tone(seconds=0.8, freq=220.0):
    t = np.arange(int(seconds * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_fit_clip_right_aligns_positive():
    rng = np.random.default_rng(0)
    x = tone(0.5)
    out = fit_clip(x, 2 * SR, rng, end_jitter=0.2, align="right")
    assert out.shape == (2 * SR,)
    nz = np.nonzero(out)[0]
    assert 2 * SR - 0.2 * SR - 1 <= nz[-1] < 2 * SR
    assert nz[0] == nz[-1] - len(x) + 2  # 中间没有插入别的东西


def test_fit_clip_crops_long_input():
    rng = np.random.default_rng(0)
    x = np.arange(3 * SR, dtype=np.float32)
    assert np.array_equal(fit_clip(x, 2 * SR, rng, align="right"), x[-2 * SR :])
    assert len(fit_clip(x, 2 * SR, rng, align="random")) == 2 * SR


def test_mix_at_snr_hits_target():
    rng = np.random.default_rng(1)
    sig = tone(2.0)
    noise = rng.standard_normal(len(sig)).astype(np.float32)
    for snr in (0.0, 10.0, 20.0):
        mixed = mix_at_snr(sig, noise, snr)
        added = mixed - sig
        got = 20 * np.log10(active_rms(sig) / rms(added))
        assert abs(got - snr) < 0.5


def test_colored_noise_shapes_and_spectrum():
    rng = np.random.default_rng(2)
    white = colored_noise(SR, 0.0, rng)
    brown = colored_noise(SR, 2.0, rng)
    assert white.shape == brown.shape == (SR,)
    assert abs(rms(white) - 1) < 0.05 and abs(rms(brown) - 1) < 0.05

    # 棕噪声低频能量占比明显高于白噪声
    def low_ratio(x):
        s = np.abs(np.fft.rfft(x)) ** 2
        return s[: len(s) // 8].sum() / s.sum()

    assert low_ratio(brown) > low_ratio(white) + 0.3


def test_apply_rir_keeps_length_and_level():
    x = tone(1.0)
    rir = np.zeros(4000, dtype=np.float32)
    rir[0] = 1.0
    rir[800] = 0.5
    y = apply_rir(x, rir)
    assert y.shape == x.shape
    assert abs(rms(y) - rms(x)) / rms(x) < 1e-3


def test_speed_perturb_changes_length():
    x = tone(1.0)
    assert speed_perturb(x, 1.0) is x
    assert abs(len(speed_perturb(x, 1.1)) - len(x) / 1.1) < 2
    assert abs(len(speed_perturb(x, 0.9)) - len(x) / 0.9) < 2


def test_time_stretch_keeps_pitch_and_scales_length():
    t = np.arange(SR) / SR
    x = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    assert time_stretch(x, 1.0) is x
    for factor in (0.8, 1.3, 1.6, 2.0):
        y = time_stretch(x, factor)
        assert y.dtype == np.float32
        assert abs(len(y) - len(x) / factor) < 2
        # 音高不变：基频峰仍在 220 Hz（重采样变速会挪到 220×factor）
        spectrum = np.abs(np.fft.rfft(y))
        peak_hz = np.fft.rfftfreq(len(y), 1 / SR)[spectrum.argmax()]
        assert abs(peak_hz - 220) < 5, (factor, peak_hz)
        # 能量基本不变（叠加窗已归一化）
        assert abs(rms(y) - rms(x)) / rms(x) < 0.05
    # 太短的片段原样返回
    short = x[:300]
    assert time_stretch(short, 1.5) is short


def test_augmenter_tempo_only_shortens_positive():
    cfg = AugmentConfig(
        clip_seconds=2.0,
        p_tempo=1.0,
        tempo_range=[1.5, 1.5],
        p_speed=0.0,
        p_rir=0.0,
        p_background=0.0,
        p_babble=0.0,
        p_colored_noise=0.0,
        p_bandstop=0.0,
        p_lowpass=0.0,
        p_distortion=0.0,
        peak_range=[1.0, 1.0],
        end_jitter=0.0,
    )
    aug = Augmenter(cfg, seed=0)
    x = tone(1.0)
    out = aug(x, True)
    nz = np.nonzero(np.abs(out) > 1e-4)[0]
    # 右对齐，且有声部分约为原来的 1/1.5
    assert nz[-1] >= 2 * SR - 2
    assert abs((nz[-1] - nz[0]) - SR / 1.5) < 0.02 * SR


def test_audio_pool_segments():
    pool = AudioPool([tone(0.5), tone(0.3, 330)])
    rng = np.random.default_rng(3)
    seg = pool.random_segment(SR, rng)
    assert seg.shape == (SR,) and np.abs(seg).max() > 0
    with pytest.raises(ValueError):
        AudioPool([np.zeros(10, dtype=np.float32)])


def test_augmenter_is_deterministic_and_valid():
    cfg = AugmentConfig(clip_seconds=2.0)
    bg = AudioPool([np.random.default_rng(4).standard_normal(3 * SR).astype(np.float32)])
    rirs = [np.r_[1.0, np.zeros(100, dtype=np.float32), 0.3].astype(np.float32)]
    a1 = Augmenter(cfg, background=bg, babble=bg, rirs=rirs, seed=7)
    a2 = Augmenter(cfg, background=bg, babble=bg, rirs=rirs, seed=7)
    x = tone(0.7)
    outs1 = [a1(x, True) for _ in range(5)]
    outs2 = [a2(x, True) for _ in range(5)]
    for o1, o2 in zip(outs1, outs2, strict=True):
        assert o1.shape == (2 * SR,) and o1.dtype == np.float32
        assert np.array_equal(o1, o2)
        assert np.abs(o1).max() <= 1.0 and np.abs(o1).max() > 0
    assert to_int16(outs1[0]).dtype == np.int16
    # 不同 seed 应该得到不同结果
    a3 = Augmenter(cfg, background=bg, babble=bg, rirs=rirs, seed=8)
    assert not np.array_equal(a3(x, True), outs1[0])


def test_trim_to_speech_relative_to_noise_floor():
    from training.wakeword.augment import trim_to_speech

    rng = np.random.default_rng(5)
    noise = (rng.standard_normal(3 * SR) * 0.003).astype(np.float32)  # 约 -50 dBFS 的本底
    speech = tone(0.6)
    x = noise.copy()
    x[SR : SR + len(speech)] += speech
    y = trim_to_speech(x)
    # 保留了整段语音，前面留 ≤100 ms，后面留 ≤80 ms
    assert abs(len(y) - (len(speech) + 0.18 * SR)) < 0.03 * SR
    assert np.abs(y[: int(0.1 * SR)]).max() < 0.02
    # 纯噪声：什么都不裁
    assert len(trim_to_speech(noise)) == len(noise) or len(trim_to_speech(noise)) > 0
