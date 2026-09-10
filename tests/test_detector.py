"""检测器测试。需要 openWakeWord 的基础模型（首次运行会联网下载约 5 MB），下载不了就跳过。"""

import numpy as np
import pytest

from catman_io.audio.frames import FRAME_SAMPLES
from catman_io.wakeword import Detection, WakeWordDetector, bundled_models, ensure_base_models


@pytest.fixture(scope="module")
def base_models():
    try:
        ensure_base_models(quiet=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"openWakeWord base models unavailable: {e}")


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """一个随机初始化的分类器，只用来测流式接口。"""
    torch = pytest.importorskip("torch")
    from training.wakeword.model import WakeWordNet, export_onnx

    torch.manual_seed(0)
    return export_onnx(WakeWordNet(layer_size=8).eval(), tmp_path_factory.mktemp("m") / "tiny.onnx")


def test_process_accepts_int16_and_float(base_models, tiny_model):
    det = WakeWordDetector([tiny_model], threshold=2.0)  # 阈值 2 永远不触发
    assert det.names == ["tiny"]
    rng = np.random.default_rng(0)
    for _ in range(10):
        assert det.process(rng.integers(-3000, 3000, FRAME_SAMPLES, dtype=np.int16)) == []
    assert det.process(rng.uniform(-0.1, 0.1, FRAME_SAMPLES).astype(np.float32)) == []
    assert 0.0 <= det.last_scores["tiny"] <= 1.0
    assert det.stream_time == pytest.approx(11 * 0.08)
    det.reset()
    assert det.stream_time == 0


def test_cooldown_and_patience(base_models, tiny_model):
    det = WakeWordDetector([tiny_model], threshold=0.0, patience=3, cooldown=1.0)
    rng = np.random.default_rng(1)
    fired = []
    for _ in range(30):
        fired += det.process(rng.integers(-3000, 3000, FRAME_SAMPLES, dtype=np.int16))
    # 前 5 帧 openWakeWord 输出恒为 0，之后每帧都 ≥ 0 → patience 满足；冷却 1 s = 12.5 帧
    assert fired and all(isinstance(d, Detection) for d in fired)
    times = [d.stream_time for d in fired]
    assert all(b - a >= 1.0 - 1e-9 for a, b in zip(times[:-1], times[1:], strict=True))


@pytest.mark.skipif(not bundled_models(), reason="no bundled wake-word model")
def test_bundled_model_on_sample_clips(base_models):
    """随包模型应该认出合成的「小貓人」（正常语速和 +100% 快语速），并且不被静音 / 「小貓」/ 日常句子触发。"""
    from pathlib import Path

    from catman_io.audio.frames import read_wav

    data = Path(__file__).parent / "data"

    def padded(name: str) -> np.ndarray:
        silence = np.zeros(16000, dtype=np.int16)
        return np.concatenate([silence, read_wav(data / name), silence])

    det = WakeWordDetector(threshold=0.5)
    for name in ("positive_siu_maau_jan_hiugaai.wav", "positive_siu_maau_jan_fast_wanlung.wav"):
        det.reset()
        assert det.process_audio(padded(name)), f"should detect the wake word in {name}"
    for name in ("negative_siu_maau_hiugaai.wav", "negative_weather_wanlung.wav"):
        det.reset()
        assert det.process_audio(padded(name)) == [], f"should not trigger on {name}"
    det.reset()
    assert det.process_audio(np.zeros(16000 * 3, dtype=np.int16)) == []
