"""ASR：模型登记表、安装检查、错误提示；有模型时（环境变量指路）跑一次真识别。"""

import os
from pathlib import Path

import pytest

from catman_io.asr import Transcript, create_recognizer
from catman_io.asr.models import ASR_MODELS, DEFAULT_MODEL, is_installed, model_path, resolve_model
from catman_io.config import Config


def test_registry_and_defaults():
    for backend, name in DEFAULT_MODEL.items():
        m = resolve_model(backend)
        assert m.name == name and m.url.endswith(".tar.bz2")
    assert resolve_model("sensevoice", "wenet-yue-2025-09").kind == "wenet_ctc"
    with pytest.raises(ValueError, match="unknown ASR model"):
        resolve_model("sensevoice", "nope")
    assert set(ASR_MODELS) == {"sense-voice-2025-09", "sense-voice-2024-07", "wenet-yue-2025-09"}


def test_is_installed_needs_both_files(tmp_path):
    m = resolve_model("wenet_yue")
    assert not is_installed(tmp_path, m)
    d = model_path(tmp_path, m)
    d.mkdir(parents=True)
    (d / m.model_file).write_bytes(b"x")
    assert not is_installed(tmp_path, m)
    (d / m.tokens_file).write_text("a\n")
    assert is_installed(tmp_path, m)


def test_create_recognizer_errors_are_actionable(tmp_path):
    cfg = Config.load(None)
    cfg.data_dir = str(tmp_path)
    with pytest.raises(RuntimeError, match="setup --asr"):
        create_recognizer(cfg)
    cfg.asr.backend = "none"
    with pytest.raises(RuntimeError, match="none"):
        create_recognizer(cfg)


def test_transcript_rtf():
    assert Transcript("x", elapsed=0.5, duration=2.0).rtf == 0.25
    assert Transcript("x").rtf == 0.0


ROOT = os.environ.get("CATMAN_IO_TEST_ASR_ROOT")


@pytest.mark.skipif(not ROOT, reason="set CATMAN_IO_TEST_ASR_ROOT to a dir with downloaded models")
@pytest.mark.parametrize("backend", ["sensevoice", "wenet_yue"])
def test_real_recognition(backend):
    pytest.importorskip("sherpa_onnx")
    from catman_io.audio.frames import read_wav

    cfg = Config.load(None)
    cfg.asr.backend = backend
    cfg.asr.model_dir = ROOT
    if not is_installed(Path(ROOT), resolve_model(backend)):
        pytest.skip("model not downloaded")
    rec = create_recognizer(cfg)
    tr = rec.transcribe(read_wav("tests/data/negative_weather_wanlung.wav"))
    assert "天" in tr.text and tr.duration > 0.5 and tr.elapsed > 0
