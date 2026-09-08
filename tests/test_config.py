import pytest

from catman_io.config import Config
from training.wakeword.config import TrainingConfig


def test_training_config_defaults_and_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model_name: x\ntrain:\n  steps: 5\n", encoding="utf-8")
    cfg = TrainingConfig.load(p)
    assert cfg.model_name == "x" and cfg.train.steps == 5
    assert cfg.tts.voices[0].startswith("zh-HK")
    assert cfg.data.resolved_positive_phrases()
    assert cfg.export_dir == cfg.work / "export"


def test_training_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("train:\n  stepz: 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        TrainingConfig.load(p)


def test_runtime_config(tmp_path):
    assert Config.load(None).wakeword.threshold == 0.5
    p = tmp_path / "r.yaml"
    p.write_text("wakeword:\n  threshold: 0.7\naudio:\n  channel: 1\n", encoding="utf-8")
    cfg = Config.load(p)
    assert cfg.wakeword.threshold == 0.7 and cfg.audio.channel == 1
    p.write_text("nope: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        Config.load(p)
