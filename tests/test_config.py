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


def test_runtime_config_nested_sections_and_paths(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text(
        "data_dir: /tmp/x\nvad:\n  trailing_silence_ms: 700\n"
        "intent:\n  llm:\n    enabled: true\n    model: m\n"
        "actions:\n  weather:\n    latitude: 22\n",
        encoding="utf-8",
    )
    cfg = Config.load(p)
    assert cfg.vad.trailing_silence_ms == 700 and cfg.vad.start_threshold == 0.6
    assert cfg.intent.llm.enabled and cfg.intent.llm.model == "m" and cfg.intent.llm.tool_choice == "auto"
    assert cfg.actions.weather.latitude == 22.0 and isinstance(cfg.actions.weather.latitude, float)
    assert str(cfg.rules_dir) == "/tmp/x/intent/rules" and str(cfg.journal_dir) == "/tmp/x/journal"
    assert str(Config.load(None).cases_path) == "data/intent/cases.jsonl"


def test_runtime_config_errors_name_the_path(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("intent:\n  llm:\n    modle: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"<root>\.intent\.llm: unknown keys \['modle'\]"):
        Config.load(p)
    p.write_text("vad:\n  start_threshold: high\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"<root>\.vad\.start_threshold: expected float"):
        Config.load(p)
    p.write_text("vad:\n  start_frames: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected int"):
        Config.load(p)
    p.write_text("audio:\n  device: 3\nwakeword:\n  models: [a.onnx]\n", encoding="utf-8")
    cfg = Config.load(p)
    assert cfg.audio.device == 3 and cfg.wakeword.models == ["a.onnx"]


def test_secret_reads_env(monkeypatch):
    from catman_io.config import ConfigError, secret

    monkeypatch.delenv("CATMAN_IO_TEST_KEY", raising=False)
    assert secret("CATMAN_IO_TEST_KEY", required=False) is None
    with pytest.raises(ConfigError, match="CATMAN_IO_TEST_KEY"):
        secret("CATMAN_IO_TEST_KEY", what="LLM")
    monkeypatch.setenv("CATMAN_IO_TEST_KEY", "abc")
    assert secret("CATMAN_IO_TEST_KEY") == "abc"
