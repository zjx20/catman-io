"""整条管线离线跑：真唤醒模型 + 真 silero + 假识别 / 假合成 / 内存扬声器。"""

import wave

import numpy as np
import pytest

from catman_io.asr import Transcript
from catman_io.audio.frames import read_wav
from catman_io.config import Config
from catman_io.dialog import Turn
from catman_io.pipeline import EchoResponder, Response, VoicePipeline, WavFrames
from catman_io.tts.speaker import ListOutput, Speaker
from catman_io.vad import SileroVAD
from catman_io.wakeword import WakeWordDetector, bundled_models, ensure_base_models


@pytest.fixture(scope="module")
def models():
    try:
        ensure_base_models(quiet=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"openWakeWord base models unavailable: {e}")
    if not bundled_models():
        pytest.skip("no bundled wake-word model")


class FakeASR:
    def __init__(self, text="而家幾點"):
        self.text = text
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        return Transcript(self.text, duration=len(audio) / 16000, elapsed=0.01)


class FakeTTS:
    def __init__(self):
        self.texts = []

    def synthesize(self, text):
        self.texts.append(text)
        return np.full(4800, 3000, dtype=np.int16)  # 0.3 s


class CountingResponder:
    def __init__(self, reply="宜家三點半。"):
        self.reply = reply
        self.turns = []
        self.cancelled = 0

    def respond(self, turn, transcript, speak):
        self.turns.append(transcript.text)
        speak(self.reply)
        return Response(info={"intent": "time.now"})

    def deliver(self, turn, speak):
        speak("時間到")
        return Response()

    def cancel(self):
        self.cancelled += 1


def make_wav(path, *parts):
    audio = np.concatenate(parts)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(audio.tobytes())
    return path


def build(tmp_path, wav, responder, asr, tts, **dialog):
    cfg = Config.load(None)
    cfg.data_dir = str(tmp_path)
    for k, v in dialog.items():
        setattr(cfg.dialog, k, v)
    out = ListOutput()
    speaker = Speaker(out, volume=1.0)
    ended = []
    pipe = VoicePipeline(
        cfg,
        frames=WavFrames(wav),
        detector=WakeWordDetector(),
        vad=SileroVAD(),
        speaker=speaker,
        responder=responder,
        asr=asr,
        tts=tts,
        on_turn=lambda turn, status: ended.append((turn, status)),
    )
    return pipe, out, ended


def test_wake_utterance_reply_end_to_end(tmp_path, models):
    wake = read_wav("tests/data/positive_siu_maau_jan_hiugaai.wav")
    sentence = read_wav("tests/data/negative_weather_wanlung.wav")
    gap = np.zeros(int(0.4 * 16000), dtype=np.int16)
    wav = make_wav(tmp_path / "in.wav", gap, wake, gap, sentence, gap)
    responder = CountingResponder()
    asr, tts = FakeASR("今日天氣點"), FakeTTS()
    pipe, out, ended = build(tmp_path, wav, responder, asr, tts, followup_seconds=1.0)
    pipe.run()
    assert responder.turns == ["今日天氣點"] and asr.calls == 1
    assert tts.texts == ["宜家三點半。"]
    assert [s for _, s in ended] == ["ok"]
    turn: Turn = ended[0][0]
    assert turn.info["asr_text"] == "今日天氣點" and turn.info["reply_text"] == "宜家三點半。"
    assert turn.info["intent"] == "time.now" and turn.info["t_first_audio"] is not None
    assert turn.wake_audio is not None and len(turn.wake_audio) > 16000
    assert turn.t_speech_end is not None and turn.info["t_first_audio"] >= turn.t_speech_end
    # 播出的声音里有唤醒提示音和一句回复（0.3 s）
    assert len(out.audio) >= 4800
    assert pipe.turns_done == 1


def test_no_speech_after_wake(tmp_path, models):
    wake = read_wav("tests/data/positive_siu_maau_jan_hiugaai.wav")
    gap = np.zeros(int(0.5 * 16000), dtype=np.int16)
    wav = make_wav(tmp_path / "in.wav", gap, wake, gap)
    responder = CountingResponder()
    pipe, out, ended = build(tmp_path, wav, responder, FakeASR(), FakeTTS(), followup_seconds=0.0)
    pipe.run()
    assert responder.turns == [] and [s for _, s in ended] == ["no_speech"]


def test_empty_transcript_reprompts_then_ends(tmp_path, models):
    wake = read_wav("tests/data/positive_siu_maau_jan_hiugaai.wav")
    sentence = read_wav("tests/data/negative_weather_wanlung.wav")
    gap = np.zeros(int(0.4 * 16000), dtype=np.int16)
    pause = np.zeros(int(1.2 * 16000), dtype=np.int16)  # 比尾静音 600 ms 长，才算两句
    wav = make_wav(tmp_path / "in.wav", gap, wake, gap, sentence, pause, sentence, gap)
    asr = FakeASR("")
    pipe, out, ended = build(tmp_path, wav, EchoResponder(), asr, FakeTTS(), followup_seconds=0.0)
    pipe.run()
    assert asr.calls == 2 and [s for _, s in ended] == ["empty"]


def test_echo_responder_and_max_seconds(tmp_path, models):
    gap = np.zeros(int(5 * 16000), dtype=np.int16)
    wav = make_wav(tmp_path / "in.wav", gap)
    pipe, out, ended = build(tmp_path, wav, EchoResponder(), FakeASR(), FakeTTS())
    pipe.max_seconds = 2.0
    pipe.run()
    assert pipe.frames_seen == 25 and ended == []


def test_timer_fires_through_the_pipeline(tmp_path, models):
    from catman_io.journal import Journal, JournalWriter
    from catman_io.responder import build_responder

    wake = read_wav("tests/data/positive_siu_maau_jan_hiugaai.wav")
    sentence = read_wav("tests/data/negative_weather_wanlung.wav")
    gap = np.zeros(int(0.4 * 16000), dtype=np.int16)
    wav = make_wav(tmp_path / "in.wav", gap, wake, gap, sentence, gap)
    cfg = Config.load(None)
    cfg.data_dir = str(tmp_path)
    cfg.dialog.followup_seconds = 1.0
    out = ListOutput()
    speaker = Speaker(out, volume=1.0)
    journal = Journal(tmp_path / "journal")
    responder = build_responder(cfg, speaker=speaker, journal=journal)
    writer = JournalWriter(journal, cfg)
    ended = []

    def on_turn(turn, status):
        writer(turn, status)
        ended.append((turn.kind, status))

    tts = FakeTTS()
    pipe = VoicePipeline(
        cfg,
        frames=WavFrames(wav),
        detector=WakeWordDetector(),
        vad=SileroVAD(),
        speaker=speaker,
        responder=responder,
        asr=FakeASR("計時兩秒"),
        tts=tts,
        on_turn=on_turn,
        timers=responder.ctx.timers,
    )
    responder.ctx.clock = pipe.clock
    pipe.run()
    assert tts.texts == ["好，兩秒後叫你。", "兩秒到喇"]
    assert ended == [("voice", "ok"), ("alarm", "ok")]
    recs = journal.records()
    assert [r["turn_kind"] for r in recs] == ["voice", "alarm"] and recs[0]["intent"] == "timer.set"
    assert (
        recs[0]["slots"] == {"duration": 2}
        and recs[0]["lat_first_audio_ms"] is not None
        and recs[0]["flags"] == []
    )
