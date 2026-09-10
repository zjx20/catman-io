"""语音流水线：采集 → 唤醒 → 端点 → 识别 → 应答（意图 / 动作 / 大脑）→ 合成 → 播放。

线程：
- 主线程跑帧循环：每帧过唤醒检测与 VAD，喂给 :class:`~catman_io.dialog.Dialog`，执行它吐出的命令；
  其他线程的事件从一个队列里排空后再喂状态机，所以状态机只有主线程碰。
- worker 线程做慢活：识别、应答、合成，通过队列收活、通过事件队列汇报（开口了 / 播完了 / 失败）。
- Speaker 自带写线程。

``--wav`` 模式用文件代替麦克风：时钟是"喂了多少帧"（每帧 80 ms），在 IDLE / LISTENING /
FOLLOWUP 时尽快喂，在等 worker（THINKING / SPEAKING）时按真实节奏喂，这样超时之类的时间逻辑仍然成立。
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from catman_io.asr import SpeechRecognizer, Transcript
from catman_io.audio.frames import FRAME_SAMPLES, FRAME_SECONDS, iter_frames, read_wav
from catman_io.config import Config
from catman_io.dialog import (
    Beep,
    Cancel,
    Deliver,
    Dialog,
    Event,
    ReplyDone,
    ReplyStarted,
    StartTurn,
    State,
    StopSpeaking,
    TimerFired,
    Transcribe,
    TranscriptEmpty,
    Turn,
    TurnEnded,
    TurnFailed,
)
from catman_io.tts import Synthesizer, clean_for_speech, split_sentences
from catman_io.tts.earcons import earcon
from catman_io.tts.speaker import Speaker
from catman_io.vad import SileroVAD
from catman_io.wakeword import WakeWordDetector

log = logging.getLogger(__name__)

ERROR_PHRASE = "唔好意思，出咗啲問題。"


@dataclass
class Response:
    """应答器一回合的结果（写日志用）。"""

    ok: bool = True
    error: str | None = None
    followup: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class ClockRef:
    """可以事后换掉的钟：部件先拿着它，pipeline 建好后再决定是墙钟还是音频时钟。"""

    def __init__(self, fn: Callable[[], float] = time.monotonic):
        self.fn = fn

    def __call__(self) -> float:
        return self.fn()


class Responder(Protocol):
    """听清一句话之后该做什么。说话通过 ``speak(text)`` 回调（可多次，按句播）。"""

    def respond(self, turn: Turn, transcript: Transcript, speak: Callable[[str], bool]) -> Response: ...

    def deliver(self, turn: Turn, speak: Callable[[str], bool]) -> Response:
        """定时器到点之类的主动播报。"""

    def cancel(self) -> None: ...


class EchoResponder:
    """骨架用：把听到的复述一遍。"""

    def respond(self, turn: Turn, transcript: Transcript, speak: Callable[[str], bool]) -> Response:
        speak(f"你話：{transcript.text}")
        return Response(info={"intent": "echo"})

    def deliver(self, turn: Turn, speak: Callable[[str], bool]) -> Response:
        speak(str(turn.payload.get("say", "時間到喇")) if isinstance(turn.payload, dict) else "時間到喇")
        return Response()

    def cancel(self) -> None:
        pass


class WavFrames:
    """用 WAV 文件代替麦克风：切帧，末尾补几秒静音让端点器收口，之后一直给静音直到调用方停。"""

    def __init__(self, path: str | Path, *, channel: int = 0, tail_seconds: float = 3.0):
        self.path, self.channel, self.tail_seconds = Path(path), channel, tail_seconds
        self.exhausted = False

    def __iter__(self) -> Iterator[np.ndarray]:
        audio = read_wav(self.path, channel=self.channel)
        tail = np.zeros(int(self.tail_seconds * 16000), dtype=np.int16)
        yield from iter_frames(np.concatenate([audio, tail]))
        self.exhausted = True
        while True:
            yield np.zeros(FRAME_SAMPLES, dtype=np.int16)


class VoicePipeline:
    def __init__(
        self,
        cfg: Config,
        *,
        frames: Any,
        detector: WakeWordDetector,
        vad: SileroVAD,
        speaker: Speaker,
        responder: Responder,
        asr: SpeechRecognizer | None = None,
        tts: Synthesizer | None = None,
        clock: Callable[[], float] | None = None,
        on_turn: Callable[[Turn, str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        stop_source: Callable[[], None] | None = None,
        realtime: bool = False,
        max_seconds: float | None = None,
        timers: Any = None,
        clock_ref: ClockRef | None = None,
        api: Any = None,
    ):
        self.cfg = cfg
        self.frames = frames
        self.detector = detector
        self.vad = vad
        self.speaker = speaker
        self.responder = responder
        self.asr = asr
        self.tts = tts
        self.on_turn = on_turn
        self.on_status = on_status
        self.stop_source = stop_source
        self.realtime = realtime
        self.max_seconds = max_seconds
        self.dialog = Dialog(cfg.dialog, cfg.vad)
        self.file_mode = isinstance(frames, WavFrames)
        self.frames_seen = 0
        self.clock = clock or (self._audio_clock if self.file_mode else time.monotonic)
        if clock_ref is not None:
            clock_ref.fn = self.clock
        self.speaker.clock = self.clock  # 首包时间要和对话时间戳用同一个钟
        if hasattr(self.responder, "clock"):
            self.responder.clock = self.clock
        self.timers = timers
        self.api = api  # ApiServer，run() 时启动、结束时停
        self.events: queue.Queue[Event] = queue.Queue()
        self.jobs: queue.Queue[tuple[str, Turn, Any] | None] = queue.Queue()
        ring_frames = max(1, int(cfg.dialog.wake_context_seconds / FRAME_SECONDS))
        self._ring: deque[np.ndarray] = deque(maxlen=ring_frames)
        self._stop = threading.Event()
        self._worker_busy = False
        self._worker = threading.Thread(target=self._worker_loop, name="responder", daemon=True)
        self.last_text = ""
        self.turns_done = 0

    # ---- 对外 ----

    def stop(self) -> None:
        self._stop.set()
        if self.stop_source is not None:
            self.stop_source()

    def status(self) -> dict[str, Any]:
        return {
            "state": self.dialog.state.value,
            "turns": self.turns_done,
            "last_text": self.last_text,
            "frames": self.frames_seen,
            "dropped": getattr(self.frames, "dropped", None),
        }

    def run(self) -> None:
        self._worker.start()
        if self.api is not None:
            self.api.start()
        try:
            for frame in self.frames:
                if self._stop.is_set():
                    break
                self.frames_seen += 1
                self._step(frame)
                if self._should_end():
                    break
        finally:
            self._shutdown()

    # ---- 主线程 ----

    def _audio_clock(self) -> float:
        return self.frames_seen * FRAME_SECONDS

    def _step(self, frame: np.ndarray) -> None:
        t_frame = time.perf_counter()
        now = self.clock()
        self._ring.append(frame)
        dets = self.detector.process(frame)
        prob = self.vad.process(frame)
        cmds = []
        if self.timers is not None:
            for t in self.timers.poll(now):
                self.events.put(TimerFired(t))
        while True:
            try:
                ev = self.events.get_nowait()
            except queue.Empty:
                break
            cmds += self.dialog.on_event(now, ev)
        cmds += self.dialog.on_frame(now, frame, dets, prob)
        for cmd in cmds:
            self._execute(cmd, now)
        if self.on_status is not None:
            score = max(self.detector.last_scores.values()) if self.detector.last_scores else 0.0
            self.on_status(f"{self.dialog.state.value:<9} wake={score:4.2f} vad={prob:4.2f} {self.last_text}")
        waiting = self.dialog.state in (State.THINKING, State.SPEAKING)
        if self.realtime or (self.file_mode and waiting):
            time.sleep(max(0.0, FRAME_SECONDS - (time.perf_counter() - t_frame)))

    def _execute(self, cmd: Any, now: float) -> None:
        if isinstance(cmd, Beep):
            if self.cfg.dialog.earcons:
                self.speaker.say(earcon(cmd.kind), gen=None)
        elif isinstance(cmd, StartTurn):
            self.speaker.set_gen(cmd.turn.gen)
            if self._ring:
                cmd.turn.wake_audio = np.concatenate(list(self._ring))
            log.info("turn %s start (gen %d, followup=%s)", cmd.turn.id, cmd.turn.gen, cmd.turn.followup)
        elif isinstance(cmd, Transcribe):
            self.jobs.put(("transcribe", cmd.turn, None))
        elif isinstance(cmd, Cancel):
            cmd.turn.cancelled.set()
            self.responder.cancel()
            self.speaker.stop()
        elif isinstance(cmd, StopSpeaking):
            self.speaker.stop()
        elif isinstance(cmd, TurnEnded):
            self.jobs.put(("finish", cmd.turn, cmd.status))
        elif isinstance(cmd, Deliver):
            self.jobs.put(("deliver", cmd.turn, None))

    def _should_end(self) -> bool:
        if self.max_seconds is not None and self.frames_seen * FRAME_SECONDS >= self.max_seconds:
            return True
        if self.file_mode and self.frames.exhausted and self.dialog.state == State.IDLE:
            return self.jobs.empty() and not self._worker_busy
        return False

    def _shutdown(self) -> None:
        if self.stop_source is not None:
            self.stop_source()
        if self.dialog.turn is not None:
            self.dialog.turn.cancelled.set()
        self.jobs.put(None)
        self._worker.join(timeout=5.0)
        self.speaker.wait(timeout=2.0)
        self.speaker.close()
        if self.api is not None:
            self.api.stop()

    # ---- worker 线程 ----

    def _worker_loop(self) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                return
            kind, turn, arg = job
            self._worker_busy = True
            try:
                if kind == "transcribe":
                    self._do_transcribe(turn)
                elif kind == "deliver":
                    self._do_deliver(turn)
                elif kind == "finish":
                    self._do_finish(turn, arg)
            except Exception:  # noqa: BLE001
                log.exception("worker job %s failed for turn %s", kind, turn.id)
                if kind != "finish":
                    self.events.put(TurnFailed(turn.id, "worker error"))
            finally:
                self._worker_busy = not self.jobs.empty()

    def _speak_for(self, turn: Turn) -> Callable[[str], bool]:
        def speak(text: str) -> bool:
            spoke = False
            for sentence in split_sentences(clean_for_speech(text)):
                if turn.cancelled.is_set():
                    break
                turn.info.setdefault("reply_text", "")
                if self.tts is None:
                    log.info("(no tts) %s", sentence)
                    turn.info["reply_text"] += sentence
                    if not spoke:
                        self.events.put(ReplyStarted(turn.id))
                    spoke = True
                    continue
                t0 = time.perf_counter()
                pcm = self.tts.synthesize(sentence)
                turn.info.setdefault("tts_seconds", 0.0)
                turn.info["tts_seconds"] += time.perf_counter() - t0
                if not self.speaker.say(pcm, gen=turn.gen):
                    break
                turn.info["reply_text"] += sentence
                if not spoke:
                    self.events.put(ReplyStarted(turn.id))
                spoke = True
            return spoke

        return speak

    def _do_transcribe(self, turn: Turn) -> None:
        if turn.cancelled.is_set() or turn.audio is None:
            return
        t0 = time.perf_counter()
        transcript = self.asr.transcribe(turn.audio) if self.asr is not None else Transcript("")
        turn.info["asr_seconds"] = time.perf_counter() - t0
        turn.info["t_asr_done"] = self.clock()
        turn.info["asr_text"] = transcript.text
        turn.info["asr_lang"] = transcript.language
        self.last_text = transcript.text
        log.info("turn %s asr %.2fs: %r", turn.id, turn.info["asr_seconds"], transcript.text)
        if turn.cancelled.is_set():
            return
        if not transcript.text.strip():
            self.events.put(TranscriptEmpty(turn.id))
            return
        self._respond(turn, lambda speak: self.responder.respond(turn, transcript, speak))

    def _do_deliver(self, turn: Turn) -> None:
        self._respond(turn, lambda speak: self.responder.deliver(turn, speak))

    def _respond(self, turn: Turn, fn: Callable[[Callable[[str], bool]], Response]) -> None:
        speak = self._speak_for(turn)
        try:
            resp = fn(speak)
        except Exception as e:  # noqa: BLE001
            log.exception("responder failed for turn %s", turn.id)
            resp = Response(ok=False, error=str(e))
            if not turn.cancelled.is_set():
                speak(ERROR_PHRASE)
        turn.info.update(resp.info)
        turn.info["ok"] = resp.ok
        if resp.error:
            turn.info["error"] = resp.error
        if turn.cancelled.is_set():
            return
        self.speaker.wait(timeout=120.0)
        if "reply_text" in turn.info:
            self.events.put(ReplyDone(turn.id, followup=resp.followup))
        elif resp.ok:
            self.events.put(ReplyDone(turn.id, followup=resp.followup))  # 有动作无话说
        else:
            self.events.put(TurnFailed(turn.id, resp.error or "no reply"))

    def _do_finish(self, turn: Turn, status: str) -> None:
        turn.info["t_first_audio"] = self.speaker.first_audio.get(turn.gen)
        self.turns_done += 1
        log.info("turn %s ended: %s %s", turn.id, status, _latency_summary(turn))
        if self.on_turn is not None:
            self.on_turn(turn, status)


def _latency_summary(turn: Turn) -> str:
    parts = []
    if turn.t_speech_end is not None and turn.info.get("t_first_audio") is not None:
        parts.append(f"first_audio={turn.info['t_first_audio'] - turn.t_speech_end:.2f}s")
    for k in ("asr_seconds", "tts_seconds"):
        if k in turn.info:
            parts.append(f"{k}={turn.info[k]:.2f}")
    return " ".join(parts)


def build_pipeline(
    cfg: Config,
    *,
    input_wav: str | Path | None = None,
    output_wav: str | Path | None = None,
    responder: Responder | None = None,
    realtime: bool = False,
    max_seconds: float | None = None,
    on_status: Callable[[str], None] | None = None,
    on_turn: Callable[[Turn, str], None] | None = None,
    model_paths: list[str] | None = None,
    api: bool = False,
) -> VoicePipeline:
    """按配置把各部件装起来。识别器 / 合成器缺失时降级（只切句不识别 / 只打日志不出声）并警告。"""
    from catman_io.journal import JournalWriter, build_journal
    from catman_io.tts import create_synthesizer
    from catman_io.tts.speaker import ListOutput, SoundDeviceOutput, WavOutput

    clock_ref = ClockRef()

    w = cfg.wakeword
    detector = WakeWordDetector(
        model_paths or w.models or None,
        threshold=w.threshold,
        patience=w.patience,
        cooldown=w.cooldown,
        vad_threshold=w.vad_threshold,
    )
    vad = SileroVAD(threads=cfg.vad.threads)
    asr = None
    if cfg.asr.backend != "none":
        try:
            from catman_io.asr import create_recognizer

            asr = create_recognizer(cfg)
            if hasattr(asr, "warmup"):
                asr.warmup()
        except Exception as e:  # noqa: BLE001
            log.warning("asr disabled: %s", e)
    tts = None
    try:
        tts = create_synthesizer(cfg)
    except Exception as e:  # noqa: BLE001
        log.warning("tts disabled: %s", e)
    if output_wav is not None:
        output: Any = WavOutput(output_wav)
    elif input_wav is not None:
        output = ListOutput()
    else:
        output = SoundDeviceOutput(
            cfg.audio.output_device, blocksize=cfg.speaker.blocksize, latency=cfg.speaker.latency
        )
    speaker = Speaker(output, volume=cfg.speaker.volume, blocksize=cfg.speaker.blocksize)
    journal = build_journal(cfg)
    timers = None
    if responder is None:
        from catman_io.responder import build_responder

        responder = build_responder(cfg, speaker=speaker, journal=journal, clock=clock_ref)
        timers = responder.ctx.timers
        if tts is not None and hasattr(tts, "prewarm"):
            threading.Thread(target=_prewarm, args=(tts, responder), name="tts-prewarm", daemon=True).start()
    user_on_turn = on_turn
    writer = JournalWriter(journal, cfg) if journal is not None else None

    def on_turn(turn: Turn, status: str) -> None:  # type: ignore[no-redef]
        if writer is not None:
            try:
                writer(turn, status)
            except Exception:  # noqa: BLE001
                log.exception("journal write failed for %s", turn.id)
        if user_on_turn is not None:
            user_on_turn(turn, status)

    stop_source = None
    if input_wav is not None:
        frames: Any = WavFrames(input_wav, channel=cfg.audio.channel)
    else:
        from catman_io.audio.capture import MicCapture

        a = cfg.audio
        mic = MicCapture(device=a.device, channel=a.channel, sample_rate=a.sample_rate).start()
        frames = mic.frames()
        stop_source = mic.stop
    pipe = VoicePipeline(
        cfg,
        frames=frames,
        detector=detector,
        vad=vad,
        speaker=speaker,
        responder=responder,
        asr=asr,
        tts=tts,
        on_turn=on_turn,
        on_status=on_status,
        stop_source=stop_source,
        realtime=realtime,
        max_seconds=max_seconds,
        timers=timers,
        clock_ref=clock_ref,
    )
    if api and cfg.api.enabled and hasattr(responder, "router"):
        try:
            from catman_io.api import ApiServer, make_app, resolve_api_token

            app = make_app(
                cfg=cfg,
                store=responder.router.store,
                journal=journal,
                token=resolve_api_token(cfg),
                status=pipe.status,
            )
            pipe.api = ApiServer(app, cfg.api.host, cfg.api.port)
        except ImportError as e:
            log.warning("api disabled: %s (pip install 'catman-io[demo]')", e)
    return pipe


def _prewarm(tts: Any, responder: Any) -> None:
    """启动时把固定短语合成进缓存，让规则回复的首包接近零延迟。"""
    from catman_io import responder as r

    phrases = [ERROR_PHRASE, r.NO_BRAIN_PHRASE, r.BRAIN_ERROR_PHRASE, r.DELEGATED_PHRASE, r.NO_CATMAN_PHRASE]
    phrases += [
        "大聲咗。",
        "細聲咗。",
        "取消咗。",
        "而家冇計時緊。",
        "頭先冇講嘢。",
        "要計幾耐呀？",
        "時間到喇",
    ]
    try:
        n = tts.prewarm(phrases)
        if n:
            log.info("tts prewarm: %d phrase(s) synthesized", n)
    except Exception as e:  # noqa: BLE001
        log.warning("tts prewarm failed: %s", e)
