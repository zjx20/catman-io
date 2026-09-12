"""网页版对话 demo：浏览器麦克风 → 整条管线（唤醒 → 提示音 → 聆听 → 识别 → 应答 → 合成）→ 浏览器扬声器。

一个 ``/ws/dialog`` 连接 = 一条 :class:`~catman_io.pipeline.VoicePipeline`（和 ``catman-io run`` 同一套部件，
只是麦克风和声卡换成了浏览器）：

- 浏览器 → 服务端：二进制帧是 16 kHz 单声道 int16 PCM（长度任意）；文本帧是 JSON：
  ``{"type":"wake"}``（按钮唤醒 = 听到了唤醒词，播报中按就是打断）、``{"type":"stop"}``（停止播报）
- 服务端 → 浏览器：二进制帧是要播的 16 kHz int16 PCM（提示音、合成语音；按实时节奏送，最多提前 0.25 s）；
  文本帧：``{"type":"hello",...各部件是否就绪}``、每收到一块音频回一条
  ``{"type":"meter",t,state,wake,vad,level_db}``、状态变化 ``{"type":"state",state,t}``、
  一回合结束 ``{"type":"turn",...识别文本 / 意图 / 回复 / 各段延迟}``、
  打断时 ``{"type":"audio_stop"}``（浏览器把排队未播的音频丢掉）、``{"type":"error",message}``

所有发往浏览器的东西都经过一个队列由单个任务发送，管线线程只往队列里放。
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from catman_io.audio.frames import FRAME_SAMPLES, SAMPLE_RATE, to_int16
from catman_io.config import Config
from catman_io.dialog import State, Turn

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
LEAD_SECONDS = 0.25  # 音频最多比实时提前送这么多：打断时浏览器里最多只有这么多要丢


def level_db(pcm: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))) if len(pcm) else 0.0
    return 20 * float(np.log10(rms / 32768 + 1e-9))


class QueueFrames:
    """浏览器送来的 PCM → 80 ms 帧的迭代器。管线线程在 ``__iter__`` 上阻塞等帧；``close()`` 让它结束。"""

    def __init__(self, maxsize: int = 256):
        self._q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=maxsize)
        self._pending = np.zeros(0, dtype=np.int16)
        self.dropped = 0

    def feed(self, pcm: np.ndarray) -> int:
        pcm = np.asarray(pcm, dtype=np.int16)
        self._pending = np.concatenate([self._pending, pcm]) if len(self._pending) else pcm
        n = 0
        while len(self._pending) >= FRAME_SAMPLES:
            frame, self._pending = self._pending[:FRAME_SAMPLES], self._pending[FRAME_SAMPLES:]
            self._put(frame)
            n += 1
        return n

    def _put(self, item: np.ndarray | None) -> None:
        try:
            self._q.put_nowait(item)
        except queue.Full:  # 管线跟不上：丢最旧的，绝不堵住 WebSocket
            self.dropped += 1
            try:
                self._q.get_nowait()
                self._q.put_nowait(item)
            except (queue.Empty, queue.Full):
                pass

    def close(self) -> None:
        self._put(None)

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            frame = self._q.get()
            if frame is None:
                return
            yield frame


class WsOutput:
    """Speaker 的输出端：音频送去浏览器播。

    按实时节奏送（最多提前 ``LEAD_SECONDS``），这样 ``speaker.wait()`` 返回的时刻、打断时要丢掉的量
    都和真声卡差不多；``flush()`` 是 ``Speaker.stop()`` 时叫浏览器把已经送过去、还没播的那一点也扔掉。
    """

    keepalive = False

    def __init__(self, send: Callable[[bytes | dict], None]):
        self._send = send
        self._t = 0.0  # 已送出的音频会在这个时刻（monotonic）播完

    def write(self, pcm: np.ndarray) -> None:
        now = time.monotonic()
        if self._t < now:
            self._t = now
        self._send(to_int16(pcm).tobytes())
        self._t += len(pcm) / SAMPLE_RATE
        ahead = self._t - now
        if ahead > LEAD_SECONDS:
            time.sleep(ahead - LEAD_SECONDS)

    def flush(self) -> None:
        self._t = 0.0
        self._send({"type": "audio_stop"})

    def close(self) -> None:
        pass


class DialogSession:
    """一个浏览器连接：建整条管线，在后台线程跑 ``pipeline.run()``。

    浏览器的音频从 ``feed()`` 进，要播的音频和事件进 outbox。构造要加载识别模型等，会花几秒，
    调用方放到线程池里做。
    """

    def __init__(
        self,
        cfg: Config,
        loop: asyncio.AbstractEventLoop,
        outbox: asyncio.Queue,
        model_paths: list[str] | None = None,
    ):
        from catman_io.pipeline import build_pipeline

        self.cfg, self.loop, self.outbox = cfg, loop, outbox
        self.frames = QueueFrames()
        self.t0 = time.monotonic()
        self.pipe = build_pipeline(
            cfg,
            frames=self.frames,
            output=WsOutput(self.post),
            stop_source=self.frames.close,
            on_turn=self._on_turn,
            model_paths=model_paths,
            api=False,
        )
        self.pipe.dialog.on_state = self._on_state
        self.thread = threading.Thread(target=self.pipe.run, name="webdemo-pipeline", daemon=True)
        self.thread.start()

    # ---- 管线线程 / worker 线程 → asyncio：只碰 outbox，而且经 call_soon_threadsafe ----

    def post(self, item: bytes | dict) -> None:
        self.loop.call_soon_threadsafe(self.outbox.put_nowait, item)

    def _on_state(self, state: State, now: float) -> None:
        self.post({"type": "state", "state": state.value, "t": round(now - self.t0, 2)})

    def _on_turn(self, turn: Turn, status: str) -> None:
        info = turn.info

        def rel(a: float | None, b: float | None) -> float | None:
            return None if a is None or b is None else round(a - b, 2)

        self.post(
            {
                "type": "turn",
                "id": turn.id,
                "status": status,
                "kind": turn.kind,
                "followup": turn.followup,
                "barge_in": turn.barge_in,
                "reprompted": turn.reprompted,
                "wake_model": turn.wake_model,
                "wake_score": None if turn.wake_score is None else round(float(turn.wake_score), 3),
                "text": info.get("asr_text", ""),
                "intent": info.get("intent"),
                "reply": info.get("reply_text", ""),
                "error": info.get("error"),
                "latency": {
                    "speech_start": rel(turn.t_speech_start, turn.t_wake),  # 唤醒后多久开口
                    "speech_end": rel(turn.t_speech_end, turn.t_wake),  # 唤醒后多久说完
                    "asr": round(info["asr_seconds"], 2) if "asr_seconds" in info else None,
                    "first_audio": rel(info.get("t_first_audio"), turn.t_speech_end),  # 说完到出声
                    "reply_done": rel(turn.t_reply_done, turn.t_wake),  # 全程
                    "tts": round(info["tts_seconds"], 2) if "tts_seconds" in info else None,
                },
            }
        )

    # ---- asyncio 线程调用 ----

    def hello(self) -> dict[str, Any]:
        p, cfg = self.pipe, self.cfg
        router = getattr(p.responder, "router", None)
        return {
            "type": "hello",
            "models": p.detector.names,
            "threshold": p.detector.threshold,
            "asr": p.asr is not None,
            "asr_backend": cfg.asr.backend,
            "tts": p.tts is not None,
            "tts_voice": cfg.tts.voice if p.tts is not None else "",
            "intents": len(router.rules.intents) if router is not None else 0,
            "llm": bool(router is not None and getattr(router, "llm", None)),
            "brain": getattr(p.responder, "brain", None) is not None,
            "catman": getattr(p.responder, "catman", None) is not None,
            "earcons": cfg.dialog.earcons,
            "followup_seconds": cfg.dialog.followup_seconds,
            "no_speech_timeout_ms": cfg.vad.no_speech_timeout_ms,
            "trailing_silence_ms": cfg.vad.trailing_silence_ms,
        }

    def feed(self, pcm: np.ndarray) -> dict[str, Any]:
        """喂一块音频，返回这一刻的仪表读数（管线在另一个线程里处理，读数是它最近处理完的那一帧）。"""
        self.frames.feed(pcm)
        m = self.pipe.meter
        return {
            "type": "meter",
            "t": round(time.monotonic() - self.t0, 2),
            "state": m["state"],
            "wake": round(float(m["wake"]), 3),
            "vad": round(float(m["vad"]), 3),
            "level_db": round(level_db(pcm), 1),
            "dropped": self.frames.dropped,
        }

    def wake(self) -> None:
        self.pipe.wake_now()

    def stop_speaking(self) -> None:
        self.pipe.speaker.stop()

    def close(self) -> None:
        self.pipe.stop()
        self.thread.join(timeout=10.0)


def add_dialog_routes(app: Any, cfg: Config, model_paths: list[str] | None = None) -> None:
    from aiohttp import WSMsgType, web

    async def page(request):
        return web.FileResponse(STATIC_DIR / "dialog.html", headers={"Cache-Control": "no-store"})

    async def ws_handler(request):
        ws = web.WebSocketResponse(max_msg_size=4 << 20, heartbeat=20)
        await ws.prepare(request)
        loop = asyncio.get_running_loop()
        outbox: asyncio.Queue = asyncio.Queue()
        try:  # 建管线要加载识别模型等，几秒钟，别卡住事件循环
            session = await loop.run_in_executor(None, DialogSession, cfg, loop, outbox, model_paths)
        except Exception as e:  # noqa: BLE001
            log.exception("cannot build the pipeline for %s", request.remote)
            await ws.send_json({"type": "error", "message": f"cannot build the pipeline: {e}"})
            await ws.close()
            return ws
        outbox.put_nowait(session.hello())
        log.info("dialog client connected: %s", request.remote)

        async def pump() -> None:
            while True:
                item = await outbox.get()
                if item is None:
                    return
                if isinstance(item, bytes):
                    await ws.send_bytes(item)
                else:
                    await ws.send_json(item)

        pump_task = asyncio.create_task(pump())
        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    outbox.put_nowait(session.feed(np.frombuffer(msg.data, dtype="<i2")))
                elif msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        kind = data.get("type")
                        if kind == "wake":
                            session.wake()
                        elif kind == "stop":
                            session.stop_speaking()
                        else:
                            outbox.put_nowait({"type": "error", "message": f"unknown message type {kind!r}"})
                    except Exception as e:  # noqa: BLE001 - 把错误回给页面显示
                        outbox.put_nowait({"type": "error", "message": str(e)})
                elif msg.type == WSMsgType.ERROR:
                    log.warning("websocket error: %s", ws.exception())
        finally:
            await loop.run_in_executor(None, session.close)
            outbox.put_nowait(None)
            try:
                await asyncio.wait_for(pump_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, ConnectionError):
                pump_task.cancel()
        log.info("dialog client disconnected: %s", request.remote)
        return ws

    app.router.add_get("/dialog", page)
    app.router.add_get("/ws/dialog", ws_handler)
