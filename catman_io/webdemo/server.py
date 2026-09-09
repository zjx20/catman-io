"""aiohttp 服务：静态页面 + /ws WebSocket。

协议（一个连接 = 一个检测会话）：
- 浏览器 → 服务端：二进制帧是 16 kHz 单声道 int16 PCM，长度任意，服务端攒够 1280 个采样点跑一帧；
  文本帧是 JSON：{"type":"config", threshold/patience/cooldown}、
  {"type":"save","label":..,"seconds":..}、{"type":"reset"}
- 服务端 → 浏览器：{"type":"hello",...}、每帧一条 {"type":"frame","t":秒,"score":..,"fired":[..]}、
  {"type":"saved",...}、{"type":"config",...}、{"type":"error",...}
"""

from __future__ import annotations

import json
import logging
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np

from catman_io.audio.frames import FRAME_SAMPLES, FRAME_SECONDS, SAMPLE_RATE
from catman_io.wakeword import Detection, WakeWordDetector

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
LABELS = ("positive", "negative", "hit")
RING_SECONDS = 12.0


class Session:
    """一个浏览器连接：检测器 + 最近几秒的音频环形缓冲（用于保存样本）。"""

    def __init__(self, detector: WakeWordDetector, record_dir: Path):
        self.detector = detector
        self.record_dir = record_dir
        self.pending = np.zeros(0, dtype=np.int16)
        self.ring: deque[np.ndarray] = deque(maxlen=int(RING_SECONDS / FRAME_SECONDS))

    def feed(self, pcm: np.ndarray) -> list[dict]:
        """喂入任意长度的 int16，返回本次跑出的逐帧消息。"""
        self.pending = np.concatenate([self.pending, pcm]) if len(self.pending) else pcm
        out: list[dict] = []
        while len(self.pending) >= FRAME_SAMPLES:
            frame, self.pending = self.pending[:FRAME_SAMPLES], self.pending[FRAME_SAMPLES:]
            self.ring.append(frame)
            fired = self.detector.process(frame)
            scores = self.detector.last_scores
            out.append(
                {
                    "type": "frame",
                    "t": round(self.detector.stream_time, 3),
                    "score": round(max(scores.values()) if scores else 0.0, 4),
                    "scores": {k: round(v, 4) for k, v in scores.items()},
                    "fired": [_det_json(d) for d in fired],
                    "level_db": round(_level_db(frame), 1),
                }
            )
        return out

    def save(self, label: str, seconds: float) -> Path:
        if label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}")
        n_frames = max(1, int(round(seconds / FRAME_SECONDS)))
        frames = list(self.ring)[-n_frames:]
        if not frames:
            raise ValueError("no audio buffered yet")
        audio = np.concatenate(frames)
        out_dir = self.record_dir / label
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / (time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}.wav")
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(audio.tobytes())
        return path

    def counts(self) -> dict[str, int]:
        return {
            lb: len(list((self.record_dir / lb).glob("*.wav"))) if (self.record_dir / lb).exists() else 0
            for lb in LABELS
        }


def _det_json(d: Detection) -> dict:
    return {"model": d.model, "score": round(d.score, 4), "t": round(d.stream_time, 3)}


def _level_db(frame: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
    return 20 * np.log10(rms / 32768 + 1e-9)


def make_app(
    model_paths=None,
    threshold: float = 0.5,
    patience: int = 1,
    cooldown: float = 2.0,
    vad_threshold: float = 0.0,
    record_dir: Path = Path("data/recordings"),
):
    from aiohttp import WSMsgType, web

    record_dir = Path(record_dir)

    def new_detector() -> WakeWordDetector:
        return WakeWordDetector(
            model_paths,
            threshold=threshold,
            patience=patience,
            cooldown=cooldown,
            vad_threshold=vad_threshold,
        )

    async def index(request):
        return web.FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

    async def favicon(request):
        return web.Response(status=204)

    async def ws_handler(request):
        ws = web.WebSocketResponse(max_msg_size=4 << 20, heartbeat=20)
        await ws.prepare(request)
        session = Session(new_detector(), record_dir)
        det = session.detector
        await ws.send_json(
            {
                "type": "hello",
                "models": det.names,
                "threshold": det.threshold,
                "patience": det.patience,
                "cooldown": det.cooldown,
                "record_dir": str(record_dir.resolve()),
                "counts": session.counts(),
            }
        )
        log.info("client connected: %s", request.remote)
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                pcm = np.frombuffer(msg.data, dtype="<i2")
                for event in session.feed(pcm):
                    await ws.send_json(event)
            elif msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    kind = data.get("type")
                    if kind == "config":
                        det.configure(data.get("threshold"), data.get("patience"), data.get("cooldown"))
                        await ws.send_json(
                            {
                                "type": "config",
                                "threshold": det.threshold,
                                "patience": det.patience,
                                "cooldown": det.cooldown,
                            }
                        )
                    elif kind == "save":
                        path = session.save(
                            str(data.get("label", "positive")), float(data.get("seconds", 3.0))
                        )
                        await ws.send_json(
                            {
                                "type": "saved",
                                "label": data.get("label"),
                                "path": str(path),
                                "counts": session.counts(),
                            }
                        )
                    elif kind == "reset":
                        det.reset()
                        session.pending = np.zeros(0, dtype=np.int16)
                        await ws.send_json({"type": "reset"})
                    else:
                        await ws.send_json({"type": "error", "message": f"unknown message type {kind!r}"})
                except Exception as e:  # noqa: BLE001 - 把错误回给页面显示
                    await ws.send_json({"type": "error", "message": str(e)})
            elif msg.type == WSMsgType.ERROR:
                log.warning("websocket error: %s", ws.exception())
        log.info("client disconnected: %s", request.remote)
        return ws

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/favicon.ico", favicon)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/static/", STATIC_DIR, show_index=False)
    return app


def run(
    host: str = "127.0.0.1",
    port: int = 8765,
    model_paths=None,
    threshold: float = 0.5,
    patience: int = 1,
    cooldown: float = 2.0,
    vad_threshold: float = 0.0,
    record_dir: Path = Path("data/recordings"),
    open_browser: bool = False,
) -> None:
    try:
        from aiohttp import web
    except ImportError as e:
        raise SystemExit("aiohttp not installed: `pip install catman-io[demo]`") from e

    # 先建一次检测器：基础模型没下载 / 模型文件不对，这里就报错，而不是等浏览器连上来
    WakeWordDetector(model_paths, threshold=threshold, patience=patience, cooldown=cooldown)
    app = make_app(model_paths, threshold, patience, cooldown, vad_threshold, record_dir)
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}/"
    print(f"webdemo: open {url}  (recordings → {Path(record_dir).resolve()})")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(
            "note: browsers only allow microphone access on http://localhost or https://; "
            "to test from another machine, use an SSH tunnel (ssh -L 8765:127.0.0.1:8765 <host>)"
        )
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    web.run_app(app, host=host, port=port, print=None)
