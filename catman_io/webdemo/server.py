"""aiohttp 服务：静态页面 + /ws WebSocket。

协议（一个连接 = 一个检测会话）：
- 浏览器 → 服务端：二进制帧是 16 kHz 单声道 int16 PCM，长度任意，服务端攒够 1280 个采样点跑一帧；
  文本帧是 JSON：{"type":"config", threshold/patience/cooldown}、
  {"type":"save","label":..,"seconds":..}、{"type":"reset"}
- 服务端 → 浏览器：{"type":"hello",...}、每帧一条 {"type":"frame","t":秒,"score":..,"fired":[..]}、
  {"type":"saved",...}、{"type":"config",...}、{"type":"error",...}

另有普通 HTTP 路由供页面回放已存样本：GET /recordings 列出所有样本
（label/name/url/seconds/bytes/mtime，最新在前）；GET /rec/{label}/{name} 提供对应的 wav 文件；
DELETE /rec/{label}/{name} 删除它。label 必须在 LABELS 里、name 必须是保存时的文件名格式，防目录穿越。

/dialog 是对话 demo 页：浏览器麦克风进整条管线、回复音频回浏览器播，协议见 ``dialog.py``。
"""

from __future__ import annotations

import json
import logging
import re
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np

from catman_io.audio.frames import FRAME_SAMPLES, FRAME_SECONDS, SAMPLE_RATE
from catman_io.config import Config
from catman_io.wakeword import Detection, WakeWordDetector

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
LABELS = ("positive", "negative", "hit")
RING_SECONDS = 12.0
# save() 写出的文件名格式：20260911-071530-123.wav。回放路由只认这个格式，杜绝 ../ 之类的目录穿越。
REC_NAME_RE = re.compile(r"^\d{8}-\d{6}-\d{3}\.wav$")


def _wav_seconds(size_bytes: int) -> float:
    """16 kHz 单声道 16-bit：秒数 = (字节数 - 44 字节头) / 2 / 16000。"""
    return round(max(0.0, (size_bytes - 44) / 2 / SAMPLE_RATE), 2)


def list_recordings(record_dir: Path) -> list[dict]:
    """列出 record_dir 下所有已存样本，最新在前。"""
    items: list[dict] = []
    for label in LABELS:
        d = record_dir / label
        if not d.exists():
            continue
        for wav in d.glob("*.wav"):
            try:
                stat = wav.stat()
            except OSError:
                continue
            items.append(
                {
                    "label": label,
                    "name": wav.name,
                    "url": f"/rec/{label}/{wav.name}",
                    "seconds": _wav_seconds(stat.st_size),
                    "bytes": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            )
    items.sort(key=lambda it: it["mtime"], reverse=True)
    return items


def _counts(record_dir: Path) -> dict[str, int]:
    return {
        lb: len(list((record_dir / lb).glob("*.wav"))) if (record_dir / lb).exists() else 0 for lb in LABELS
    }


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
        return _counts(self.record_dir)


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
    cfg: Config | None = None,
):
    """唤醒词测试页（/）用上面几个检测参数；对话 demo 页（/dialog）按 ``cfg`` 组装整条管线。"""
    from aiohttp import WSMsgType, web

    from .dialog import add_dialog_routes

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

    async def recordings(request):
        return web.json_response({"recordings": list_recordings(record_dir)})

    async def recording_file(request):
        label, name = request.match_info["label"], request.match_info["name"]
        if label not in LABELS or not REC_NAME_RE.match(name):
            raise web.HTTPNotFound()
        path = record_dir / label / name
        if not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={"Content-Type": "audio/wav", "Cache-Control": "no-store"})

    async def recording_delete(request):
        label, name = request.match_info["label"], request.match_info["name"]
        if label not in LABELS or not REC_NAME_RE.match(name):
            raise web.HTTPNotFound()
        path = record_dir / label / name
        if not path.is_file():
            raise web.HTTPNotFound()
        path.unlink()
        return web.json_response({"deleted": f"/rec/{label}/{name}", "counts": _counts(record_dir)})

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
                        label = path.parent.name
                        await ws.send_json(
                            {
                                "type": "saved",
                                "label": label,
                                "path": str(path),
                                "name": path.name,
                                "url": f"/rec/{label}/{path.name}",
                                "seconds": _wav_seconds(path.stat().st_size),
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
    app.router.add_get("/recordings", recordings)
    app.router.add_get("/rec/{label}/{name}", recording_file)
    app.router.add_delete("/rec/{label}/{name}", recording_delete)
    app.router.add_get("/ws", ws_handler)
    add_dialog_routes(app, cfg or Config(), model_paths)
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
    cfg: Config | None = None,
) -> None:
    try:
        from aiohttp import web
    except ImportError as e:
        raise SystemExit("aiohttp not installed: `pip install catman-io[demo]`") from e

    # 先建一次检测器：基础模型没下载 / 模型文件不对，这里就报错，而不是等浏览器连上来
    WakeWordDetector(model_paths, threshold=threshold, patience=patience, cooldown=cooldown)
    app = make_app(model_paths, threshold, patience, cooldown, vad_threshold, record_dir, cfg=cfg)
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}/"
    print(f"webdemo: open {url}  (wake-word test)  or  {url}dialog  (full dialog demo)")
    print(f"recordings → {Path(record_dir).resolve()}")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(
            "note: browsers only allow microphone access on http://localhost or https://; "
            "to test from another machine, use an SSH tunnel (ssh -L 8765:127.0.0.1:8765 <host>)"
        )
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    web.run_app(app, host=host, port=port, print=None)
