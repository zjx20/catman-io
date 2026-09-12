"""网页 demo 服务端测试：用 aiohttp 的测试客户端走一遍 WebSocket 协议（不需要浏览器）。"""

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

aiohttp = pytest.importorskip("aiohttp")

from catman_io.audio.frames import read_wav  # noqa: E402
from catman_io.wakeword import bundled_models, ensure_base_models  # noqa: E402
from catman_io.webdemo.server import STATIC_DIR, Session, make_app  # noqa: E402

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def ready():
    try:
        ensure_base_models(quiet=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"openWakeWord base models unavailable: {e}")
    if not bundled_models():
        pytest.skip("no bundled wake-word model")


def test_static_files_present():
    for name in ("index.html", "app.js", "pcm-worklet.js", "demo.css", "dialog.html", "dialog.js"):
        assert (STATIC_DIR / name).exists(), name


def test_session_feeds_partial_chunks_and_saves(ready, tmp_path):
    from catman_io.wakeword import WakeWordDetector

    s = Session(WakeWordDetector(threshold=0.5), tmp_path)
    audio = np.concatenate(
        [
            np.zeros(16000, np.int16),
            read_wav(DATA / "positive_siu_maau_jan_hiugaai.wav"),
            np.zeros(16000, np.int16),
        ]
    )
    events = []
    for i in range(0, len(audio), 700):  # 故意用不是 1280 整数倍的块
        events += s.feed(audio[i : i + 700])
    assert len(events) == len(audio) // 1280
    assert any(e["fired"] for e in events), "should detect the wake word"
    assert all(0.0 <= e["score"] <= 1.0 for e in events)
    path = s.save("positive", 2.0)
    assert path.exists() and path.stat().st_size > 44
    assert len(read_wav(path)) == 25 * 1280
    assert s.counts() == {"positive": 1, "negative": 0, "hit": 0}
    with pytest.raises(ValueError):
        s.save("bogus", 1.0)


def test_websocket_roundtrip(ready, tmp_path):
    from aiohttp.test_utils import TestClient, TestServer

    async def run():
        app = make_app(threshold=0.5, record_dir=tmp_path)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/")
            assert resp.status == 200 and "小貓人" in await resp.text()
            resp = await client.get("/static/app.js")
            assert resp.status == 200

            ws = await client.ws_connect("/ws")
            hello = await ws.receive_json()
            assert hello["type"] == "hello" and hello["models"] == ["siu_maau_jan"]

            await ws.send_json({"type": "config", "threshold": 0.4, "patience": 2, "cooldown": 1.0})
            cfg = await ws.receive_json()
            assert cfg == {"type": "config", "threshold": 0.4, "patience": 2, "cooldown": 1.0}

            audio = np.concatenate(
                [
                    np.zeros(16000, np.int16),
                    read_wav(DATA / "positive_siu_maau_jan_hiugaai.wav"),
                    np.zeros(16000, np.int16),
                ]
            )
            fired = []
            for i in range(0, len(audio), 1280 * 4):
                await ws.send_bytes(audio[i : i + 1280 * 4].tobytes())
                for _ in range(len(audio[i : i + 1280 * 4]) // 1280):
                    ev = await ws.receive_json()
                    assert ev["type"] == "frame"
                    fired += ev["fired"]
            assert fired and fired[0]["model"] == "siu_maau_jan"

            await ws.send_json({"type": "save", "label": "hit", "seconds": 1.0})
            saved = await ws.receive_json()
            assert saved["type"] == "saved" and Path(saved["path"]).exists()
            assert saved["counts"]["hit"] == 1
            assert saved["url"] == f"/rec/hit/{saved['name']}" and saved["seconds"] > 0

            # 列表里能看到它，带类型标签
            listed = (await (await client.get("/recordings")).json())["recordings"]
            assert any(r["label"] == "hit" and r["name"] == saved["name"] for r in listed)

            # 能取到 wav 回放
            audio = await client.get(saved["url"])
            assert audio.status == 200 and audio.headers["Content-Type"] == "audio/wav"
            assert len(await audio.read()) > 44

            # 非法 label / 文件名（含目录穿越）一律 404
            assert (await client.get("/rec/bogus/x.wav")).status == 404
            assert (await client.get("/rec/hit/notaname")).status == 404
            assert (await client.get("/recordings/../server.py")).status == 404

            # 删除后文件没了、计数归零
            deleted = await client.delete(saved["url"])
            assert deleted.status == 200 and (await deleted.json())["counts"]["hit"] == 0
            assert (await client.get(saved["url"])).status == 404
            assert not Path(saved["path"]).exists()

            await ws.send_json({"type": "nope"})
            err = await ws.receive_json()
            assert err["type"] == "error"
            await ws.close()

    asyncio.run(run())


def test_frontend_protocol_matches_server():
    """页面发的消息类型服务端都认识，服务端发的页面都处理。"""
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    server = (Path(__file__).parent.parent / "catman_io" / "webdemo" / "server.py").read_text(
        encoding="utf-8"
    )
    for kind in ("config", "save", "reset"):
        assert f"type: '{kind}'" in js and f'kind == "{kind}"' in server, kind
    for kind in ("hello", "frame", "saved", "config", "error"):
        assert f"m.type === '{kind}'" in js, kind
    assert json.loads('{"ok": true}')["ok"]

    # 对话页与 dialog.py 同样对得上
    js = (STATIC_DIR / "dialog.js").read_text(encoding="utf-8")
    server = (Path(__file__).parent.parent / "catman_io" / "webdemo" / "dialog.py").read_text("utf-8")
    for kind in ("wake", "stop"):
        assert f"type: '{kind}'" in js and f'kind == "{kind}"' in server, kind
    for kind in ("hello", "meter", "state", "turn", "audio_stop", "error"):
        assert f"m.type === '{kind}'" in js and f'"type": "{kind}"' in server, kind


def test_dialog_websocket_flow(ready, tmp_path, monkeypatch):
    """对话页：按钮唤醒 → 一句话 → 假识别 → 规则应答（不合成，回复只有文字）→ 回合结果，中间有提示音音频。"""
    from aiohttp import WSMsgType
    from aiohttp.test_utils import TestClient, TestServer

    import catman_io.asr as asr_mod
    from catman_io.asr import Transcript
    from catman_io.config import Config

    class FakeASR:
        def transcribe(self, audio):
            return Transcript("而家幾點", duration=len(audio) / 16000, elapsed=0.01)

    monkeypatch.setattr(asr_mod, "create_recognizer", lambda cfg: FakeASR())
    cfg = Config.load(None)
    cfg.data_dir = str(tmp_path)
    cfg.tts.backend = "none"
    cfg.dialog.followup_seconds = 0.0  # 回完直接回到待唤醒，序列好断言
    speech = read_wav(DATA / "negative_weather_wanlung.wav")
    silence = np.zeros(1280, np.int16)

    async def run():
        app = make_app(threshold=0.5, record_dir=tmp_path, cfg=cfg)
        async with TestClient(TestServer(app)) as client:
            assert "对话 demo" in await (await client.get("/dialog")).text()
            assert (await client.get("/static/dialog.js")).status == 200
            ws = await client.ws_connect("/ws/dialog")
            hello = await ws.receive_json()
            assert hello["type"] == "hello" and hello["models"] == ["siu_maau_jan"]
            assert hello["asr"] and not hello["tts"] and hello["intents"] > 0 and hello["earcons"]

            got = {"audio": 0, "states": [], "turn": None, "meters": 0}

            async def drain(timeout=0.05):
                while True:
                    try:
                        m = await ws.receive(timeout=timeout)
                    except asyncio.TimeoutError:
                        return
                    if m.type == WSMsgType.BINARY:
                        got["audio"] += len(m.data)
                        continue
                    d = json.loads(m.data)
                    if d["type"] == "meter":
                        got["meters"] += 1
                    elif d["type"] == "state":
                        got["states"].append(d["state"])
                    elif d["type"] == "turn":
                        got["turn"] = d

            await ws.send_json({"type": "wake"})
            audio = np.concatenate([np.zeros(8000, np.int16), speech, np.zeros(16000, np.int16)])
            for i in range(0, len(audio), 1280):
                await ws.send_bytes(audio[i : i + 1280].tobytes())
            # 管线的钟是墙钟，事件要等下一帧才被处理：持续喂静音直到回合结束
            for _ in range(200):
                await ws.send_bytes(silence.tobytes())
                await drain()
                if got["turn"] is not None:
                    break
            turn = got["turn"]
            assert turn is not None, got
            assert turn["status"] == "ok" and turn["text"] == "而家幾點" and turn["intent"] == "time.now"
            assert turn["reply"] and turn["wake_model"] == "manual"
            assert turn["latency"]["speech_end"] is not None and turn["latency"]["asr"] is not None
            assert got["states"][:2] == ["listening", "thinking"] and got["states"][-1] == "idle"
            assert "speaking" in got["states"]
            assert got["audio"] > 0 and got["meters"] > 0  # 唤醒提示音回来了、仪表在走

            await ws.send_json({"type": "nope"})
            for _ in range(20):
                await drain()
            await ws.close()
        # 回合写进了日志目录
        assert list((tmp_path / "journal").rglob("*.jsonl"))

    asyncio.run(run())
