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
    for name in ("index.html", "app.js", "pcm-worklet.js"):
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
