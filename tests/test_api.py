"""HTTP API：鉴权、日志与复盘、规则 PUT 的闸门与热加载、干跑、用例。"""

import asyncio
import json

import pytest

pytest.importorskip("aiohttp")

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from catman_io.api import make_app  # noqa: E402
from catman_io.api.server import resolve_api_token  # noqa: E402
from catman_io.config import Config  # noqa: E402
from catman_io.intent import rule_store  # noqa: E402
from catman_io.journal import Journal  # noqa: E402

SITE_OK = """version: 1
intents:
  - name: light.on
    slots: {room: "room?"}
    examples: ["開燈", "幫我開廳燈"]
    patterns: ["^{polite}開(?:埋)?({room})?(?:盞)?燈{tail}$"]
    action: {type: http, url: "http://x"}
"""
SITE_BAD_LINT = """version: 1
intents:
  - name: light.on
    examples: ["開燈"]
    patterns: ["^熄燈$"]
    action: {type: http, url: "http://x"}
"""


def setup(tmp_path):
    cfg = Config.load(None)
    cfg.data_dir = str(tmp_path)
    cfg.rules_dir.mkdir(parents=True)
    store = rule_store(cfg)
    journal = Journal(cfg.journal_dir)
    journal.write(
        {
            "turn_id": "a",
            "at": 100.0,
            "status": "ok",
            "text": "聲音調去三成",
            "intent": "volume.set",
            "tier": "llm",
            "slots": {"percent": 30},
            "flags": ["rules_miss_llm_hit"],
            "candidate_case": {
                "text": "聲音調去三成",
                "intent": "volume.set",
                "slots": {"percent": 30},
                "source": "llm",
            },
        }
    )
    journal.write(
        {
            "turn_id": "b",
            "at": 200.0,
            "status": "ok",
            "text": "而家幾點",
            "intent": "time.now",
            "tier": "rule",
            "flags": [],
        }
    )
    app = make_app(cfg=cfg, store=store, journal=journal, token="tok", status=lambda: {"state": "idle"})
    return cfg, store, journal, app


def run(coro):
    return asyncio.run(coro)


async def client_for(app):
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


H = {"X-Catman-IO-Token": "tok"}


def test_auth_and_health(tmp_path):
    cfg, store, journal, app = setup(tmp_path)

    async def go():
        c = await client_for(app)
        try:
            r = await c.get("/api/health")
            data = await r.json()
            assert r.status == 200 and data["ok"] and data["intents"] >= 12 and data["state"] == "idle"
            assert (await c.get("/api/journal")).status == 401
            assert (await c.get("/api/journal", headers={"X-Catman-IO-Token": "nope"})).status == 401
            assert (await c.get("/api/journal", headers={"Authorization": "Bearer tok"})).status == 200
        finally:
            await c.close()

    run(go())


def test_journal_review_and_ack(tmp_path):
    cfg, store, journal, app = setup(tmp_path)

    async def go():
        c = await client_for(app)
        try:
            data = await (await c.get("/api/journal?bad=1", headers=H)).json()
            assert data["count"] == 1 and data["turns"][0]["turn_id"] == "a"
            data = await (await c.get("/api/journal?since=150", headers=H)).json()
            assert [t["turn_id"] for t in data["turns"]] == ["b"]
            review = await (await c.get("/api/journal/review", headers=H)).json()
            assert (
                review["count"] == 1
                and "rules_miss_llm_hit" in review["groups"]
                and review["intents"]["time.now"]
            )
            md = await (await c.get("/api/journal/review?format=md", headers=H)).text()
            assert md.startswith("# catman-io") and "聲音調去三成" in md
            r = await c.post("/api/journal/review/ack", headers=H, json={"until": review["until"]})
            assert r.status == 200
            assert (await (await c.get("/api/journal/review", headers=H)).json())["count"] == 0
            assert (await c.post("/api/journal/review/ack", headers=H, json={})).status == 400
        finally:
            await c.close()

    run(go())


def test_rules_put_gate_and_hot_reload(tmp_path):
    cfg, store, journal, app = setup(tmp_path)
    cfg.cases_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.cases_path.write_text(json.dumps({"text": "而家幾點", "intent": "time.now"}) + "\n", encoding="utf-8")

    async def go():
        c = await client_for(app)
        try:
            data = await (await c.get("/api/intent/rules", headers=H)).json()
            assert [f["name"] for f in data["files"]] == ["builtin.yaml"] and data["files"][0]["builtin"]
            assert (await c.get("/api/intent/rules/site.yaml", headers=H)).status == 404
            assert (await c.get("/api/intent/rules/builtin.yaml", headers=H)).status == 200
            assert (await c.put("/api/intent/rules/builtin.yaml", headers=H, data=SITE_OK)).status == 403
            assert (await c.put("/api/intent/rules/../x.yaml", headers=H, data=SITE_OK)).status in (400, 404)
            r = await c.put("/api/intent/rules/site.yaml", headers=H, data=SITE_BAD_LINT)
            body = await r.json()
            assert r.status == 409 and not body["accepted"] and any("example" in e for e in body["errors"])
            assert not (cfg.rules_dir / "site.yaml").exists()
            r = await c.put("/api/intent/rules/site.yaml", headers=H, data=SITE_OK)
            body = await r.json()
            assert r.status == 200 and body["accepted"] and body["cases"]["failed"] == 0
            assert (cfg.rules_dir / "site.yaml").read_text(encoding="utf-8") == SITE_OK
            assert store.current.match("幫我開廳燈").name == "light.on"  # 热加载
            data = await (await c.get("/api/intent/rules", headers=H)).json()
            assert [f["name"] for f in data["files"]] == ["builtin.yaml", "site.yaml"]
            # 破坏旧用例：time.now 被覆盖成别的说法 → 409，带 .bak 不动
            broken = (
                SITE_OK
                + "  - name: time.now\n    examples: ['報時']\n"
                + "    patterns: ['^報時$']\n    action: builtin.time\n"
            )
            r = await c.put("/api/intent/rules/site.yaml", headers=H, data=broken)
            body = await r.json()
            assert r.status == 409 and body["cases"]["failed"] == 1 and body["errors"] == []
            assert (cfg.rules_dir / "site.yaml").read_text(encoding="utf-8") == SITE_OK
            r = await c.put("/api/intent/rules/site.yaml?force=1", headers=H, data=broken)
            assert (
                r.status == 200 and (cfg.rules_dir / "site.yaml.bak").read_text(encoding="utf-8") == SITE_OK
            )
            assert store.current.match("而家幾點") is None
        finally:
            await c.close()

    run(go())


def test_intent_test_parse_and_cases(tmp_path):
    cfg, store, journal, app = setup(tmp_path)

    async def go():
        c = await client_for(app)
        try:
            r = await c.post("/api/intent/test", headers=H, json={"rules": {"site.yaml": SITE_OK}})
            body = await r.json()
            assert r.status == 200 and body["ok"] and body["intents"] == 13 and body["errors"] == []
            assert not (cfg.rules_dir / "site.yaml").exists()  # 干跑不落盘
            r = await c.post("/api/intent/test", headers=H, json={"rules": {"builtin.yaml": "x"}})
            assert r.status == 400
            body = await (
                await c.post("/api/intent/parse", headers=H, json={"text": "唔該幫我較個十分鐘嘅鬧鐘"})
            ).json()
            assert body["intent"]["name"] == "timer.set" and body["intent"]["slots"] == {"duration": 600}
            assert body["normalized"] == "唔該幫我較個十分鐘嘅鬧鐘"
            assert (await c.post("/api/intent/parse", headers=H, json={})).status == 400
            r = await c.post(
                "/api/intent/cases",
                headers=H,
                json={
                    "cases": [
                        {"text": "聲音調去三成", "intent": "volume.set", "slots": {"percent": 30}},
                        {"text": "我想飲咖啡", "intent": "none"},
                    ]
                },
            )
            assert (await r.json())["added"] == 2
            body = await (await c.get("/api/intent/cases", headers=H)).json()
            assert body["count"] == 2 and body["cases"][0]["source"] == "catman"
            assert (
                await c.post("/api/intent/cases", headers=H, json={"cases": [{"text": "x"}]})
            ).status == 400
            # 新用例现在会挡住让它失败的规则
            r = await c.post("/api/intent/test", headers=H, json={})
            assert (await r.json())["cases"]["failed"] == 1  # 聲音調去三成 规则还没覆盖
        finally:
            await c.close()

    run(go())


def test_resolve_api_token(tmp_path, monkeypatch):
    cfg = Config.load(None)
    cfg.data_dir = str(tmp_path)
    monkeypatch.delenv("CATMAN_IO_API_TOKEN", raising=False)
    t1 = resolve_api_token(cfg)
    assert len(t1) > 20 and cfg.api_token_path.read_text().strip() == t1
    assert resolve_api_token(cfg) == t1
    monkeypatch.setenv("CATMAN_IO_API_TOKEN", "env-token")
    assert resolve_api_token(cfg) == "env-token"


def test_api_server_thread_starts_and_stops(tmp_path):
    import urllib.request

    from catman_io.api import ApiServer

    cfg, store, journal, app = setup(tmp_path)
    server = ApiServer(app, "127.0.0.1", 0)
    server.start()
    try:
        port = server._runner.addresses[0][1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5) as r:
            assert json.loads(r.read())["ok"]
    finally:
        server.stop()
