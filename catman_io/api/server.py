"""HTTP API（aiohttp，可选依赖 ``[demo]``），随 ``catman-io run`` 在线程里跑。

鉴权：除 ``GET /api/health`` 外都要 ``X-Catman-IO-Token``（或 ``Authorization: Bearer``）。
令牌来自 ``api.token_env`` → ``api.token_file`` → 首次运行自动生成到 ``<data_dir>/api_token``。
给 catman 访问时 ``api.host`` 要设成 ``0.0.0.0``（它在 Docker 里，走 host.docker.internal）。

路由：
  GET  /api/health                    状态（不鉴权）
  GET  /api/journal?since=&flags=&bad=1&limit=   回合记录
  GET  /api/journal/review?since=&all=1&format=md  bad case 复盘包（JSON 或 Markdown）
  POST /api/journal/review/ack {until}          记下"看到这里了"
  GET  /api/intent/rules              规则文件列表；GET /api/intent/rules/{name} 取 YAML
  PUT  /api/intent/rules/{name}       写现场规则：先 lint + 跑全部用例，过了才写盘（留 .bak）并热加载；
                                      不过返回 409 带报告（?force=1 只跳过用例失败，lint 错误不能跳）
  POST /api/intent/test {rules?: {name: yaml}}  干跑：用候选 YAML 覆盖后 lint + 用例
  POST /api/intent/parse {text}       看一句话命中哪条规则
  GET  /api/intent/cases?limit=；POST /api/intent/cases {cases: [...]}  回归用例
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from aiohttp import web

from catman_io.config import Config
from catman_io.intent.cases import Case, append_cases, evaluate, read_cases
from catman_io.intent.normalize import normalize
from catman_io.intent.rules import RuleError, RuleStore
from catman_io.journal import BAD_FLAGS, Journal
from catman_io.journal.review import build_review, render_markdown, write_ack

log = logging.getLogger(__name__)

TOKEN_HEADER = "X-Catman-IO-Token"
_NAME = re.compile(r"^[A-Za-z0-9_.-]+\.yaml$")
CTX = web.AppKey("catman_io", dict)


def resolve_api_token(cfg: Config) -> str:
    env = os.environ.get(cfg.api.token_env)
    if env:
        return env
    path = cfg.api_token_path
    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token
    token = secrets.token_urlsafe(24)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    log.info("generated API token at %s", path)
    return token


def make_app(
    *,
    cfg: Config,
    store: RuleStore,
    journal: Journal | None,
    token: str,
    status: Callable[[], dict[str, Any]] | None = None,
):
    app = web.Application(middlewares=[_auth_middleware(token)])
    app[CTX] = {"cfg": cfg, "store": store, "journal": journal, "status": status, "started": time.time()}
    app.router.add_get("/api/health", h_health)
    app.router.add_get("/api/journal", h_journal)
    app.router.add_get("/api/journal/review", h_review)
    app.router.add_post("/api/journal/review/ack", h_review_ack)
    app.router.add_get("/api/intent/rules", h_rules_list)
    app.router.add_get("/api/intent/rules/{name}", h_rules_get)
    app.router.add_put("/api/intent/rules/{name}", h_rules_put)
    app.router.add_post("/api/intent/test", h_intent_test)
    app.router.add_post("/api/intent/parse", h_intent_parse)
    app.router.add_get("/api/intent/cases", h_cases_get)
    app.router.add_post("/api/intent/cases", h_cases_post)
    return app


def _auth_middleware(token: str):
    @web.middleware
    async def middleware(request, handler):
        if request.path == "/api/health":
            return await handler(request)
        given = request.headers.get(TOKEN_HEADER)
        if not given:
            auth = request.headers.get("Authorization", "")
            if auth.lower().startswith("bearer "):
                given = auth[7:].strip()
        if not given or not hmac.compare_digest(given, token):
            return web.json_response({"error": "missing or bad token"}, status=401)
        return await handler(request)

    return middleware


def _json(data: Any, status: int = 200):
    return web.json_response(data, status=status, dumps=lambda d: json.dumps(d, ensure_ascii=False))


async def _body(request) -> dict[str, Any]:
    try:
        data = await request.json()
    except Exception:  # noqa: BLE001
        return {}
    return data if isinstance(data, dict) else {}


# ---- handlers ----


async def h_health(request):
    store: RuleStore = request.app[CTX]["store"]
    data = {
        "ok": True,
        "uptime_s": round(time.time() - request.app[CTX]["started"], 1),
        "intents": len(store.current.intents),
        "patterns": len(store.current.rules),
        "rules_files": [p.name for p in store.paths],
        "journal": request.app[CTX]["journal"] is not None,
    }
    if request.app[CTX]["status"] is not None:
        try:
            data.update(request.app[CTX]["status"]())
        except Exception as e:  # noqa: BLE001
            data["status_error"] = str(e)
    return _json(data)


def _since(request) -> float | None:
    raw = request.query.get("since")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


async def h_journal(request):
    journal: Journal | None = request.app[CTX]["journal"]
    if journal is None:
        return _json({"error": "journal disabled"}, 404)
    flags = None
    if request.query.get("flags"):
        flags = set(request.query["flags"].split(","))
    elif request.query.get("bad"):
        flags = set(BAD_FLAGS)
    limit = int(request.query.get("limit", 100))
    recs = await asyncio.to_thread(journal.records, since=_since(request), flags=flags, limit=limit)
    return _json({"turns": recs, "count": len(recs)})


async def h_review(request):
    journal: Journal | None = request.app[CTX]["journal"]
    if journal is None:
        return _json({"error": "journal disabled"}, 404)
    store: RuleStore = request.app[CTX]["store"]
    intents = {name: spec.description for name, spec in store.current.intents.items()}
    review = await asyncio.to_thread(
        build_review,
        journal,
        since=_since(request),
        include_acked=bool(request.query.get("all")),
        intents=intents,
    )
    if request.query.get("format") == "md":
        return web.Response(text=render_markdown(review), content_type="text/markdown", charset="utf-8")
    return _json(review)


async def h_review_ack(request):
    journal: Journal | None = request.app[CTX]["journal"]
    if journal is None:
        return _json({"error": "journal disabled"}, 404)
    body = await _body(request)
    try:
        until = float(body.get("until"))
    except (TypeError, ValueError):
        return _json({"error": "until (timestamp) required"}, 400)
    write_ack(journal, until)
    return _json({"ok": True, "until": until})


async def h_rules_list(request):
    store: RuleStore = request.app[CTX]["store"]
    files = []
    for p in store.paths:
        n = sum(1 for spec in store.current.intents.values() if spec.source.startswith(str(p)))
        files.append({"name": p.name, "path": str(p), "builtin": p == store.builtin, "intents": n})
    intents = [
        {
            "name": s.name,
            "description": s.description,
            "slots": s.slots,
            "patterns": len(s.patterns),
            "source": s.source,
        }
        for s in store.current.intents.values()
    ]
    return _json({"files": files, "intents": intents, "site_dir": str(store.site_dir)})


def _rule_path(store: RuleStore, name: str) -> Path | None:
    if not _NAME.match(name):
        return None
    if name == store.builtin.name:
        return store.builtin
    return store.site_dir / name


async def h_rules_get(request):
    store: RuleStore = request.app[CTX]["store"]
    path = _rule_path(store, request.match_info["name"])
    if path is None or not path.exists():
        return _json({"error": "no such rules file"}, 404)
    return web.Response(text=path.read_text(encoding="utf-8"), content_type="text/yaml", charset="utf-8")


def _check(store: RuleStore, cfg: Config, override: dict[str, str]) -> tuple[Any, dict[str, Any]]:
    """建候选规则集并检查；返回 (RuleSet 或 None, 报告)。"""
    try:
        rs = store.load_candidate(override)
    except (RuleError, Exception) as e:  # noqa: BLE001
        return None, {"ok": False, "errors": [f"parse: {e}"], "warnings": [], "cases": None}
    errors, warnings = rs.lint()
    report = evaluate(rs, read_cases(cfg.cases_path)).to_dict()
    ok = not errors and report["failed"] == 0
    return rs, {"ok": ok, "errors": errors, "warnings": warnings, "cases": report, "intents": len(rs.intents)}


async def h_rules_put(request):
    store: RuleStore = request.app[CTX]["store"]
    cfg: Config = request.app[CTX]["cfg"]
    name = request.match_info["name"]
    path = _rule_path(store, name)
    if path is None:
        return _json({"error": "bad file name (use something like site.yaml)"}, 400)
    if path == store.builtin:
        return _json({"error": "builtin.yaml is read-only; put your rules in a site file"}, 403)
    text = await request.text()
    if not text.strip():
        return _json({"error": "empty body"}, 400)
    rs, report = await asyncio.to_thread(_check, store, cfg, {name: text})
    force = bool(request.query.get("force"))
    if rs is None or report["errors"] or (report["cases"]["failed"] and not force):
        return _json({"accepted": False, **report}, 409)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.with_suffix(".yaml.bak").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    store.replace(rs)
    log.info("rules file %s updated via API (%d intents)", name, len(rs.intents))
    return _json({"accepted": True, **report})


async def h_intent_test(request):
    store: RuleStore = request.app[CTX]["store"]
    cfg: Config = request.app[CTX]["cfg"]
    body = await _body(request)
    override = body.get("rules") or {}
    if not isinstance(override, dict) or not all(isinstance(v, str) for v in override.values()):
        return _json({"error": "rules must map file name to YAML text"}, 400)
    for name in override:
        if not _NAME.match(name) or name == store.builtin.name:
            return _json({"error": f"bad rules file name {name!r}"}, 400)
    _, report = await asyncio.to_thread(_check, store, cfg, override)
    return _json(report)


async def h_intent_parse(request):
    store: RuleStore = request.app[CTX]["store"]
    body = await _body(request)
    text = str(body.get("text") or "")
    if not text:
        return _json({"error": "text required"}, 400)
    hits = store.current.match_all(text)
    return _json(
        {
            "normalized": normalize(text),
            "intent": hits[0].to_dict() if hits else None,
            "matches": [h.to_dict() for h in hits],
        }
    )


async def h_cases_get(request):
    cfg: Config = request.app[CTX]["cfg"]
    cases = await asyncio.to_thread(read_cases, cfg.cases_path)
    limit = int(request.query.get("limit", 500))
    return _json(
        {"cases": [c.__dict__ for c in cases[-limit:]], "count": len(cases), "path": str(cfg.cases_path)}
    )


async def h_cases_post(request):
    cfg: Config = request.app[CTX]["cfg"]
    body = await _body(request)
    items = body.get("cases")
    if not isinstance(items, list) or not items:
        return _json({"error": "cases (list) required"}, 400)
    cases = []
    for it in items:
        if not isinstance(it, dict) or not it.get("text") or not it.get("intent"):
            return _json({"error": "each case needs text and intent"}, 400)
        cases.append(
            Case(
                text=str(it["text"]),
                intent=str(it["intent"]),
                slots=dict(it.get("slots") or {}),
                source=str(it.get("source") or "catman"),
                turn_id=it.get("turn_id"),
                note=str(it.get("note") or ""),
            )
        )
    n = await asyncio.to_thread(append_cases, cfg.cases_path, cases)
    return _json({"added": n})


# ---- 在线程里跑 ----


class ApiServer:
    def __init__(self, app, host: str, port: int):
        self.app, self.host, self.port = app, host, port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._runner = None
        self._ready = threading.Event()
        self.error: Exception | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="api", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10)
        if self.error is not None:
            raise self.error

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)

        async def serve() -> None:
            self._runner = web.AppRunner(self.app)
            await self._runner.setup()
            site = web.TCPSite(self._runner, self.host, self.port)
            await site.start()

        try:
            loop.run_until_complete(serve())
        except Exception as e:  # noqa: BLE001
            self.error = e
            self._ready.set()
            return
        log.info("api: http://%s:%d/api/ (token header %s)", self.host, self.port, TOKEN_HEADER)
        self._ready.set()
        loop.run_forever()
        loop.run_until_complete(self._runner.cleanup())
        loop.close()

    def stop(self) -> None:
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
