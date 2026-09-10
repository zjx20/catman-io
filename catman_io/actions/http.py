"""通用 http 动作与模板：``{slot}`` / ``{slot|默认}`` / ``{r.a.b}`` 取值，``${ENV}`` 取环境变量。

YAML 里这样写就是一个新技能：
    action: {type: http, method: POST, url: "http://host/api", headers: {Authorization: "Bearer ${TOKEN}"},
             json: {entity_id: "light.{room|living_room}"}, timeout: 5}
    say: "開咗{room|}燈喇"
响应是 JSON 时可以在 say 里用 {r.current.temperature_2m} 取字段。
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from catman_io.intent import Intent

from . import ActionContext, ActionResult

log = logging.getLogger(__name__)

_PLACEHOLDER = re.compile(r"(?<!\$)\{([a-zA-Z_][\w.]*)(?:\|([^}]*))?\}")
_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _lookup(ctx: dict[str, Any], path: str) -> Any:
    cur: Any = ctx
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit():
            cur = cur[int(part)] if int(part) < len(cur) else None
        else:
            return None
        if cur is None:
            return None
    return cur


def render(template: str, ctx: dict[str, Any]) -> str:
    def sub(m: re.Match[str]) -> str:
        value = _lookup(ctx, m.group(1))
        if value is None or value == "":
            return m.group(2) if m.group(2) is not None else ""
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return str(value)

    out = _PLACEHOLDER.sub(sub, template)
    return _ENV.sub(lambda m: os.environ.get(m.group(1), ""), out)


def render_value(value: Any, ctx: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return render(value, ctx)
    if isinstance(value, dict):
        return {k: render_value(v, ctx) for k, v in value.items()}
    if isinstance(value, list):
        return [render_value(v, ctx) for v in value]
    return value


def urllib_http(
    method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def http_action(action: dict[str, Any], intent: Intent, ctx: ActionContext) -> ActionResult:
    tctx: dict[str, Any] = {**intent.slots, "text": intent.raw or ""}
    method = str(action.get("method", "GET")).upper()
    url = render(str(action.get("url", "")), tctx)
    if not url:
        return ActionResult(ok=False, error="http action without url")
    query = render_value(action.get("query") or {}, tctx)
    if query:
        sep = "&" if "?" in url else "?"
        url = url + sep + urllib.parse.urlencode(query)
    headers = {str(k): str(v) for k, v in render_value(action.get("headers") or {}, tctx).items()}
    body: bytes | None = None
    if "json" in action:
        body = json.dumps(render_value(action["json"], tctx), ensure_ascii=False).encode("utf-8")
        headers.setdefault("Content-Type", "application/json")
    elif "body" in action:
        body = str(render(str(action["body"]), tctx)).encode("utf-8")
    timeout = float(action.get("timeout") or ctx.cfg.actions.http_timeout)
    http = ctx.http or urllib_http
    try:
        status, raw = http(method, url, headers, body, timeout)
    except Exception as e:  # noqa: BLE001
        log.warning("http action %s %s failed: %s", method, url, e)
        return ActionResult(say="做唔到，連唔到服務。", ok=False, error=str(e))
    data: Any = None
    if raw:
        try:
            data = json.loads(raw)
        except ValueError:
            data = raw.decode("utf-8", "replace")
    if status >= 400:
        return ActionResult(
            say="做唔到，服務出錯。", ok=False, error=f"HTTP {status}", data={"response": data}
        )
    result_data = data if isinstance(data, dict) else {"response": data}
    return ActionResult(say=str(action.get("say") or "搞掂"), data=result_data)
