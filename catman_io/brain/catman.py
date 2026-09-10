"""catman 客户端：只做两件事——把一句话发进 catman 的管理员聊天（POST /api/chat），以及探活（GET /health）。

回复不在这里收：catman 现在没有独立的 LLM 接口，一般对话由 catman-io 自己的 LLMBrain 处理；
交给 catman 的任务，结果照旧从微信回来。将来 catman 加了 voice 渠道再实现 Brain 接口。
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from catman_io.config import Config

log = logging.getLogger(__name__)


class CatmanError(RuntimeError):
    pass


class CatmanClient:
    def __init__(self, base_url: str, *, token: str, timeout: float = 5.0, http=None):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._http = http or self._urllib

    @staticmethod
    def _urllib(
        method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
    ) -> tuple[int, bytes]:
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def post(self, text: str) -> None:
        """发一句话给 catman（管理员身份）。失败抛 CatmanError。"""
        body = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "X-Catman-Token": self.token}
        try:
            status, raw = self._http("POST", f"{self.base_url}/api/chat", headers, body, self.timeout)
        except Exception as e:  # noqa: BLE001
            raise CatmanError(f"catman unreachable: {e}") from e
        if status >= 400:
            raise CatmanError(f"catman rejected the message: HTTP {status} {raw[:200]!r}")

    def probe(self) -> dict[str, Any] | None:
        try:
            status, raw = self._http("GET", f"{self.base_url}/health", {}, None, self.timeout)
            if status != 200:
                return None
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
        except Exception as e:  # noqa: BLE001
            log.debug("catman probe failed: %s", e)
            return None


def build_catman(cfg: Config) -> CatmanClient | None:
    c = cfg.brain.catman
    if not c.base_url:
        return None
    from catman_io.config import secret

    return CatmanClient(c.base_url, token=secret(c.token_env, what="brain.catman") or "", timeout=c.timeout)
