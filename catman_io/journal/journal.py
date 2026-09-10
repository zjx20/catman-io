"""回合日志：一天一个 JSONL，每回合一行 ``{"kind":"turn",...}``，事后补记用 ``{"kind":"amend",...}``。

音频（唤醒前后那一段、整句）存成 WAV 放在 ``audio/<日期>/`` 下，路径写进记录。读取时把 amend 按序
合并回对应回合（patch 覆盖字段、flags_add 并入标记）。
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import shutil
import threading
import time
import wave
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from catman_io.audio.frames import SAMPLE_RATE, to_int16

log = logging.getLogger(__name__)


class Journal:
    def __init__(self, root: str | Path, *, save_audio: bool = True, keep_days: int = 30, clock=time.time):
        self.root = Path(root)
        self.save_audio_enabled = save_audio
        self.keep_days = keep_days
        self.clock = clock
        self._lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- 写 ----

    def _day(self, at: float | None = None) -> str:
        return dt.datetime.fromtimestamp(at if at is not None else self.clock()).strftime("%Y-%m-%d")

    def _append(self, line: dict[str, Any]) -> None:
        path = self.root / f"{self._day(line.get('at'))}.jsonl"
        data = json.dumps(line, ensure_ascii=False, default=_json_default)
        with self._lock, open(path, "a", encoding="utf-8") as f:
            f.write(data + "\n")

    def write(self, record: dict[str, Any]) -> None:
        record.setdefault("at", self.clock())
        self._append({"kind": "turn", **record})

    def amend(
        self,
        turn_id: str,
        *,
        patch: dict[str, Any] | None = None,
        flags_add: Iterable[str] | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        line: dict[str, Any] = {"kind": "amend", "turn_id": turn_id, "at": self.clock()}
        if patch:
            line["patch"] = patch
        if flags_add:
            line["flags_add"] = list(flags_add)
        if detail:
            line["detail"] = detail
        self._append(line)

    def save_audio(self, turn_id: str, kind: str, audio: np.ndarray) -> str | None:
        if not self.save_audio_enabled or audio is None or len(audio) == 0:
            return None
        d = self.root / "audio" / self._day()
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{turn_id}-{kind}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(to_int16(audio).tobytes())
        return str(path.relative_to(self.root))

    # ---- 读 ----

    def files(self) -> list[Path]:
        return sorted(self.root.glob("????-??-??.jsonl"))

    def records(self, *, since: float | None = None, flags: set[str] | None = None, limit: int | None = None):
        """按时间顺序返回合并后的回合记录。"""
        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        pending: dict[str, list[dict[str, Any]]] = {}
        for path in self.files():
            if since is not None and _day_end(path.stem) < since:
                continue
            with open(path, encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        line = json.loads(raw)
                    except ValueError:
                        log.warning("bad journal line in %s", path)
                        continue
                    if line.get("kind") == "turn":
                        tid = line.get("turn_id")
                        if not tid:
                            continue
                        line.pop("kind", None)
                        line.setdefault("flags", [])
                        merged[tid] = line
                        order.append(tid)
                        for am in pending.pop(tid, []):
                            _apply(line, am)
                    elif line.get("kind") == "amend":
                        tid = line.get("turn_id")
                        if tid in merged:
                            _apply(merged[tid], line)
                        elif tid:
                            pending.setdefault(tid, []).append(line)
        out = [merged[t] for t in order]
        if since is not None:
            out = [r for r in out if r.get("at", 0) >= since]
        if flags:
            out = [r for r in out if flags & set(r.get("flags") or [])]
        if limit is not None and len(out) > limit:
            out = out[-limit:]
        return out

    def get(self, turn_id: str) -> dict[str, Any] | None:
        for r in self.records():
            if r.get("turn_id") == turn_id:
                return r
        return None

    # ---- 清理 ----

    def cleanup(self) -> int:
        if self.keep_days <= 0:
            return 0
        cutoff = (dt.datetime.fromtimestamp(self.clock()) - dt.timedelta(days=self.keep_days)).strftime(
            "%Y-%m-%d"
        )
        n = 0
        for path in self.files():
            if path.stem < cutoff:
                path.unlink()
                n += 1
        audio = self.root / "audio"
        if audio.exists():
            for d in audio.iterdir():
                if d.is_dir() and d.name < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
                    n += 1
        return n


def _apply(record: dict[str, Any], amend: dict[str, Any]) -> None:
    for k, v in (amend.get("patch") or {}).items():
        record[k] = v
    flags = list(record.get("flags") or [])
    for f in amend.get("flags_add") or []:
        if f not in flags:
            flags.append(f)
    record["flags"] = flags
    if amend.get("detail"):
        record.setdefault("flag_details", {}).update(amend["detail"])


def _day_end(day: str) -> float:
    try:
        d = dt.datetime.strptime(day, "%Y-%m-%d") + dt.timedelta(days=1)
        return d.timestamp()
    except ValueError:
        return 0.0


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
