"""把 bad case 整理成给人 / 给 catman 看的复盘包：按标记分组，附候选用例与当前意图清单。

带一个游标（review_ack.json）：catman 处理完一批就 ack，下次不再重复投喂。
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

from .flags import BAD_FLAGS, FLAG_HELP, is_bad
from .journal import Journal

ACK_FILE = "review_ack.json"


def read_ack(journal: Journal) -> float | None:
    p = journal.root / ACK_FILE
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text(encoding="utf-8")).get("until", 0)) or None
    except (ValueError, AttributeError):
        return None


def write_ack(journal: Journal, until: float) -> None:
    (journal.root / ACK_FILE).write_text(json.dumps({"until": until, "at": time.time()}), encoding="utf-8")


def build_review(
    journal: Journal,
    *,
    since: float | None = None,
    include_acked: bool = False,
    intents: dict[str, str] | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    acked = None if include_acked else read_ack(journal)
    start = max(x for x in (since, acked) if x is not None) if (since or acked) else None
    recs = [
        r for r in journal.records(since=start) if is_bad(r) and (acked is None or (r.get("at") or 0) > acked)
    ]
    recs = recs[-limit:]
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in recs:
        for f in r.get("flags") or []:
            if f in BAD_FLAGS:
                groups.setdefault(f, []).append(_brief(r))
    candidates = [r["candidate_case"] for r in recs if r.get("candidate_case")]
    until = max((r.get("at") or 0) for r in recs) if recs else (start or 0)
    return {
        "generated_at": time.time(),
        "since": start,
        "until": until,
        "count": len(recs),
        "groups": {f: {"help": FLAG_HELP.get(f, ""), "turns": v} for f, v in groups.items()},
        "candidate_cases": candidates,
        "intents": intents or {},
    }


def _brief(r: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "turn_id",
        "at",
        "status",
        "text",
        "asr_raw",
        "intent",
        "tier",
        "rule_id",
        "slots",
        "reply_text",
        "reply_source",
        "flags",
        "flag_details",
        "candidate_case",
        "audio_utterance",
        "audio_wake",
        "lat_first_audio_ms",
        "error",
    )
    out = {k: r.get(k) for k in keys if r.get(k) not in (None, "", [], {})}
    if r.get("at"):
        out["time"] = dt.datetime.fromtimestamp(r["at"]).strftime("%Y-%m-%d %H:%M:%S")
    return out


def render_markdown(review: dict[str, Any]) -> str:
    lines = ["# catman-io 意图 bad case 复盘", ""]
    lines.append(f"共 {review['count']} 个回合需要看。处理完请 ack `until={review['until']}`。")
    lines.append("")
    if review.get("intents"):
        lines.append("## 当前意图")
        for name, desc in review["intents"].items():
            lines.append(f"- `{name}`：{desc}")
        lines.append("")
    for flag, group in review["groups"].items():
        lines.append(f"## {flag}（{len(group['turns'])}）")
        if group.get("help"):
            lines.append(f"> {group['help']}")
        for t in group["turns"]:
            head = f"- [{t.get('time', '')}] `{t.get('turn_id')}` 「{t.get('text', '')}」"
            if t.get("intent"):
                slots = json.dumps(t.get("slots") or {}, ensure_ascii=False)
                head += f" → {t['intent']}({t.get('tier')}) {slots}"
            if t.get("reply_text"):
                head += f" ⇒ 「{t['reply_text'][:60]}」"
            lines.append(head)
            if t.get("flag_details", {}).get(flag):
                lines.append(f"  - {json.dumps(t['flag_details'][flag], ensure_ascii=False)}")
        lines.append("")
    if review["candidate_cases"]:
        lines.append("## 候选用例（LLM 判出的意图，确认后追加到 cases.jsonl）")
        lines.append("```jsonl")
        for c in review["candidate_cases"]:
            lines.append(json.dumps(c, ensure_ascii=False))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def write_markdown(review: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(review), encoding="utf-8")
    return path
