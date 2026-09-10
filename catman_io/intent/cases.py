"""回归用例：一行一个 JSON，{"text","intent","slots","source","turn_id","note"}；intent 为 none 表示不该命中。

``catman-io intent test`` 与 HTTP API 的 PUT 规则都用它做闸门：改规则不能把旧用例改坏。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import Intent
    from .rules import RuleSet


@dataclass
class Case:
    text: str
    intent: str  # 意图名或 "none"
    slots: dict[str, Any] = field(default_factory=dict)
    source: str = "manual"  # manual | llm | user | builtin
    turn_id: str | None = None
    note: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def read_cases(path: Path) -> list[Case]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                d = json.loads(line)
                out.append(Case(**{k: d.get(k, v) for k, v in _DEFAULTS.items()}))
            except (ValueError, TypeError) as e:
                raise ValueError(f"{path}:{i}: bad case line: {e}") from e
    return out


_DEFAULTS = {"text": "", "intent": "none", "slots": {}, "source": "manual", "turn_id": None, "note": ""}


def append_cases(path: Path, cases: Iterable[Case]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for c in cases:
            f.write(c.to_json() + "\n")
            n += 1
    return n


@dataclass
class CaseResult:
    case: Case
    got: Intent | None
    ok: bool
    reason: str = ""


@dataclass
class Report:
    results: list[CaseResult]

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.ok)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.ok)

    @property
    def ok(self) -> bool:
        return self.failed == 0

    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if not r.ok]

    def format(self) -> str:
        lines = [f"{self.passed} passed, {self.failed} failed"]
        for r in self.failures():
            got = f"{r.got.name} {r.got.slots}" if r.got else "none"
            expected = f"{r.case.intent} {r.case.slots}" if r.case.slots else r.case.intent
            lines.append(f"  FAIL {r.case.text!r}: expected {expected} got {got} ({r.reason})")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "failures": [
                {
                    "text": r.case.text,
                    "expected": r.case.intent,
                    "expected_slots": r.case.slots,
                    "got": r.got.name if r.got else None,
                    "got_slots": r.got.slots if r.got else None,
                    "reason": r.reason,
                }
                for r in self.failures()
            ],
        }


def evaluate(ruleset: RuleSet, cases: Iterable[Case]) -> Report:
    results = []
    for c in cases:
        got = ruleset.match(c.text)
        if c.intent == "none":
            ok = got is None
            reason = "" if ok else "matched but should not"
        elif got is None:
            ok, reason = False, "no match"
        elif got.name != c.intent:
            ok, reason = False, "wrong intent"
        else:
            bad = [k for k, v in c.slots.items() if got.slots.get(k) != v]
            ok = not bad
            reason = "" if ok else f"slots differ: {bad}"
        results.append(CaseResult(c, got, ok, reason))
    return Report(results)
