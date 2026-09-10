"""规则层：YAML 里的意图 + 正则模式（含 {槽位} 占位）→ 已编译、不可变的 RuleSet；RuleStore 负责热加载。

模式写在归一化后的文本上（见 ``normalize.py``）：无标点、小写、香港繁体。``{duration}`` 这类占位展开成
命名组，值交给对应槽位解析器得到规范值；``{tail}`` / ``{polite}`` 是常用的语气词 / 客气话宏（非捕获）。
每个意图的 ``examples`` 既是文档也是 lint 用的自检：每一句都必须命中本意图。
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import Intent
from .normalize import normalize, to_hk
from .slots import SlotType, slot_types

log = logging.getLogger(__name__)

MACROS = {
    "tail": "(?:呀|啊|喇|啦|嘅|呢|吖|噃|喎|囉|囖|嘞|架|㗎|㖭|添|先|下|吓|呀嘛|好唔好|得唔得|可以嗎|得嗎)*",
    "polite": "(?:唔該|唔該你|麻煩|麻煩你|幫我|幫幫我|幫手|請|請你|可唔可以|可以|同我|幫下我|你|貓人|喂)*",
}
_PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


@dataclass(frozen=True)
class IntentSpec:
    name: str
    description: str = ""
    slots: dict[str, str] = field(default_factory=dict)  # 槽位名 → 类型名，类型名带 ? 表示可选
    examples: tuple[str, ...] = ()
    negatives: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()
    action: Any = None  # "builtin.time" 这类名字，或 {type: http, ...}
    say: str | None = None
    followup: bool = False
    priority: int = 0
    source: str = ""

    def slot_type(self, slot: str) -> str:
        return self.slots.get(slot, slot).rstrip("?")

    def slot_optional(self, slot: str) -> bool:
        return self.slots.get(slot, "").endswith("?")


@dataclass(frozen=True)
class Rule:
    intent: str
    index: int
    raw: str
    regex: re.Pattern[str]
    slots: dict[str, SlotType]
    priority: int

    @property
    def id(self) -> str:
        return f"{self.intent}#{self.index}"


class RuleError(ValueError):
    pass


class RuleSet:
    def __init__(self, intents: dict[str, IntentSpec], rules: list[Rule], types: dict[str, SlotType]):
        self.intents = intents
        self.rules = sorted(rules, key=lambda r: -r.priority)
        self.types = types

    # ---- 构造 ----

    @classmethod
    def load(cls, paths: Sequence[str | Path], *, rooms: list[str] | None = None) -> RuleSet:
        """按顺序加载，后面的文件里同名意图覆盖前面的（builtin 在前，现场规则在后）。"""
        docs = []
        for p in paths:
            p = Path(p)
            with open(p, encoding="utf-8") as f:
                docs.append((str(p), yaml.safe_load(f) or {}))
        return cls.from_docs(docs, rooms=rooms)

    @classmethod
    def from_docs(
        cls, docs: Sequence[tuple[str, dict[str, Any]]], *, rooms: list[str] | None = None
    ) -> RuleSet:
        types = slot_types(rooms)
        intents: dict[str, IntentSpec] = {}
        for source, doc in docs:
            if not isinstance(doc, dict):
                raise RuleError(f"{source}: expected a mapping at top level")
            unknown = set(doc) - {"version", "intents"}
            if unknown:
                raise RuleError(f"{source}: unknown top-level keys {sorted(unknown)}")
            for i, item in enumerate(doc.get("intents") or []):
                spec = _parse_spec(item, source=f"{source}#{i}")
                intents[spec.name] = spec
        rules = []
        for spec in intents.values():
            for i, raw in enumerate(spec.patterns):
                rules.append(_compile(spec, i, raw, types))
        return cls(intents, rules, types)

    # ---- 匹配 ----

    def match_all(self, text: str, *, normalized: bool = False) -> list[Intent]:
        t = text if normalized else normalize(text)
        hits: list[tuple[int, int, int, Intent]] = []
        for order, rule in enumerate(self.rules):
            m = rule.regex.search(t)
            if not m:
                continue
            slots: dict[str, Any] = {}
            bad = False
            for name, st in rule.slots.items():
                raw = m.group(name)
                if raw is None:
                    continue
                value = st.parse(raw)
                if value is None:
                    bad = True
                    break
                slots[name] = value
            if bad:
                continue
            span = m.end() - m.start()
            intent = Intent(rule.intent, slots, 1.0, "rule", rule.id, m.group(0))
            hits.append((rule.priority, span, -order, intent))
        hits.sort(key=lambda h: (h[0], h[1], h[2]), reverse=True)
        return [h[3] for h in hits]

    def match(self, text: str, *, normalized: bool = False) -> Intent | None:
        hits = self.match_all(text, normalized=normalized)
        return hits[0] if hits else None

    # ---- 检查 ----

    def lint(self) -> tuple[list[str], list[str]]:
        """返回 (errors, warnings)。"""
        errors: list[str] = []
        warnings: list[str] = []
        for spec in self.intents.values():
            if not spec.patterns:
                warnings.append(f"{spec.name}: no patterns")
            for raw in spec.patterns:
                if not raw.startswith("^"):
                    warnings.append(f"{spec.name}: pattern not anchored with ^: {raw!r}")
            for ex in spec.examples:
                got = self.match(ex)
                if got is None:
                    errors.append(f"{spec.name}: example {ex!r} matches nothing")
                elif got.name != spec.name:
                    errors.append(f"{spec.name}: example {ex!r} matches {got.name} ({got.rule_id}) instead")
            for neg in spec.negatives:
                got = self.match(neg)
                if got is not None and got.name == spec.name:
                    errors.append(f"{spec.name}: negative {neg!r} matches ({got.rule_id})")
        for rule in self.rules:
            if rule.regex.search(""):
                errors.append(f"{rule.id}: pattern matches the empty string")
        return errors, warnings

    # ---- 给 LLM 的工具定义 ----

    def tools(self) -> list[dict[str, Any]]:
        out = []
        for spec in self.intents.values():
            props: dict[str, Any] = {}
            required = []
            for slot in spec.slots:
                st = self.types.get(spec.slot_type(slot))
                props[slot] = {
                    "type": st.json_type if st else "string",
                    "description": st.description if st else slot,
                }
                if not spec.slot_optional(slot):
                    required.append(slot)
            desc = spec.description or spec.name
            if spec.examples:
                desc += "。例如：" + "；".join(spec.examples[:3])
            fn: dict[str, Any] = {
                "name": tool_name(spec.name),
                "description": desc,
                "parameters": {"type": "object", "properties": props, "required": required},
            }
            out.append({"type": "function", "function": fn})
        return out

    def intent_for_tool(self, tool: str) -> str | None:
        for name in self.intents:
            if tool_name(name) == tool:
                return name
        return None


def tool_name(intent: str) -> str:
    return intent.replace(".", "_")


def _parse_spec(item: Any, *, source: str) -> IntentSpec:
    if not isinstance(item, dict) or not item.get("name"):
        raise RuleError(f"{source}: each intent needs a name")
    allowed = {
        "name",
        "description",
        "slots",
        "examples",
        "negatives",
        "patterns",
        "action",
        "say",
        "followup",
        "priority",
    }
    unknown = set(item) - allowed
    if unknown:
        raise RuleError(f"{source} ({item['name']}): unknown keys {sorted(unknown)}")
    slots = item.get("slots") or {}
    if not isinstance(slots, dict) or not all(isinstance(v, str) for v in slots.values()):
        raise RuleError(f"{source} ({item['name']}): slots must map slot name → type name")
    return IntentSpec(
        name=str(item["name"]),
        description=str(item.get("description") or ""),
        slots={str(k): v for k, v in slots.items()},
        examples=tuple(to_hk(str(x)) for x in item.get("examples") or []),
        negatives=tuple(to_hk(str(x)) for x in item.get("negatives") or []),
        patterns=tuple(to_hk(str(x)) for x in item.get("patterns") or []),
        action=item.get("action"),
        say=item.get("say"),
        followup=bool(item.get("followup", False)),
        priority=int(item.get("priority", 0)),
        source=source,
    )


def _compile(spec: IntentSpec, index: int, raw: str, types: dict[str, SlotType]) -> Rule:
    used: dict[str, SlotType] = {}

    def sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name in MACROS:
            return MACROS[name]
        if name in used:
            raise RuleError(f"{spec.name}#{index}: slot {{{name}}} used twice in one pattern")
        type_name = spec.slot_type(name)
        st = types.get(type_name)
        if st is None:
            raise RuleError(f"{spec.name}#{index}: unknown slot type {type_name!r} for {{{name}}}")
        used[name] = st
        return f"(?P<{name}>{st.pattern})"

    expanded = _PLACEHOLDER.sub(sub, raw)
    try:
        regex = re.compile(expanded)
    except re.error as e:
        raise RuleError(f"{spec.name}#{index}: bad pattern {raw!r}: {e}") from e
    return Rule(spec.name, index, raw, regex, used, spec.priority)


class RuleStore:
    """持有当前 RuleSet；文件变了就重新加载（解析失败保留旧的）；API 可以整体替换。"""

    def __init__(self, paths: Sequence[Path], *, rooms: list[str] | None = None):
        self.paths = [Path(p) for p in paths]
        self.rooms = rooms
        self._lock = threading.Lock()
        self._stamp: tuple[tuple[str, int, int], ...] = ()
        self.current = RuleSet.load(self._existing(), rooms=rooms)
        self._stamp = self._current_stamp()

    def _existing(self) -> list[Path]:
        return [p for p in self.paths if p.exists()]

    def _current_stamp(self) -> tuple[tuple[str, int, int], ...]:
        out = []
        for p in self.paths:
            try:
                st = os.stat(p)
                out.append((str(p), st.st_mtime_ns, st.st_size))
            except FileNotFoundError:
                out.append((str(p), 0, 0))
        return tuple(out)

    def reload_if_changed(self) -> bool:
        stamp = self._current_stamp()
        if stamp == self._stamp:
            return False
        with self._lock:
            if stamp == self._stamp:
                return False
            try:
                rs = RuleSet.load(self._existing(), rooms=self.rooms)
                errors, _ = rs.lint()
                if errors:
                    raise RuleError("; ".join(errors[:3]))
            except Exception as e:  # noqa: BLE001
                log.error("rules reload failed, keeping the old set: %s", e)
                self._stamp = stamp
                return False
            self.current = rs
            self._stamp = stamp
            log.info("rules reloaded: %d intents, %d patterns", len(rs.intents), len(rs.rules))
            return True

    def replace(self, rs: RuleSet) -> None:
        with self._lock:
            self.current = rs
            self._stamp = self._current_stamp()
