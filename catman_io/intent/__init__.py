"""意图识别：三层。

0. 规则（``rules.py``）：YAML 里的正则 + 槽位解析，零延迟、可离线。
1. LLM（``llm.py``）：OpenAI 兼容端点一次 function calling，规则没中才问。
2. 大脑：闲聊 / 任务交给 ``catman_io.brain``。

``router.py`` 把三层串起来；``cases.py`` 是回归用例；规则文件在 ``rules/builtin.yaml``（随包）
与现场目录（catman 改的那份）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from catman_io.config import Config

    from .rules import RuleSet, RuleStore

BUILTIN_RULES = Path(__file__).parent / "rules" / "builtin.yaml"
SITE_RULES_NAME = "site.yaml"


@dataclass
class Intent:
    name: str
    slots: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    tier: str = "rule"  # rule | llm | none
    rule_id: str | None = None
    raw: str | None = None  # 命中的那段文本，或 LLM 的原始输出

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IntentRecognizer(Protocol):
    def recognize(self, text: str) -> Intent | None: ...


def rule_paths(cfg: Config) -> list[Path]:
    """builtin 在前，现场目录里的 *.yaml 按名字排序在后（同名意图后者覆盖）。"""
    paths = [BUILTIN_RULES]
    d = cfg.rules_dir
    if d.exists():
        paths += sorted(p for p in d.glob("*.yaml") if p.is_file())
    else:
        paths.append(d / SITE_RULES_NAME)  # 还不存在也登记，之后创建了会被热加载
    return paths


def load_rules(cfg: Config) -> RuleSet:
    from .rules import RuleSet

    return RuleSet.load([p for p in rule_paths(cfg) if p.exists()], rooms=cfg.intent.context.rooms or None)


def rule_store(cfg: Config) -> RuleStore:
    from .rules import RuleStore

    return RuleStore(BUILTIN_RULES, cfg.rules_dir, rooms=cfg.intent.context.rooms or None)


__all__ = ["BUILTIN_RULES", "Intent", "IntentRecognizer", "load_rules", "rule_paths", "rule_store"]
