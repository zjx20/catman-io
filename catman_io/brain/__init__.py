"""大脑层：一般对话。``LLMBrain`` 是 catman-io 自己的 LLM 对话（流式、短期记忆）；
``CatmanClient`` 只把任务单向发给 catman（结果照旧走微信）。"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from catman_io.config import Config


class Brain(Protocol):
    def chat(self, text: str, *, session: str = "default") -> Iterator[str]:
        """流式回复：逐段文本，调用方按句切给 TTS。"""

    def cancel(self) -> None: ...


def build_brain(cfg: Config) -> Brain | None:
    c = cfg.brain.llm
    if not c.enabled:
        return None
    from catman_io.config import secret
    from catman_io.llm import OpenAICompatClient

    from .llm import LLMBrain

    if not c.model:
        raise ValueError("brain.llm.model is empty")
    client = OpenAICompatClient(
        c.base_url, api_key=secret(c.api_key_env, what="brain.llm"), model=c.model, timeout=c.timeout
    )
    return LLMBrain(
        client,
        system_prompt=c.system_prompt or None,
        memory_minutes=c.memory_minutes,
        memory_turns=c.memory_turns,
        max_tokens=c.max_tokens,
        timeout=c.timeout,
    )


__all__ = ["Brain", "build_brain"]
