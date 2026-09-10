"""LLMBrain：OpenAI 兼容端点的流式对话，带几分钟的短期记忆。"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from catman_io.llm import OpenAICompatClient

log = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "你係「貓人」，一個講廣東話嘅家居語音助手。用戶係用語音同你講嘢，你嘅回答會被讀出嚟，所以：\n"
    "- 用簡短嘅粵語口語，一至三句講完；\n"
    "- 唔好用 markdown、列表、表格、emoji 或者網址；\n"
    "- 唔好重複用戶嘅問題，直接答；\n"
    "- 唔知就話唔知，唔好作。"
)


@dataclass
class Exchange:
    at: float
    user: str
    assistant: str


class LLMBrain:
    def __init__(
        self,
        client: OpenAICompatClient,
        *,
        system_prompt: str | None = None,
        memory_minutes: float = 10.0,
        memory_turns: int = 6,
        max_tokens: int = 400,
        timeout: float = 30.0,
        temperature: float = 0.7,
        clock: Callable[[], float] = time.time,
    ):
        self.client = client
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.memory_minutes = memory_minutes
        self.memory_turns = memory_turns
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.temperature = temperature
        self.clock = clock
        self._memory: dict[str, list[Exchange]] = {}
        self._cancel = threading.Event()
        self._lock = threading.Lock()

    def messages(self, text: str, session: str) -> list[dict[str, str]]:
        msgs = [{"role": "system", "content": self.system_prompt}]
        cutoff = self.clock() - self.memory_minutes * 60
        with self._lock:
            history = [e for e in self._memory.get(session, []) if e.at >= cutoff][-self.memory_turns :]
        for e in history:
            msgs.append({"role": "user", "content": e.user})
            msgs.append({"role": "assistant", "content": e.assistant})
        msgs.append({"role": "user", "content": text})
        return msgs

    def chat(self, text: str, *, session: str = "default") -> Iterator[str]:
        self._cancel.clear()
        pieces: list[str] = []
        try:
            for piece in self.client.stream(
                self.messages(text, session),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            ):
                if self._cancel.is_set():
                    break
                pieces.append(piece)
                yield piece
        finally:
            reply = "".join(pieces).strip()
            if reply:
                with self._lock:
                    self._memory.setdefault(session, []).append(Exchange(self.clock(), text, reply))
                    del self._memory[session][: -self.memory_turns * 2]

    def cancel(self) -> None:
        self._cancel.set()

    def forget(self, session: str | None = None) -> None:
        with self._lock:
            if session is None:
                self._memory.clear()
            else:
                self._memory.pop(session, None)
