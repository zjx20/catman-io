"""进程内定时器：加 / 取消 / 到点。由主线程每帧 ``poll(now)``，到点的交给对话状态机去播。"""

from __future__ import annotations

import heapq
import itertools
import threading
from dataclasses import dataclass, field


@dataclass(order=True)
class Timer:
    due: float
    seq: int
    id: int = field(compare=False)
    seconds: int = field(compare=False)
    label: str = field(compare=False, default="")
    cancelled: bool = field(compare=False, default=False)

    @property
    def say(self) -> str:
        return f"{self.label}到喇" if self.label else "時間到喇"


class TimerService:
    def __init__(self) -> None:
        self._heap: list[Timer] = []
        self._lock = threading.Lock()
        self._seq = itertools.count()

    def add(self, now: float, seconds: int, label: str = "") -> Timer:
        t = Timer(due=now + seconds, seq=next(self._seq), id=next(self._seq), seconds=seconds, label=label)
        with self._lock:
            heapq.heappush(self._heap, t)
        return t

    def pending(self) -> list[Timer]:
        with self._lock:
            return sorted(t for t in self._heap if not t.cancelled)

    def cancel_all(self) -> int:
        with self._lock:
            n = sum(1 for t in self._heap if not t.cancelled)
            for t in self._heap:
                t.cancelled = True
            self._heap.clear()
        return n

    def poll(self, now: float) -> list[Timer]:
        """到点的定时器（按到点顺序），弹出后不再保留。"""
        fired = []
        with self._lock:
            while self._heap and self._heap[0].due <= now:
                t = heapq.heappop(self._heap)
                if not t.cancelled:
                    fired.append(t)
        return fired
