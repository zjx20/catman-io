"""OpenAI 兼容的 chat completions 客户端（第 1 层意图与 LLMBrain 共用）。"""

from .openai_compat import ChatResult, LLMError, OpenAICompatClient, ToolCall

__all__ = ["ChatResult", "LLMError", "OpenAICompatClient", "ToolCall"]
