from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol


class ChatClient(Protocol):
    async def complete(
        self,
        *,
        user: str,
        query: str,
        conversation_id: str | None,
        files: list[dict[str, Any]] | None = None,
        inputs: dict[str, Any] | None = None,
    ) -> tuple[str, str | None]:
        """返回 (answer_text, conversation_id)。"""
        ...

    async def stream_complete(
        self,
        *,
        user: str,
        query: str,
        conversation_id: str | None,
        files: list[dict[str, Any]] | None = None,
        inputs: dict[str, Any] | None = None,
        conversation_id_holder: list[str | None],
    ) -> AsyncIterator[bytes]:
        """流式 SSE 行（``data: {json}\\n\\n``），与前端解析兼容。"""
        ...
