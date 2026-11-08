"""Task 8 — session memory with LangChain InMemoryChatMessageHistory + RunnableWithMessageHistory.

In-process only (does not survive a restart), keyed by session_id. Only MASKED text is stored.
RunnableWithMessageHistory emits a LangChainDeprecationWarning pointing to LangGraph persistence;
that is expected per the brief and deliberately not silenced.
"""

from __future__ import annotations

import re
import threading
from typing import Callable

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

import app.config as config

RECORD_ID_RE = re.compile(r"\bAPT-\d{4}\b", re.IGNORECASE)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, InMemoryChatMessageHistory] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> InMemoryChatMessageHistory:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = InMemoryChatMessageHistory()
            return self._sessions[session_id]

    def recent(self, session_id: str, limit: int = config.MAX_HISTORY_MESSAGES) -> list[BaseMessage]:
        return list(self.get(session_id).messages)[-limit:]

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._sessions


def resolve_record_id(query: str, history: list[BaseMessage]) -> tuple[str | None, str]:
    """Record ID from the current query, else the most recent one mentioned in this session."""
    match = RECORD_ID_RE.search(query)
    if match:
        return match.group(0).upper(), "query"
    for message in reversed(history):
        match = RECORD_ID_RE.search(str(message.content))
        if match:
            return match.group(0).upper(), "session_memory"
    return None, "none"


def with_session_memory(core: Callable[[dict], dict], store: SessionStore) -> RunnableWithMessageHistory:
    """Wrap `core({"input": masked_query, "history": [...]}) -> {"answer": str, ...}`."""
    return RunnableWithMessageHistory(
        RunnableLambda(core),
        store.get,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="answer",
    )
