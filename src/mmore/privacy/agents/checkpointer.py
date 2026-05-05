"""Checkpoint builder using LangGraph.

``MemorySaver`` is intended for tests and ephemeral runs, and
``SqliteSaver`` for persistence across processes.
"""

import sqlite3

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from .config import AgentConfig


def build_checkpointer(config: AgentConfig) -> BaseCheckpointSaver | None:
    """Build a checkpointer from an agent config.

    Args:
        config: Agent config specifying the checkpointer type and path.

    Returns:
        A ``MemorySaver`` for ``"memory"``, a ``SqliteSaver`` for ``"sqlite"``,
        or ``None`` if no checkpointer is configured.
    """
    checkpointer = config.checkpointer
    if checkpointer is None:
        return None
    if checkpointer == "memory":
        return MemorySaver()
    if checkpointer == "sqlite":
        if not config.checkpoint_path:
            raise ValueError("'sqlite' checkpointer requires checkpoint_path to be set")
        cx = sqlite3.connect(config.checkpoint_path, check_same_thread=False)
        return SqliteSaver(cx)
    raise ValueError(
        f"Unknown checkpointer type: '{checkpointer}' (expected 'memory' or 'sqlite')"
    )
