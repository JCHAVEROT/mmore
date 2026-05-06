"""Checkpoint builder using LangGraph.

``MemorySaver`` is intended for tests and temporary runs, and
``SqliteSaver`` for persistence across processes.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

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
        path = Path(config.checkpoint_path)
        path.parent.mkdir(exist_ok=True, parents=True)
        cx = sqlite3.connect(path, check_same_thread=False)
        return SqliteSaver(cx)
    raise ValueError(
        f"Unknown checkpointer type: '{checkpointer}' (expected 'memory' or 'sqlite')"
    )


@contextmanager
def open_checkpointer(config: AgentConfig):
    """Build a checkpointer to share across multiple agents and close its
    connection on exit.

    Example:
        >>> with open_checkpointer(config) as cp:
        ...     a = BaseAgent.from_config(cfg_a, checkpointer=cp)
        ...     b = BaseAgent.from_config(cfg_b, checkpointer=cp)
    """
    cp = build_checkpointer(config)
    try:
        yield cp
    finally:
        if cp is not None:
            conn = getattr(cp, "conn", None)
            if conn is not None:
                conn.close()
