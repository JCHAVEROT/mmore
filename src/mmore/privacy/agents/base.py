"""Base class for privacy agents.

A ``BaseAgent`` is one LangGraph node. By default it calls an LLM on the
message history. Subclasses override ``state_schema`` and ``_node`` to act on
a different state (with or without an LLM), and ``node`` exposes the bound
node so several agents can be combined into one pipeline graph.
"""

import threading
from typing import (
    Annotated,
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    NamedTuple,
    Optional,
    TypedDict,
    Union,
)

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Self

from ...rag.llm import LLM, LLMConfig
from ...utils import load_config
from .checkpointer import build_checkpointer
from .config import AgentConfig
from .registry import resolve_tools


class _LLMCacheKey(NamedTuple):
    llm_name: str
    base_url: str | None
    provider: str | None
    max_new_tokens: int | None


_llm_cache: Dict[_LLMCacheKey, BaseChatModel] = {}
_llm_cache_lock = threading.Lock()


def _llm_cache_key(cfg: LLMConfig) -> _LLMCacheKey:
    return _LLMCacheKey(
        llm_name=cfg.llm_name,
        base_url=cfg.base_url,
        provider=cfg.provider,
        max_new_tokens=cfg.max_new_tokens,
    )


def _get_or_load_llm(cfg: LLMConfig) -> BaseChatModel:
    key = _llm_cache_key(cfg)
    with _llm_cache_lock:
        cached = _llm_cache.get(key)
        if cached is None:
            cached = LLM.from_config(cfg)
            _llm_cache[key] = cached
        return cached


def clear_llm_cache() -> None:
    with _llm_cache_lock:
        _llm_cache.clear()


class AgentState(TypedDict):
    """Default typed state shared by all single-node privacy agents."""

    messages: Annotated[List[BaseMessage], add_messages]


class BaseAgent:
    """Single LangGraph node compiled from a config."""

    state_schema: ClassVar[type] = AgentState
    node_name: ClassVar[Optional[str]] = None

    def __init__(
        self,
        config: Any,
        llm_config: Optional[LLMConfig] = None,
        tools: Optional[List[Callable]] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        self.config = config
        self._llm_config = llm_config
        self._tools: List[Callable] = list(tools) if tools else []
        self._llm: Optional[BaseChatModel] = None
        self.checkpointer = checkpointer
        self._owns_checkpointer = False
        self.graph = self._build_graph()

    @property
    def name(self) -> str:
        return (
            self.node_name or getattr(self.config, "name", None) or type(self).__name__
        )

    @property
    def system_prompt(self) -> str:
        return getattr(self.config, "system_prompt", "") or ""

    @classmethod
    def from_config(
        cls,
        config: Union[AgentConfig, str, Dict[str, Any]],
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ) -> Self:
        if not isinstance(config, AgentConfig):
            config = load_config(config, AgentConfig)

        owns_checkpointer = False
        if checkpointer is None and config.checkpointer is not None:
            checkpointer = build_checkpointer(config)
            owns_checkpointer = True

        tools = resolve_tools(config.tools) if config.tools else []

        agent = cls(config, config.llm, tools, checkpointer)
        agent._owns_checkpointer = owns_checkpointer
        return agent

    @property
    def llm(self) -> BaseChatModel:
        """Lazy-load and cache the LLM on first access.

        Raises:
            ValueError: if the agent has no LLM configured. An agent whose
                node never touches the LLM (e.g. the Detector) is valid.
        """
        if self._llm is None:
            if self._llm_config is None:
                raise ValueError(f"{type(self).__name__} has no LLM configured ")
            self._llm = _get_or_load_llm(self._llm_config)
        return self._llm

    def release(self) -> None:
        """Release LLM and close checkpointer resources if necessary."""
        if self._owns_checkpointer and self.checkpointer is not None:
            conn = getattr(self.checkpointer, "conn", None)
            if conn is not None:
                conn.close()
        self._llm = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        self.release()

    @property
    def node(self) -> Callable[[Any], Dict[str, Any]]:
        """The bound node callable, for composing into a larger graph."""
        return self._node

    def _build_graph(self):
        graph = StateGraph(self.state_schema)
        graph.add_node(self.name, self._node)
        graph.add_edge(START, self.name)
        graph.add_edge(self.name, END)
        return graph.compile(checkpointer=self.checkpointer)

    def _node(self, state: Any) -> Dict[str, Any]:
        messages: List[BaseMessage] = list(state["messages"])
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt), *messages]
        llm = self.llm.bind_tools(self._tools) if self._tools else self.llm
        response = llm.invoke(messages)
        return {"messages": [response]}

    def invoke(
        self,
        query: Union[str, AgentState],
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, Any]:
        """Run the agent graph on the given query.

        Args:
            query: A user message string or a pre-built state dict.
            config: Optional LangGraph runtime config.

        Returns:
            The final graph state dict.
        """
        if isinstance(query, str):
            input_state: AgentState = {"messages": [HumanMessage(content=query)]}
        else:
            input_state = query
        return self.graph.invoke(input_state, config=config)
