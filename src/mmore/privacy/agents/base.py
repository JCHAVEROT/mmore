"""Wrapper around LangGraph's StateGraph where each agent is a single node.
It resolves registered tools and prepends a global agent system prompt before
calling the LLM.
"""

from dataclasses import replace
from typing import Annotated, Any, Dict, List, Optional, TypedDict, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Self

from ...rag.llm import LLM
from ...utils import load_config
from .checkpointer import build_checkpointer
from .config import AgentConfig
from .registry import resolve_tools


class AgentState(TypedDict):
    """Default typed state shared by all single-node privacy agents."""

    messages: Annotated[List[BaseMessage], add_messages]


class BaseAgent:
    """Single-node LangGraph agent compiled from an AgentConfig."""

    def __init__(
        self,
        config: AgentConfig,
        llm: BaseChatModel,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        self.config = config
        self.llm = llm
        self.checkpointer = checkpointer
        self.graph = self._build_graph()

    @classmethod
    def from_config(
        cls,
        config: Union[AgentConfig, str, Dict[str, Any]],
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ) -> Self:
        if not isinstance(config, AgentConfig):
            config = load_config(config, AgentConfig)

        if checkpointer is None and config.checkpointer is not None:
            checkpointer = build_checkpointer(config)

        llm_config = replace(config.llm, temperature=config.resolve_temperature())
        llm = LLM.from_config(llm_config)

        if config.tools:
            tools = resolve_tools(config.tools)
            llm = llm.bind_tools(tools)

        return cls(config, llm, checkpointer)

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node(self.config.name, self._node)
        graph.add_edge(START, self.config.name)
        graph.add_edge(self.config.name, END)
        return graph.compile(checkpointer=self.checkpointer)

    def _node(self, state: AgentState) -> Dict[str, List[BaseMessage]]:
        messages: List[BaseMessage] = list(state["messages"])
        if self.config.system_prompt:
            messages = [SystemMessage(content=self.config.system_prompt), *messages]
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def invoke(
        self,
        query: Union[str, Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if isinstance(query, str):
            input_state: Dict[str, Any] = {"messages": [HumanMessage(content=query)]}
        else:
            input_state = query
        return self.graph.invoke(input_state, config=config)
