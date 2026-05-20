"""Shared state for the privacy pipeline graph.

A single ``StateGraph(PrivacyState)`` flows through analyzer -> detector ->
sanitizer (and more later). Each agent contributes a node that reads what
it needs and writes its output back.
"""

from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from ..detection.base import PIISpan
from ..policy import PrivacyPolicy
from ..risk import RiskAssessment


class PrivacyState(TypedDict, total=False):
    """State carried through the privacy pipeline."""

    query: str
    raw_chunks: List[str]
    messages: Annotated[List[BaseMessage], add_messages]
    policy: Optional[PrivacyPolicy]
    spans: List[List[PIISpan]]
    risk: Optional[RiskAssessment]
    sanitized_chunks: List[str]
