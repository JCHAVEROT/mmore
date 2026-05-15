"""Top-level configuration for the privacy pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional

from ..rag.llm import LLMConfig


@dataclass
class ContextAnalyzerConfig:
    """``privacy.context_analyzer`` block."""

    llm: LLMConfig
    system_prompt: Optional[str] = None


@dataclass
class PrivacyDetectionConfig:
    """``privacy.detection`` block.

    ``engines`` are short names (``presidio``, ``gliner``, ``openai``,
    ``llm``) resolved to registered detection tools by the Detector agent.
    ``llm`` selects the model for the ``llm`` detection engine.
    """

    engines: List[str] = field(default_factory=list)
    confidence_threshold: Optional[float] = None
    entity_types: List[str] = field(default_factory=list)
    llm: Optional[LLMConfig] = None


@dataclass
class SanitizationConfig:
    """``privacy.sanitization`` block."""

    strategy: Optional[str] = None
    consistency: Optional[bool] = None
    llm: Optional[LLMConfig] = None


@dataclass
class PrivacyConfig:
    """The ``privacy:`` block of a MMORE config."""

    domain: Optional[str] = None
    context_analyzer: Optional[ContextAnalyzerConfig] = None
    detection: PrivacyDetectionConfig = field(default_factory=PrivacyDetectionConfig)
    sanitization: SanitizationConfig = field(default_factory=SanitizationConfig)
