"""Top-level configuration for the privacy pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional

from ..rag.llm import LLMConfig


@dataclass
class AnalyzerConfig:
    llm: LLMConfig
    system_prompt: Optional[str] = None


@dataclass
class DetectionConfig:
    engine: Optional[str] = None
    confidence_threshold: Optional[float] = None
    entity_types: List[str] = field(default_factory=list)
    llm: Optional[LLMConfig] = None


@dataclass
class SanitizationConfig:
    engine: Optional[str] = None
    strategy: Optional[str] = None
    consistency: Optional[bool] = None
    llm: Optional[LLMConfig] = None


@dataclass
class PrivacyConfig:
    domain: Optional[str] = None
    context_analyzer: Optional[AnalyzerConfig] = None
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    sanitization: SanitizationConfig = field(default_factory=SanitizationConfig)
