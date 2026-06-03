"""Presidio-based PII detection engine.

Wraps ``presidio_analyzer.AnalyzerEngine`` (rule-based and spaCy NER) with
possibility to add custom clinical recognizers.
"""

import threading
from typing import Any, List, Optional, Sequence

from typing_extensions import Self

from ..agents.registry import register_tool
from ..config import DetectionConfig
from ..domains.profile import PRESIDIO_CLINICAL_PATTERNS
from ..policy import PrivacyPolicy
from .base import DetectionEngine, PIISpan
from .constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_LANGUAGE,
)


def _build_clinical_recognizers() -> List[Any]:
    """Build the clinical-domain custom recognizers."""
    from presidio_analyzer import Pattern, PatternRecognizer

    recognizers: List[Any] = []
    for spec in PRESIDIO_CLINICAL_PATTERNS:
        recognizers.append(
            PatternRecognizer(
                supported_entity=spec["entity"],
                patterns=[
                    Pattern(name=name, regex=regex, score=score)
                    for name, regex, score in spec["patterns"]
                ],
                context=list(spec["context"]),
            )
        )
    return recognizers


def _load_presidio_analyzer() -> Any:
    """Build a ``presidio_analyzer.AnalyzerEngine`` with custom clinical recognizers."""
    from presidio_analyzer import AnalyzerEngine

    analyzer = AnalyzerEngine()
    for recognizer in _build_clinical_recognizers():
        analyzer.registry.add_recognizer(recognizer)
    return analyzer


_analyzer_cache: Optional[Any] = None
_analyzer_cache_lock = threading.Lock()


def _get_or_load_analyzer() -> Any:
    global _analyzer_cache
    if _analyzer_cache is not None:
        return _analyzer_cache
    with _analyzer_cache_lock:
        if _analyzer_cache is None:
            _analyzer_cache = _load_presidio_analyzer()
        return _analyzer_cache


def clear_presidio_cache() -> None:
    """Drop the cached analyzer."""
    global _analyzer_cache
    with _analyzer_cache_lock:
        _analyzer_cache = None


class PresidioEngine(DetectionEngine):
    """Detect PII spans with Microsoft Presidio + custom clinical recognizers.

    Each instance carries its own ``sensitive_entities`` and
    ``confidence_threshold`, the analyzer is shared across instances via
    ``_analyzer_cache``.
    """

    def __init__(
        self,
        sensitive_entities: Optional[Sequence[str]] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        language: str = DEFAULT_LANGUAGE,
    ):
        self._sensitive_entities: Optional[List[str]] = (
            list(sensitive_entities) if sensitive_entities else None
        )
        self._confidence_threshold = confidence_threshold
        self._language = language

    @classmethod
    def from_config(cls, config: DetectionConfig) -> Self:
        return cls(
            sensitive_entities=config.entity_types or None,
            confidence_threshold=(
                config.confidence_threshold
                if config.confidence_threshold is not None
                else DEFAULT_CONFIDENCE_THRESHOLD
            ),
        )

    @property
    def analyzer(self) -> Any:
        return _get_or_load_analyzer()

    def detect(self, text: str) -> List[PIISpan]:
        results = self.analyzer.analyze(
            text=text,
            language=self._language,
            entities=self._sensitive_entities,
            score_threshold=self._confidence_threshold,
        )
        return [
            PIISpan(
                start=int(r.start),
                end=int(r.end),
                label=str(r.entity_type),
                score=float(r.score),
            )
            for r in results
        ]


@register_tool("detect_pii_presidio")
def detect_pii_presidio(text: str, policy: PrivacyPolicy) -> List[PIISpan]:
    """Detect PII spans in ``text`` using a Presidio engine configured from ``policy``."""
    engine = PresidioEngine(
        sensitive_entities=policy.sensitive_entities or None,
        **policy.detection_params,
    )
    return engine.detect(text)
