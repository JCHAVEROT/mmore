"""GLiNER-based PII detection engine."""

import logging
import threading
from typing import Any, Dict, List, Optional, Sequence

from typing_extensions import Self

from ..agents.registry import register_tool
from ..config import DetectionConfig
from ..policy import PrivacyPolicy
from .base import DetectionEngine, PIISpan
from .constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_ENTITIES,
    DEFAULT_GLINER_MODEL,
)

logger = logging.getLogger(__name__)

_model_cache: Dict[str, Any] = {}
_model_cache_lock = threading.Lock()


def _load_gliner_model(model_name: str) -> Any:
    from gliner import GLiNER

    return GLiNER.from_pretrained(model_name)


def _get_or_load_model(model_name: str) -> Any:
    cached = _model_cache.get(model_name)
    if cached is not None:
        return cached
    with _model_cache_lock:
        cached = _model_cache.get(model_name)
        if cached is None:
            cached = _load_gliner_model(model_name)
            _model_cache[model_name] = cached
        return cached


def clear_gliner_cache() -> None:
    """Drop all cached GLiNER models."""
    with _model_cache_lock:
        _model_cache.clear()


class GLiNEREngine(DetectionEngine):
    """Detect PII spans with a GLiNER model.

    Each instance carries its own ``entity_types`` and ``confidence_threshold``,
    models with the same ``model_name`` are shared via ``_models_cache``.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_GLINER_MODEL,
        sensitive_entities: Optional[Sequence[str]] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        multi_label: bool = False,
    ):
        self._model_name = model_name
        self._sensitive_entities: List[str] = (
            list(sensitive_entities) if sensitive_entities else list(DEFAULT_ENTITIES)
        )
        self._confidence_threshold = confidence_threshold
        self._multi_label = multi_label

    @classmethod
    def from_config(cls, config: DetectionConfig) -> Self:
        """Build an engine from a ``DetectionConfig``."""
        return cls(
            sensitive_entities=config.entity_types or None,
            confidence_threshold=(
                config.confidence_threshold
                if config.confidence_threshold is not None
                else DEFAULT_CONFIDENCE_THRESHOLD
            ),
        )

    @property
    def model(self) -> Any:
        """Lazy-load and cache the LLM on first access."""
        return _get_or_load_model(self._model_name)

    def detect(self, text: str) -> List[PIISpan]:
        raw = self.model.predict_entities(
            text=text,
            labels=self._sensitive_entities,
            threshold=self._confidence_threshold,
            multi_label=self._multi_label,
        )
        return [
            PIISpan(
                start=int(r["start"]),
                end=int(r["end"]),
                label=str(r["label"]),
                score=float(r["score"]),
            )
            for r in raw
        ]


@register_tool("detect_pii_gliner")
def detect_pii_gliner(text: str, policy: PrivacyPolicy) -> List[PIISpan]:
    """Detect PII spans in ``text`` using a GLiNER engine configured from ``policy``."""
    engine = GLiNEREngine(
        sensitive_entities=policy.sensitive_entities or None,
        **policy.detection_params,
    )
    return engine.detect(text)
