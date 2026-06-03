"""HuggingFace ``openai/privacy-filter`` PII detection engine."""

import logging
import threading
from typing import Any, Dict, List

from typing_extensions import Self

from ..agents.registry import register_tool
from ..config import DetectionConfig
from ..policy import PrivacyPolicy
from .base import DetectionEngine, PIISpan
from .constants import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_OPENAI_FILTER_MODEL

logger = logging.getLogger(__name__)


def _load_openai_filter_pipeline(model_name: str) -> Any:
    from transformers import pipeline

    return pipeline(
        task="token-classification",
        model=model_name,
    )


_pipeline_cache: Dict[str, Any] = {}
_pipeline_cache_lock = threading.Lock()


def _get_or_load_pipeline(model_name: str) -> Any:
    cached = _pipeline_cache.get(model_name)
    if cached is not None:
        return cached
    with _pipeline_cache_lock:
        cached = _pipeline_cache.get(model_name)
        if cached is None:
            cached = _load_openai_filter_pipeline(model_name)
            _pipeline_cache[model_name] = cached
        return cached


def clear_openai_filter_cache() -> None:
    """Drop all cached HF pipelines."""
    with _pipeline_cache_lock:
        _pipeline_cache.clear()


class OpenAIFilterEngine(DetectionEngine):
    """Detect PII spans with the token classification model
    ``openai/privacy-filter`` from HuggingFace.

    The model has a fixed label set so entity selection is not configurable.
    Each instance carries its own ``confidence_threshold``, pipelines with the
    same ``model_name`` are shared via ``_pipeline_cache``.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_OPENAI_FILTER_MODEL,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        self._model_name = model_name
        self._confidence_threshold = confidence_threshold

    @classmethod
    def from_config(cls, config: DetectionConfig) -> Self:
        return cls(
            confidence_threshold=(
                config.confidence_threshold
                if config.confidence_threshold is not None
                else DEFAULT_CONFIDENCE_THRESHOLD
            ),
        )

    @property
    def pipeline(self) -> Any:
        return _get_or_load_pipeline(self._model_name)

    def detect(self, text: str) -> List[PIISpan]:
        raw = self.pipeline(text)
        spans: List[PIISpan] = []
        for r in raw:
            score = float(r["score"])
            if score < self._confidence_threshold:
                continue
            label = str(r.get("entity_group") or r.get("entity") or "")
            spans.append(
                PIISpan(
                    start=int(r["start"]),
                    end=int(r["end"]),
                    label=label,
                    score=score,
                )
            )
        return spans


@register_tool("detect_pii_openai_filter")
def detect_pii_openai_filter(text: str, policy: PrivacyPolicy) -> List[PIISpan]:
    """Detect PII spans in ``text`` using an openai/privacy-filter engine
    configured from ``policy``."""
    if policy.sensitive_entities:
        logger.warning("OpenAi Filter has a fixed sensitive label lists.")
    engine = OpenAIFilterEngine(**policy.detection_params)
    return engine.detect(text)
