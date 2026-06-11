"""Presidio-based sanitization strategy.

Delegates sanitization to ``presidio_anonymizer.AnonymizerEngine``. Detected
PII spans are converted to ``RecognizerResult`` records and replaced with
``<LABEL>`` placeholders by default.
"""

import logging
import threading
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Union

from ..agents.registry import register_tool
from ..detection.base import PIISpan
from ..policy import PrivacyPolicy
from .base import SanitizationStrategy, select_non_overlapping

if TYPE_CHECKING:
    from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)


class PresidioOperator(str, Enum):
    """Supported Presidio ``AnonymizerEngine`` operators.

    See https://microsoft.github.io/presidio/anonymizer/ for more info.
    """

    REPLACE = "replace"
    REDACT = "redact"
    MASK = "mask"
    HASH = "hash"
    ENCRYPT = "encrypt"
    KEEP = "keep"
    CUSTOM = "custom"


DEFAULT_OPERATOR = PresidioOperator.REPLACE


_anonymizer_cache: Optional["AnonymizerEngine"] = None
_anonymizer_cache_lock = threading.Lock()


def _get_or_load_anonymizer() -> "AnonymizerEngine":
    """Lazily build and cache a Presidio ``AnonymizerEngine``."""
    global _anonymizer_cache
    if _anonymizer_cache is not None:
        return _anonymizer_cache
    with _anonymizer_cache_lock:
        if _anonymizer_cache is None:
            from presidio_anonymizer import AnonymizerEngine

            _anonymizer_cache = AnonymizerEngine()
        return _anonymizer_cache


def clear_presidio_anonymizer_cache() -> None:
    """Drop the cached anonymizer."""
    global _anonymizer_cache
    with _anonymizer_cache_lock:
        _anonymizer_cache = None


def _normalize_operator(raw: Union[str, PresidioOperator]) -> str:
    """Normalize an operator value (str or ``PresidioOperator``) to its string.

    ``PresidioOperator`` is a ``str`` enum, so its constructor accepts both a
    raw string and one of its own members; anything unknown raises ``ValueError``.
    """
    try:
        return PresidioOperator(raw).value
    except ValueError as error:
        supported = ", ".join(operator.value for operator in PresidioOperator)
        raise ValueError(
            f"Unsupported Presidio operator '{raw}'. Supported: {supported}"
        ) from error


class PresidioSanitizationStrategy(SanitizationStrategy):
    """Sanitize each chunk via Presidio's ``AnonymizerEngine``."""

    def apply(
        self,
        chunks: List[str],
        spans_per_chunk: List[List[PIISpan]],
        policy: PrivacyPolicy,
    ) -> List[str]:
        from presidio_anonymizer.entities import OperatorConfig, RecognizerResult

        anonymizer = _get_or_load_anonymizer()
        params = policy.sanitization_params or {}
        operator = _normalize_operator(params.get("operator", DEFAULT_OPERATOR))
        operator_params = params.get("operator_params", {}) or {}
        operators = {"DEFAULT": OperatorConfig(operator, operator_params)}

        out: List[str] = []
        for chunk, spans in zip(chunks, spans_per_chunk):
            kept = select_non_overlapping(list(spans))
            if not kept:
                out.append(chunk)
                continue
            recognizer_results = [
                RecognizerResult(
                    entity_type=s.label,
                    start=s.start,
                    end=s.end,
                    score=s.score,
                )
                for s in kept
            ]
            try:
                result = anonymizer.anonymize(
                    text=chunk,
                    analyzer_results=recognizer_results,
                    operators=operators,
                )
                out.append(result.text)
            except Exception as e:
                logger.warning(
                    "Presidio anonymize failed (%s), leaving chunk unchanged", e
                )
                out.append(chunk)
        return out


@register_tool("sanitize_presidio")
def sanitize_presidio(
    chunks: List[str],
    spans_per_chunk: List[List[PIISpan]],
    policy: PrivacyPolicy,
) -> List[str]:
    """Apply the default-configured Presidio anonymizer sanitization strategy."""
    return PresidioSanitizationStrategy().apply(chunks, spans_per_chunk, policy)
