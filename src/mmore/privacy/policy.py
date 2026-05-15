"""The request-specific privacy policy.

Emitted by the Context/Policy Analyzer and consumed downstream by the
agents in the system (Detector, Sanitizer and Adversarial agents).
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PrivacyPolicy:
    """Resolved privacy rules for a single retrieval request."""

    domain: str
    sensitive_entities: List[str]
    detection_engines: List[str]
    confidence_threshold: float
    sanitization_strategy: str
    consistency: bool
    domain_prompt: str
    redaction_strictness: str = "standard"
    sanitizer_system_prompt: str = ""
    flagged_fields: List[str] = field(default_factory=list)
