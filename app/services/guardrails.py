# app/services/guardrails.py

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GuardrailDecision:
    """
    Represents the outcome of similarity-based guardrail evaluation.
    """
    refused: bool
    reason: Optional[str]
    confidence: float


def decide(similarities: List[float], threshold: float) -> GuardrailDecision:
    """
    Apply similarity threshold logic.

    Refusal rule:
        If no results OR max similarity < threshold → refuse.

    Confidence:
        Mean of similarities (heuristic measure of retrieval strength).
    """

    if not similarities:
        return GuardrailDecision(
            refused=True,
            reason="no_retrieval_results",
            confidence=0.0
        )

    max_similarity = max(similarities)
    confidence = sum(similarities) / len(similarities)

    if max_similarity < threshold:
        return GuardrailDecision(
            refused=True,
            reason="max_similarity_below_threshold",
            confidence=confidence
        )

    return GuardrailDecision(
        refused=False,
        reason=None,
        confidence=confidence
    )
