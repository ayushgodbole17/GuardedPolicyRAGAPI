from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GuardrailDecision:
    refused: bool
    reason: Optional[str]
    confidence: float


def decide(similarities: List[float], threshold: float) -> GuardrailDecision:
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
