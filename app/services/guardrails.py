# app/services/guardrails.py
from dataclasses import dataclass
from typing import List

@dataclass
class GuardrailDecision:
    refused: bool
    reason: str | None
    confidence: float

def decide(similarities: List[float], threshold: float) -> GuardrailDecision:
    if not similarities:
        return GuardrailDecision(True, "no_retrieval_results", 0.0)
    max_sim = max(similarities)
    conf = float(sum(similarities) / len(similarities))
    if max_sim < threshold:
        return GuardrailDecision(True, "max_similarity_below_threshold", conf)
    return GuardrailDecision(False, None, conf)
