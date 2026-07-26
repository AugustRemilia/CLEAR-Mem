"""Fairness and sensitive-construct policy helpers."""

from __future__ import annotations

from clear.models.claim import LearnerStateClaim
from clear.types import ConstructType, UseType


SENSITIVE_OR_AFFECTIVE = {
    ConstructType.IDENTITY_OR_SENSITIVE,
    ConstructType.AFFECTIVE_STATE,
}

ACCOMMODATION_TYPES = {
    ConstructType.LANGUAGE_NEED,
    ConstructType.ACCESSIBILITY_NEED,
}

HIGH_IMPACT_PROHIBITED_FOR_SENSITIVE = {
    UseType.SCAFFOLD_LEVEL,
    UseType.SUPPORT_REDUCTION,
    UseType.CHALLENGE_ESCALATION,
    UseType.DIFFICULTY_ADAPTATION,
    UseType.LEARNING_PATH_RECOMMENDATION,
    UseType.TEACHER_ALERT,
    UseType.ABILITY_LABEL,
}


def use_is_forbidden_by_fairness(claim: LearnerStateClaim, use_type: UseType) -> str | None:
    if claim.sensitive_flags and use_type in HIGH_IMPACT_PROHIBITED_FOR_SENSITIVE:
        return "sensitive_flags_high_impact_use"
    if claim.construct_type in SENSITIVE_OR_AFFECTIVE and use_type in HIGH_IMPACT_PROHIBITED_FOR_SENSITIVE:
        return "sensitive_or_affective_high_impact_use"
    if claim.construct_type in ACCOMMODATION_TYPES and use_type in {
        UseType.DIFFICULTY_ADAPTATION,
        UseType.SUPPORT_REDUCTION,
        UseType.CHALLENGE_ESCALATION,
        UseType.ABILITY_LABEL,
    }:
        return "accommodation_cannot_lower_expectations"
    return None
