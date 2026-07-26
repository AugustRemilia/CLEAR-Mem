"""Construct-type router for revision regimes."""

from __future__ import annotations

from clear.models.claim import LearnerStateClaim
from clear.models.policy import Policy
from clear.types import ConstructType, RevisionRegime


CONSTRUCT_TO_REGIME = {
    ConstructType.KNOWLEDGE_STATE: RevisionRegime.QUANTIFIABLE_KNOWLEDGE,
    ConstructType.PARTIAL_MASTERY: RevisionRegime.QUANTIFIABLE_KNOWLEDGE,
    ConstructType.MISCONCEPTION: RevisionRegime.QUANTIFIABLE_KNOWLEDGE,
    ConstructType.STRATEGY: RevisionRegime.BEHAVIORAL_PREFERENCE,
    ConstructType.PREFERENCE: RevisionRegime.BEHAVIORAL_PREFERENCE,
    ConstructType.AFFECTIVE_STATE: RevisionRegime.BEHAVIORAL_PREFERENCE,
    ConstructType.BEHAVIOR_PATTERN: RevisionRegime.BEHAVIORAL_PREFERENCE,
    ConstructType.HELP_SEEKING: RevisionRegime.BEHAVIORAL_PREFERENCE,
    ConstructType.IDENTITY_OR_SENSITIVE: RevisionRegime.SENSITIVE_NEED,
    ConstructType.LANGUAGE_NEED: RevisionRegime.SENSITIVE_NEED,
    ConstructType.ACCESSIBILITY_NEED: RevisionRegime.SENSITIVE_NEED,
}


def regime_for(claim: LearnerStateClaim, policy: Policy) -> RevisionRegime:
    if not policy.use_construct_router:
        return RevisionRegime.QUANTIFIABLE_KNOWLEDGE
    return CONSTRUCT_TO_REGIME[claim.construct_type]


def revision_match(stored: LearnerStateClaim, new_ev: LearnerStateClaim, policy: Policy) -> bool:
    if not construct_types_compatible(stored.construct_type, new_ev.construct_type):
        return False
    if stored.construct_key != new_ev.construct_key:
        return False
    return scope_compatible(stored, new_ev, policy)


def scope_compatible(stored: LearnerStateClaim, new_ev: LearnerStateClaim, _policy: Policy) -> bool:
    if stored.scope_domain and new_ev.scope_domain and stored.scope_domain != new_ev.scope_domain:
        return False
    return True


def construct_types_compatible(stored: ConstructType, new_ev: ConstructType) -> bool:
    if stored == new_ev:
        return True
    mastery_family = {
        ConstructType.KNOWLEDGE_STATE,
        ConstructType.PARTIAL_MASTERY,
    }
    return stored in mastery_family and new_ev in mastery_family
