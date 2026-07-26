"""Pure use-authorization gate."""

from __future__ import annotations

from clear.context import Evidence
from clear.crosscut.fairness import use_is_forbidden_by_fairness
from clear.models.claim import LearnerStateClaim
from clear.models.decision import UseDecision
from clear.models.policy import Policy
from clear.types import (
    AdmissionStatus,
    ConstructType,
    ImpactTier,
    Observation,
    UseType,
)
from clear.update.decision_layer import belief_lower_bound


IMPACT_OF = {
    UseType.FEEDBACK_WORDING: ImpactTier.LOW,
    UseType.EXAMPLE_SELECTION: ImpactTier.LOW,
    UseType.CORRECTIVE_FEEDBACK: ImpactTier.MEDIUM,
    UseType.SCAFFOLD_LEVEL: ImpactTier.HIGH,
    UseType.SUPPORT_REDUCTION: ImpactTier.HIGH,
    UseType.CHALLENGE_ESCALATION: ImpactTier.HIGH,
    UseType.DIFFICULTY_ADAPTATION: ImpactTier.HIGH,
    UseType.LEARNING_PATH_RECOMMENDATION: ImpactTier.HIGH,
    UseType.TEACHER_ALERT: ImpactTier.HIGH,
    UseType.ABILITY_LABEL: ImpactTier.HIGH,
}

QUANTIFIABLE = {
    ConstructType.KNOWLEDGE_STATE,
    ConstructType.PARTIAL_MASTERY,
    ConstructType.MISCONCEPTION,
}


def authorize_use(
    claim: LearnerStateClaim,
    use_type: UseType,
    current_evidence: list[Evidence] | None,
    policy: Policy,
) -> UseDecision:
    tier = IMPACT_OF[use_type]
    if not policy.structured_adjudication:
        return _decision(claim, use_type, tier, True, "no_governance_authorized")

    if claim.admission_status not in {AdmissionStatus.ACTIVE, AdmissionStatus.WEAK_CONTEXT}:
        return _decision(claim, use_type, tier, False, "claim_not_active")
    if claim.admission_status is AdmissionStatus.WEAK_CONTEXT and tier is ImpactTier.HIGH:
        return _decision(claim, use_type, tier, False, "weak_context_not_for_high_impact")
    if use_type in claim.prohibited_uses:
        return _decision(claim, use_type, tier, False, "prohibited_use")
    if claim.allowed_uses and use_type not in claim.allowed_uses:
        return _decision(claim, use_type, tier, False, "not_in_allowed_uses")

    construct_denial = _construct_policy_denial(claim, use_type, policy)
    if construct_denial:
        return _decision(claim, use_type, tier, False, construct_denial)

    if _conflicts_with_current_evidence(claim, current_evidence or []) and tier is ImpactTier.HIGH:
        return _decision(claim, use_type, tier, False, "frozen_by_current_evidence")

    if tier is ImpactTier.LOW:
        return _decision(claim, use_type, tier, True, "low_impact_after_construct_policy")

    if (
        use_type is UseType.SCAFFOLD_LEVEL
        and claim.construct_type in {ConstructType.MISCONCEPTION, ConstructType.PARTIAL_MASTERY}
        and _supported_by_current_evidence(claim, current_evidence or [])
    ):
        return _decision(claim, use_type, tier, True, "current_evidence_scaffold_support")

    if tier is ImpactTier.HIGH and claim.construct_type in QUANTIFIABLE:
        lower_bound = belief_lower_bound(claim.belief, policy)
        return UseDecision(
            claim_id=claim.claim_id,
            use_type=use_type,
            impact_tier=tier,
            authorized=lower_bound > policy.high_impact_threshold,
            reason="lower_bound_rule",
            required_lower_bound=policy.high_impact_threshold,
            belief_lower_bound=lower_bound,
        )

    return _decision(claim, use_type, tier, tier is not ImpactTier.HIGH, "default_tier")


def _construct_policy_denial(claim: LearnerStateClaim, use_type: UseType, policy: Policy) -> str | None:
    if not policy.construct_policy_required:
        return None
    fairness_reason = use_is_forbidden_by_fairness(claim, use_type)
    if fairness_reason:
        return fairness_reason
    if use_type is UseType.CORRECTIVE_FEEDBACK and claim.construct_type not in QUANTIFIABLE:
        return "non_knowledge_corrective_feedback_use"
    if claim.construct_type in {ConstructType.STRATEGY, ConstructType.PREFERENCE, ConstructType.HELP_SEEKING}:
        if use_type in {
            UseType.SCAFFOLD_LEVEL,
            UseType.SUPPORT_REDUCTION,
            UseType.CHALLENGE_ESCALATION,
            UseType.DIFFICULTY_ADAPTATION,
            UseType.LEARNING_PATH_RECOMMENDATION,
            UseType.ABILITY_LABEL,
        }:
            return "behavioral_preference_high_impact_use"
    return None


def _conflicts_with_current_evidence(claim: LearnerStateClaim, evidence_items: list[Evidence]) -> bool:
    for evidence in evidence_items:
        if evidence.construct_type == claim.construct_type and evidence.construct_key == claim.construct_key:
            text = claim.claim_text.lower()
            if evidence.observation == "incorrect" and ("mastered" in text or "no scaffold" in text):
                return True
            if evidence.observation == "correct" and ("cannot" in text or "does not understand" in text):
                return True
    return False


def _supported_by_current_evidence(claim: LearnerStateClaim, evidence_items: list[Evidence]) -> bool:
    evidence_ids = {evidence.evidence_id for evidence in evidence_items}
    if not set(claim.source_evidence_ids).intersection(evidence_ids):
        return False
    return any(
        evidence.construct_type == claim.construct_type
        and evidence.construct_key == claim.construct_key
        and evidence.observation in {Observation.INCORRECT, Observation.NEUTRAL}
        for evidence in evidence_items
    )


def _decision(
    claim: LearnerStateClaim,
    use_type: UseType,
    tier: ImpactTier,
    authorized: bool,
    reason: str,
) -> UseDecision:
    return UseDecision(
        claim_id=claim.claim_id,
        use_type=use_type,
        impact_tier=tier,
        authorized=authorized,
        reason=reason,
    )
