"""Pure admission gate for learner-state claims."""

from __future__ import annotations

from clear.context import GovernanceContext
from clear.models.claim import LearnerStateClaim
from clear.models.decision import AdmissionDecision
from clear.models.policy import Policy
from clear.types import (
    AdmissionStatus,
    ConstructType,
    InferenceLevel,
    ScopeGranularity,
    SourceChannel,
    TemporalStatus,
    UseType,
)


SENSITIVE_TYPES = {ConstructType.IDENTITY_OR_SENSITIVE}

AFFECTIVE_TYPES = {ConstructType.AFFECTIVE_STATE}

QUANTIFIABLE_TYPES = {
    ConstructType.KNOWLEDGE_STATE,
    ConstructType.PARTIAL_MASTERY,
    ConstructType.MISCONCEPTION,
}

SUPPORT_TYPES = {
    ConstructType.LANGUAGE_NEED,
    ConstructType.ACCESSIBILITY_NEED,
}


def run_admission(
    claim: LearnerStateClaim,
    ctx: GovernanceContext | None,
    policy: Policy,
) -> AdmissionDecision:
    if not policy.structured_adjudication:
        return _decision(
            claim,
            AdmissionStatus.ACTIVE,
            [],
            ["no_governance_accept"],
            permitted_uses=_initial_use_permissions(claim, policy),
        )
    context = ctx or GovernanceContext()
    failed = [criterion for criterion in policy.enabled_criteria if not CHECKS[criterion](claim, context)]

    if "schema_validity" in failed or "learner_specificity" in failed:
        reasons, risk_flags, review_required = _with_sensitive_governance(
            claim,
            ["invalid_boundary_input"],
        )
        return _decision(
            claim,
            AdmissionStatus.REJECTED,
            failed,
            reasons,
            review_required=review_required,
            risk_flags=risk_flags,
        )
    if "construct_clarity" in failed:
        reasons, risk_flags, _ = _with_sensitive_governance(
            claim,
            ["unclear_construct"],
            review_required=True,
        )
        return _decision(
            claim,
            AdmissionStatus.QUARANTINED,
            failed,
            reasons,
            review_required=True,
            risk_flags=risk_flags,
        )
    if "evidence_grounding" in failed:
        reasons, risk_flags, review_required = _with_sensitive_governance(
            claim,
            ["unsupported"],
        )
        return _decision(
            claim,
            AdmissionStatus.REJECTED,
            failed,
            reasons,
            review_required=review_required,
            risk_flags=risk_flags,
        )
    if "temporal_validity" in failed:
        reasons, risk_flags, review_required = _with_sensitive_governance(
            claim,
            ["stale_or_expired"],
        )
        return _decision(
            claim,
            AdmissionStatus.EXPIRED,
            failed,
            reasons,
            review_required=review_required,
            risk_flags=risk_flags,
        )
    if claim.construct_type in SENSITIVE_TYPES:
        reasons, risk_flags, _ = _with_sensitive_governance(
            claim,
            ["sensitive_governance"],
            review_required=True,
        )
        return _decision(
            claim,
            AdmissionStatus.QUARANTINED,
            failed,
            reasons,
            review_required=True,
            risk_flags=risk_flags,
        )
    if claim.sensitive_flags and claim.construct_type not in SUPPORT_TYPES:
        reasons, risk_flags, _ = _with_sensitive_governance(
            claim,
            ["sensitive_flags_governance"],
            review_required=True,
        )
        return _decision(
            claim,
            AdmissionStatus.QUARANTINED,
            failed,
            reasons,
            review_required=True,
            risk_flags=risk_flags,
        )
    if claim.construct_type in AFFECTIVE_TYPES:
        return _decision(
            claim,
            AdmissionStatus.WEAK_CONTEXT,
            failed,
            ["affective_state_as_weak_context"],
            permitted_uses=_low_impact_safe_uses(claim),
            review_required=claim.inference_level is InferenceLevel.GLOBAL_PROFILE_CLAIM,
            risk_flags=["affective_state_not_for_high_impact"],
        )
    if _conflicts_with_current_evidence(claim, context):
        return _decision(
            claim,
            AdmissionStatus.WEAK_CONTEXT,
            failed,
            ["current_evidence_conflict"],
            permitted_uses=_low_impact_safe_uses(claim),
            risk_flags=["current_evidence_conflict"],
        )
    if "scope_appropriateness" in failed:
        if claim.inference_level is InferenceLevel.OBSERVATION:
            reasons, risk_flags, review_required = _with_sensitive_governance(
                claim,
                ["scope_overreach_restricted"],
                risk_flags=["scope_overreach"],
            )
            return _decision(
                claim,
                AdmissionStatus.WEAK_CONTEXT,
                failed,
                reasons,
                permitted_uses=_low_impact_safe_uses(claim),
                review_required=review_required,
                risk_flags=risk_flags,
            )
        reasons, risk_flags, review_required = _with_sensitive_governance(
            claim,
            ["scope_overreach"],
        )
        return _decision(
            claim,
            AdmissionStatus.REJECTED,
            failed,
            reasons,
            review_required=review_required,
            risk_flags=risk_flags,
        )
    if claim.construct_type in SUPPORT_TYPES and not failed:
        flags = ["accommodation_not_difficulty_reduction"]
        if claim.sensitive_flags:
            flags.extend(["sensitive_accommodation_restricted", *claim.sensitive_flags])
        return _decision(
            claim,
            AdmissionStatus.ACTIVE,
            failed,
            ["accommodation_only"],
            permitted_uses=[UseType.FEEDBACK_WORDING, UseType.EXAMPLE_SELECTION],
            review_required=bool(claim.sensitive_flags),
            risk_flags=list(dict.fromkeys(flags)),
        )
    if not failed:
        return _decision(
            claim,
            AdmissionStatus.ACTIVE,
            failed,
            ["accepted"],
            permitted_uses=_initial_use_permissions(claim, policy),
        )
    if _only_weak(failed):
        reasons, risk_flags, review_required = _with_sensitive_governance(claim, ["weak_context"])
        return _decision(
            claim,
            AdmissionStatus.WEAK_CONTEXT,
            failed,
            reasons,
            permitted_uses=_low_impact_safe_uses(claim),
            review_required=review_required,
            risk_flags=risk_flags,
        )
    reasons, risk_flags, review_required = _with_sensitive_governance(claim, ["failed_admission"])
    return _decision(
        claim,
        AdmissionStatus.REJECTED,
        failed,
        reasons,
        review_required=review_required,
        risk_flags=risk_flags,
    )


def _decision(
    claim: LearnerStateClaim,
    outcome: AdmissionStatus,
    failed: list[str],
    reasons: list[str],
    permitted_uses: list[UseType] | None = None,
    review_required: bool = False,
    risk_flags: list[str] | None = None,
) -> AdmissionDecision:
    return AdmissionDecision(
        claim_id=claim.claim_id,
        outcome=outcome,
        failed_criteria=failed,
        reasons=reasons,
        channel=claim.source_channel,
        permitted_uses_initial=permitted_uses or [],
        review_required=review_required,
        risk_flags=risk_flags or [],
    )


def _with_sensitive_governance(
    claim: LearnerStateClaim,
    reasons: list[str],
    risk_flags: list[str] | None = None,
    review_required: bool = False,
) -> tuple[list[str], list[str], bool]:
    merged_reasons = list(reasons)
    merged_flags = list(risk_flags or [])
    merged_review = review_required
    sensitive_reason = _sensitive_governance_reason(claim)
    if sensitive_reason:
        merged_reasons.append(sensitive_reason)
        merged_flags.extend([sensitive_reason, *claim.sensitive_flags])
        merged_review = True
    return (
        list(dict.fromkeys(merged_reasons)),
        list(dict.fromkeys(merged_flags)),
        merged_review,
    )


def _schema_validity(claim: LearnerStateClaim, _ctx: GovernanceContext) -> bool:
    return bool(claim.claim_id and claim.learner_id and claim.claim_text and claim.construct_key)


def _learner_specificity(claim: LearnerStateClaim, _ctx: GovernanceContext) -> bool:
    lower = claim.claim_text.lower()
    rejected_markers = ("problem asks", "system should", "tutor should", "answer is")
    return not any(marker in lower for marker in rejected_markers)


def _construct_clarity(claim: LearnerStateClaim, _ctx: GovernanceContext) -> bool:
    lower = claim.claim_text.lower()
    vague = ("bad student", "weak student", "low ability", "not smart")
    return claim.construct_key.strip() != "unknown" and not any(term in lower for term in vague)


def _evidence_grounding(claim: LearnerStateClaim, _ctx: GovernanceContext) -> bool:
    ids = set(claim.source_evidence_ids)
    registry = {evidence.evidence_id for evidence in _ctx.current_evidence} | _ctx.evidence_registry_ids
    if ids:
        return ids.issubset(registry)
    return bool(claim.evidence_span and claim.source_channel in _trusted_span_channels())


def _current_consistency(claim: LearnerStateClaim, ctx: GovernanceContext) -> bool:
    return not _conflicts_with_current_evidence(claim, ctx)


def _scope_appropriateness(claim: LearnerStateClaim, _ctx: GovernanceContext) -> bool:
    if claim.source_channel in {
        SourceChannel.COMPACTION_SUMMARY,
        SourceChannel.CROSS_SESSION_SYNTHESIS,
    }:
        return claim.inference_level is not InferenceLevel.GLOBAL_PROFILE_CLAIM or len(claim.source_evidence_ids) >= 2
    if claim.scope_granularity in {ScopeGranularity.UNIT, ScopeGranularity.CROSS_SESSION}:
        return len(claim.source_evidence_ids) >= 2
    return True


def _temporal_validity(claim: LearnerStateClaim, _ctx: GovernanceContext) -> bool:
    return claim.temporal_status not in {TemporalStatus.STALE, TemporalStatus.EXPIRED}


def _governance_admissibility(claim: LearnerStateClaim, _ctx: GovernanceContext) -> bool:
    if claim.construct_type in SUPPORT_TYPES and claim.sensitive_flags:
        return True
    return not _has_sensitive_governance_signal(claim)


def _conflicts_with_current_evidence(claim: LearnerStateClaim, ctx: GovernanceContext) -> bool:
    for evidence in ctx.current_evidence:
        if evidence.construct_type == claim.construct_type and evidence.construct_key == claim.construct_key:
            text = claim.claim_text.lower()
            if evidence.observation == "incorrect" and ("mastered" in text or "no scaffold" in text):
                return True
            if evidence.observation == "correct" and ("cannot" in text or "does not understand" in text):
                return True
    return False


def _low_impact_safe_uses(claim: LearnerStateClaim) -> list[UseType]:
    if claim.construct_type in SENSITIVE_TYPES:
        return []
    return [UseType.FEEDBACK_WORDING, UseType.EXAMPLE_SELECTION]


def _initial_use_permissions(claim: LearnerStateClaim, policy: Policy) -> list[UseType]:
    if claim.allowed_uses:
        return claim.allowed_uses
    if claim.construct_type in SENSITIVE_TYPES:
        return []
    if claim.construct_type in SUPPORT_TYPES:
        return [UseType.FEEDBACK_WORDING, UseType.EXAMPLE_SELECTION]
    if claim.construct_type in QUANTIFIABLE_TYPES:
        return list(
            dict.fromkeys(
                policy.low_impact_uses
                + [UseType.CORRECTIVE_FEEDBACK]
                + policy.high_impact_uses
            )
        )
    return list(dict.fromkeys(policy.low_impact_uses + [UseType.CORRECTIVE_FEEDBACK]))


def _only_weak(failed: list[str]) -> bool:
    return set(failed).issubset({"current_consistency", "scope_appropriateness", "temporal_validity"})


CHECKS = {
    "schema_validity": _schema_validity,
    "learner_specificity": _learner_specificity,
    "construct_clarity": _construct_clarity,
    "evidence_grounding": _evidence_grounding,
    "current_consistency": _current_consistency,
    "scope_appropriateness": _scope_appropriateness,
    "temporal_validity": _temporal_validity,
    "governance_admissibility": _governance_admissibility,
}


def _trusted_span_channels() -> set[SourceChannel]:
    return {
        SourceChannel.EXPLICIT_STATEMENT,
        SourceChannel.TEACHER_NOTE,
        SourceChannel.EXTERNAL_RECORD,
    }


def _has_sensitive_governance_signal(claim: LearnerStateClaim) -> bool:
    return claim.construct_type in SENSITIVE_TYPES or bool(claim.sensitive_flags)


def _sensitive_governance_reason(claim: LearnerStateClaim) -> str | None:
    if claim.construct_type in SENSITIVE_TYPES:
        return "sensitive_governance"
    if claim.sensitive_flags:
        return "sensitive_flags_governance"
    return None
