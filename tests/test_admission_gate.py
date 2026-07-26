from clear.context import Evidence, GovernanceContext
from clear.gates.admission import run_admission
from clear.models.claim import LearnerStateClaim
from clear.models.policy import Policy
from clear.types import (
    AdmissionStatus,
    ConstructType,
    InferenceLevel,
    Observation,
    ScopeGranularity,
    SourceChannel,
    UseType,
)


def make_claim(**overrides):
    data = {
        "claim_id": "claim-admit-1",
        "learner_id": "learner-1",
        "claim_text": "Student has partially mastered adding fractions.",
        "construct_type": ConstructType.PARTIAL_MASTERY,
        "construct_key": "math.fractions.addition",
        "source_channel": SourceChannel.TUTOR_INFERENCE,
        "source_evidence_ids": ["turn-1"],
        "inference_level": InferenceLevel.LOCAL_INFERENCE,
        "scope_granularity": ScopeGranularity.SKILL,
        "confidence": 0.7,
    }
    data.update(overrides)
    return LearnerStateClaim(**data)


def context_for(claim: LearnerStateClaim, current_evidence: list[Evidence] | None = None) -> GovernanceContext:
    evidence = current_evidence or []
    return GovernanceContext(
        current_evidence=evidence,
        evidence_registry_ids=set(claim.source_evidence_ids) | {item.evidence_id for item in evidence},
    )


def test_admission_accepts_evidence_grounded_claim():
    claim = make_claim()
    decision = run_admission(claim, context_for(claim), Policy())

    assert decision.outcome is AdmissionStatus.ACTIVE
    assert UseType.CORRECTIVE_FEEDBACK in decision.permitted_uses_initial


def test_admission_rejects_unsupported_claim():
    claim = make_claim(source_evidence_ids=[], evidence_span=None)
    decision = run_admission(claim, GovernanceContext(), Policy())

    assert decision.outcome is AdmissionStatus.REJECTED
    assert decision.failed_criteria == ["evidence_grounding"]


def test_c0_no_governance_accepts_unsupported_claim():
    claim = make_claim(source_evidence_ids=[], evidence_span=None)
    decision = run_admission(claim, GovernanceContext(), Policy.for_condition("C0"))

    assert decision.outcome is AdmissionStatus.ACTIVE
    assert decision.reasons == ["no_governance_accept"]


def test_admission_quarantines_sensitive_claim_before_ordinary_use():
    claim = make_claim(
        construct_type=ConstructType.IDENTITY_OR_SENSITIVE,
        construct_key="identity.religion",
        claim_text="Student belongs to a sensitive identity group.",
    )
    decision = run_admission(claim, context_for(claim), Policy())

    assert decision.outcome is AdmissionStatus.QUARANTINED
    assert decision.review_required is True
    assert "sensitive_governance" in decision.risk_flags


def test_admission_downgrades_current_conflict_to_weak_context():
    claim = make_claim(
        claim_text="Student has mastered fractions and needs no scaffolding.",
        construct_type=ConstructType.KNOWLEDGE_STATE,
        construct_key="math.fractions.addition",
    )
    current_evidence = [
        Evidence(
            evidence_id="turn-2",
            construct_type=ConstructType.KNOWLEDGE_STATE,
            construct_key="math.fractions.addition",
            observation=Observation.INCORRECT,
        )
    ]
    ctx = context_for(claim, current_evidence)

    decision = run_admission(claim, ctx, Policy())

    assert decision.outcome is AdmissionStatus.WEAK_CONTEXT
    assert "current_evidence_conflict" in decision.risk_flags


def test_admission_rejects_summary_laundering_global_claim_with_single_evidence():
    claim = make_claim(
        source_channel=SourceChannel.COMPACTION_SUMMARY,
        inference_level=InferenceLevel.GLOBAL_PROFILE_CLAIM,
        scope_granularity=ScopeGranularity.CROSS_SESSION,
        source_evidence_ids=["turn-1"],
    )

    decision = run_admission(claim, context_for(claim), Policy())

    assert decision.outcome is AdmissionStatus.REJECTED
    assert "scope_appropriateness" in decision.failed_criteria


def test_support_need_still_respects_temporal_validity():
    claim = make_claim(
        construct_type=ConstructType.LANGUAGE_NEED,
        construct_key="language.plain_explanation",
        claim_text="Student needs plain-language explanation.",
        temporal_status="stale",
    )

    decision = run_admission(claim, context_for(claim), Policy())

    assert decision.outcome is AdmissionStatus.EXPIRED


def test_evidence_grounding_rejects_unknown_evidence_ids():
    claim = make_claim(source_evidence_ids=["fake-turn-999"])
    ctx = GovernanceContext(evidence_registry_ids={"turn-1"})

    decision = run_admission(claim, ctx, Policy())

    assert decision.outcome is AdmissionStatus.REJECTED
    assert decision.failed_criteria == ["evidence_grounding"]


def test_stale_sensitive_claim_preserves_both_reasons():
    claim = make_claim(
        construct_type=ConstructType.IDENTITY_OR_SENSITIVE,
        construct_key="identity.sensitive",
        claim_text="Student has a sensitive identity attribute.",
        temporal_status="stale",
    )

    decision = run_admission(claim, context_for(claim), Policy())

    assert decision.outcome is AdmissionStatus.EXPIRED
    assert "stale_or_expired" in decision.reasons
    assert "sensitive_governance" in decision.reasons


def test_affective_state_admitted_as_weak_context_not_quarantined_by_default():
    claim = make_claim(
        construct_type=ConstructType.AFFECTIVE_STATE,
        construct_key="affect.frustration",
        claim_text="Student appears frustrated with fractions.",
    )

    decision = run_admission(claim, context_for(claim), Policy())

    assert decision.outcome is AdmissionStatus.WEAK_CONTEXT
    assert decision.review_required is False
    assert decision.permitted_uses_initial == [UseType.FEEDBACK_WORDING, UseType.EXAMPLE_SELECTION]
    assert "affective_state_not_for_high_impact" in decision.risk_flags


def test_sensitive_flags_trigger_admission_governance_even_when_construct_is_preference():
    claim = make_claim(
        claim_text="Student prefers simpler examples due to dyslexia.",
        construct_type=ConstructType.PREFERENCE,
        construct_key="preference.simple_examples",
        sensitive_flags=["disability"],
    )

    decision = run_admission(claim, context_for(claim), Policy())

    assert decision.outcome is AdmissionStatus.QUARANTINED
    assert decision.review_required is True
    assert "sensitive_flags_governance" in decision.risk_flags
    assert "disability" in decision.risk_flags


def test_sensitive_support_need_is_active_but_restricted():
    claim = make_claim(
        claim_text="Student needs accessible visual formatting.",
        construct_type=ConstructType.ACCESSIBILITY_NEED,
        construct_key="accessibility.visual_formatting",
        sensitive_flags=["disability"],
    )

    decision = run_admission(claim, context_for(claim), Policy())

    assert decision.outcome is AdmissionStatus.ACTIVE
    assert decision.review_required is True
    assert decision.permitted_uses_initial == [UseType.FEEDBACK_WORDING, UseType.EXAMPLE_SELECTION]
    assert "sensitive_accommodation_restricted" in decision.risk_flags


def test_evidence_grounding_rejection_preserves_secondary_failed_criteria():
    claim = make_claim(
        source_evidence_ids=["fake"],
        scope_granularity=ScopeGranularity.CROSS_SESSION,
    )

    decision = run_admission(claim, GovernanceContext(evidence_registry_ids={"real"}), Policy())

    assert decision.outcome is AdmissionStatus.REJECTED
    assert "evidence_grounding" in decision.failed_criteria
    assert "scope_appropriateness" in decision.failed_criteria


def test_unsupported_sensitive_claim_preserves_sensitive_risk_flags():
    claim = make_claim(
        construct_type=ConstructType.IDENTITY_OR_SENSITIVE,
        construct_key="identity.sensitive",
        claim_text="Student has a sensitive identity attribute.",
        source_evidence_ids=[],
        evidence_span=None,
    )

    decision = run_admission(claim, GovernanceContext(), Policy())

    assert decision.outcome is AdmissionStatus.REJECTED
    assert "unsupported" in decision.reasons
    assert "sensitive_governance" in decision.reasons
    assert "sensitive_governance" in decision.risk_flags
    assert decision.review_required is True


def test_unsupported_sensitive_flagged_preference_preserves_sensitive_risk_flags():
    claim = make_claim(
        construct_type=ConstructType.PREFERENCE,
        construct_key="preference.simple_examples",
        claim_text="Student prefers simpler examples due to dyslexia.",
        source_evidence_ids=[],
        evidence_span=None,
        sensitive_flags=["disability"],
    )

    decision = run_admission(claim, GovernanceContext(), Policy())

    assert decision.outcome is AdmissionStatus.REJECTED
    assert "unsupported" in decision.reasons
    assert "sensitive_flags_governance" in decision.reasons
    assert "sensitive_flags_governance" in decision.risk_flags
    assert "disability" in decision.risk_flags
    assert decision.review_required is True


def test_construct_clarity_rejection_preserves_sensitive_risk_flags():
    claim = make_claim(
        construct_type=ConstructType.PREFERENCE,
        construct_key="unknown",
        claim_text="Student is a weak student due to dyslexia.",
        sensitive_flags=["disability"],
    )

    decision = run_admission(claim, context_for(claim), Policy())

    assert decision.outcome is AdmissionStatus.QUARANTINED
    assert "unclear_construct" in decision.reasons
    assert "sensitive_flags_governance" in decision.reasons
    assert "sensitive_flags_governance" in decision.risk_flags
    assert "disability" in decision.risk_flags
    assert decision.review_required is True


def test_learner_boundary_rejection_preserves_sensitive_risk_flags():
    claim = make_claim(
        construct_type=ConstructType.PREFERENCE,
        construct_key="preference.simple_examples",
        claim_text="The answer is 12, and student needs simpler examples due to dyslexia.",
        sensitive_flags=["disability"],
    )

    decision = run_admission(claim, context_for(claim), Policy())

    assert decision.outcome is AdmissionStatus.REJECTED
    assert "invalid_boundary_input" in decision.reasons
    assert "sensitive_flags_governance" in decision.reasons
    assert "sensitive_flags_governance" in decision.risk_flags
    assert "disability" in decision.risk_flags
    assert decision.review_required is True


def test_scope_overreach_rejection_preserves_sensitive_support_flags():
    claim = make_claim(
        claim_text="Student needs accessible visual formatting across future sessions.",
        construct_type=ConstructType.ACCESSIBILITY_NEED,
        construct_key="accessibility.visual_formatting",
        source_channel=SourceChannel.COMPACTION_SUMMARY,
        inference_level=InferenceLevel.GLOBAL_PROFILE_CLAIM,
        scope_granularity=ScopeGranularity.CROSS_SESSION,
        sensitive_flags=["disability"],
    )

    decision = run_admission(claim, context_for(claim), Policy())

    assert decision.outcome is AdmissionStatus.REJECTED
    assert "scope_overreach" in decision.reasons
    assert "sensitive_flags_governance" in decision.reasons
    assert "sensitive_flags_governance" in decision.risk_flags
    assert "disability" in decision.risk_flags
    assert decision.review_required is True
