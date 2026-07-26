from clear.context import Evidence
from clear.gates.use_authorization import authorize_use
from clear.models.claim import BeliefState, LearnerStateClaim
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
        "claim_id": "claim-use-1",
        "learner_id": "learner-1",
        "claim_text": "Student has mastered adding fractions.",
        "construct_type": ConstructType.KNOWLEDGE_STATE,
        "construct_key": "math.fractions.addition",
        "source_channel": SourceChannel.TUTOR_INFERENCE,
        "source_evidence_ids": ["turn-1", "turn-2"],
        "inference_level": InferenceLevel.LOCAL_INFERENCE,
        "scope_granularity": ScopeGranularity.SKILL,
        "confidence": 0.8,
        "admission_status": AdmissionStatus.ACTIVE,
        "belief": BeliefState(p_mastery=0.95, variance=0.01, n_evidence=3),
    }
    data.update(overrides)
    return LearnerStateClaim(**data)


def test_low_impact_use_passes_after_construct_policy():
    decision = authorize_use(make_claim(), UseType.FEEDBACK_WORDING, [], Policy())

    assert decision.authorized is True
    assert decision.reason == "low_impact_after_construct_policy"


def test_high_impact_use_requires_lower_bound():
    decision = authorize_use(make_claim(), UseType.CHALLENGE_ESCALATION, [], Policy())

    assert decision.authorized is True
    assert decision.reason == "lower_bound_rule"
    assert decision.belief_lower_bound is not None


def test_current_misconception_can_drive_scaffold_support_but_not_challenge():
    claim = make_claim(
        claim_text="Student currently confuses subtraction with addition.",
        construct_type=ConstructType.MISCONCEPTION,
        construct_key="math.word_problem.operation_direction",
        allowed_uses=[UseType.SCAFFOLD_LEVEL, UseType.CHALLENGE_ESCALATION],
        prohibited_uses=[],
        belief=BeliefState(p_mastery=0.30, variance=0.02, n_evidence=2),
    )
    evidence = [
        Evidence(
            evidence_id="turn-1",
            construct_type=ConstructType.MISCONCEPTION,
            construct_key="math.word_problem.operation_direction",
            observation=Observation.INCORRECT,
        )
    ]

    scaffold = authorize_use(claim, UseType.SCAFFOLD_LEVEL, evidence, Policy())
    challenge = authorize_use(claim, UseType.CHALLENGE_ESCALATION, evidence, Policy())

    assert scaffold.authorized is True
    assert scaffold.reason == "current_evidence_scaffold_support"
    assert challenge.authorized is False
    assert challenge.reason == "lower_bound_rule"


def test_high_impact_use_frozen_by_current_evidence_conflict():
    claim = make_claim()
    evidence = [
        Evidence(
            evidence_id="turn-3",
            construct_type=ConstructType.KNOWLEDGE_STATE,
            construct_key="math.fractions.addition",
            observation=Observation.INCORRECT,
        )
    ]

    decision = authorize_use(claim, UseType.CHALLENGE_ESCALATION, evidence, Policy())

    assert decision.authorized is False
    assert decision.reason == "frozen_by_current_evidence"


def test_sensitive_claim_denied_for_high_impact_even_if_active():
    claim = make_claim(
        claim_text="Student is anxious in mathematics.",
        construct_type=ConstructType.AFFECTIVE_STATE,
        construct_key="affect.math_anxiety",
    )

    decision = authorize_use(claim, UseType.DIFFICULTY_ADAPTATION, [], Policy())

    assert decision.authorized is False
    assert decision.reason == "sensitive_or_affective_high_impact_use"


def test_sensitive_flags_block_high_impact_use_even_when_construct_is_not_identity():
    claim = make_claim(
        claim_text="Student prefers simpler examples due to dyslexia.",
        construct_type=ConstructType.PREFERENCE,
        construct_key="preference.simple_examples",
        sensitive_flags=["disability"],
    )

    decision = authorize_use(claim, UseType.DIFFICULTY_ADAPTATION, [], Policy())

    assert decision.authorized is False
    assert decision.reason == "sensitive_flags_high_impact_use"


def test_c0_no_governance_authorizes_high_impact_sensitive_use():
    claim = make_claim(
        claim_text="Student is anxious in mathematics.",
        construct_type=ConstructType.AFFECTIVE_STATE,
        construct_key="affect.math_anxiety",
    )

    decision = authorize_use(claim, UseType.DIFFICULTY_ADAPTATION, [], Policy.for_condition("C0"))

    assert decision.authorized is True
    assert decision.reason == "no_governance_authorized"


def test_behavioral_preference_denied_for_challenge_escalation():
    claim = make_claim(
        claim_text="Student prefers visual explanations.",
        construct_type=ConstructType.PREFERENCE,
        construct_key="preference.visual_examples",
    )

    decision = authorize_use(claim, UseType.CHALLENGE_ESCALATION, [], Policy())

    assert decision.authorized is False
    assert decision.reason == "behavioral_preference_high_impact_use"


def test_non_high_impact_use_must_respect_allowed_uses():
    claim = make_claim(
        allowed_uses=[UseType.FEEDBACK_WORDING],
    )

    decision = authorize_use(claim, UseType.CORRECTIVE_FEEDBACK, [], Policy())

    assert decision.authorized is False
    assert decision.reason == "not_in_allowed_uses"


def test_preference_cannot_drive_corrective_feedback():
    claim = make_claim(
        claim_text="Student prefers visual explanations.",
        construct_type=ConstructType.PREFERENCE,
        construct_key="preference.visual_examples",
        allowed_uses=[UseType.FEEDBACK_WORDING, UseType.EXAMPLE_SELECTION, UseType.CORRECTIVE_FEEDBACK],
    )

    decision = authorize_use(claim, UseType.CORRECTIVE_FEEDBACK, [], Policy())

    assert decision.authorized is False
    assert decision.reason == "non_knowledge_corrective_feedback_use"


def test_knowledge_claim_can_drive_corrective_feedback():
    decision = authorize_use(make_claim(), UseType.CORRECTIVE_FEEDBACK, [], Policy())

    assert decision.authorized is True


def test_high_impact_use_must_respect_allowed_uses():
    claim = make_claim(allowed_uses=[UseType.FEEDBACK_WORDING])

    decision = authorize_use(claim, UseType.CHALLENGE_ESCALATION, [], Policy())

    assert decision.authorized is False
    assert decision.reason == "not_in_allowed_uses"


def test_weak_context_never_authorizes_high_impact_use():
    claim = make_claim(
        admission_status=AdmissionStatus.WEAK_CONTEXT,
        allowed_uses=[UseType.FEEDBACK_WORDING, UseType.EXAMPLE_SELECTION],
    )

    decision = authorize_use(claim, UseType.CHALLENGE_ESCALATION, [], Policy())

    assert decision.authorized is False
    assert decision.reason == "weak_context_not_for_high_impact"
