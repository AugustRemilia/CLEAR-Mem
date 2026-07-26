from datetime import timedelta

from clear.context import GovernanceContext
from clear.models.claim import BeliefState, LearnerStateClaim, utc_now
from clear.models.policy import Policy
from clear.types import (
    AdmissionStatus,
    ConstructType,
    InferenceLevel,
    RevisionAction,
    RevisionRegime,
    ScopeGranularity,
    SourceChannel,
)
from clear.update import apply_decay, regime_for, revise, revision_match
from clear.update.router import construct_types_compatible
from clear.update.inference_layer import bayes_update
from clear.types import Observation


def make_claim(**overrides):
    data = {
        "claim_id": "claim-rev-1",
        "learner_id": "learner-1",
        "claim_text": "Student has mastered adding fractions.",
        "construct_type": ConstructType.KNOWLEDGE_STATE,
        "construct_key": "math.fractions.addition",
        "source_channel": SourceChannel.TUTOR_INFERENCE,
        "source_evidence_ids": ["turn-1", "turn-2"],
        "inference_level": InferenceLevel.LOCAL_INFERENCE,
        "scope_granularity": ScopeGranularity.SKILL,
        "scope_domain": "math",
        "confidence": 0.8,
        "admission_status": AdmissionStatus.ACTIVE,
        "belief": BeliefState(p_mastery=0.85, variance=0.04, n_evidence=3),
    }
    data.update(overrides)
    return LearnerStateClaim(**data)


def context_for(*claims: LearnerStateClaim) -> GovernanceContext:
    return GovernanceContext(
        evidence_registry_ids={
            evidence_id
            for claim in claims
            for evidence_id in claim.source_evidence_ids
        }
    )


def test_revision_match_requires_construct_key():
    stored = make_claim()
    new_ev = make_claim(claim_id="new-1", construct_key="math.division.remainder")

    assert revision_match(stored, new_ev, Policy()) is False
    decision = revise(stored, new_ev, GovernanceContext(), Policy())
    assert decision.action is RevisionAction.NO_OP


def test_knowledge_state_and_partial_mastery_are_revision_compatible():
    assert construct_types_compatible(ConstructType.KNOWLEDGE_STATE, ConstructType.PARTIAL_MASTERY)
    assert construct_types_compatible(ConstructType.PARTIAL_MASTERY, ConstructType.KNOWLEDGE_STATE)
    assert not construct_types_compatible(ConstructType.MISCONCEPTION, ConstructType.KNOWLEDGE_STATE)


def test_rejects_override_when_new_evidence_fails_admission():
    stored = make_claim()
    new_ev = make_claim(claim_id="new-2", source_evidence_ids=[], evidence_span=None)

    decision = revise(stored, new_ev, GovernanceContext(), Policy())

    assert decision.action is RevisionAction.REJECT_OVERRIDE
    assert decision.posterior_belief == stored.belief


def test_c0_self_adjudication_supersedes_without_structural_checks():
    stored = make_claim()
    new_ev = make_claim(claim_id="new-c0", source_evidence_ids=[], belief=BeliefState(p_mastery=0.1))

    decision = revise(stored, new_ev, GovernanceContext(), Policy.for_condition("C0"))

    assert decision.action is RevisionAction.SUPERSEDE
    assert decision.reasons == ["no_governance_supersede"]


def test_c1_naive_revision_uses_last_write_wins():
    stored = make_claim()
    new_ev = make_claim(claim_id="new-c1", belief=BeliefState(p_mastery=0.1), risk_flags=["obs:incorrect"])

    decision = revise(stored, new_ev, context_for(new_ev), Policy.for_condition("C1"))

    assert decision.action is RevisionAction.SUPERSEDE
    assert decision.reasons == ["naive_last_write_wins"]


def test_single_conflicting_evidence_challenges_but_does_not_supersede():
    stored = make_claim()
    new_ev = make_claim(
        claim_id="new-3",
        claim_text="Student made an error adding fractions.",
        source_evidence_ids=["turn-3"],
        risk_flags=["obs:incorrect"],
    )

    decision = revise(stored, new_ev, context_for(new_ev), Policy())

    assert decision.action is RevisionAction.CHALLENGE
    assert decision.posterior_belief.p_mastery < stored.belief.p_mastery


def test_strong_conflicting_evidence_can_supersede():
    stored = make_claim(belief=BeliefState(p_mastery=0.95, variance=0.04, n_evidence=1))
    new_ev = make_claim(
        claim_id="new-4",
        claim_text="Student made an error adding fractions.",
        risk_flags=["obs:incorrect"],
    )
    policy = Policy(supersede_delta=0.2, supersede_min_evidence=2)

    decision = revise(stored, new_ev, context_for(new_ev), policy)

    assert decision.action is RevisionAction.SUPERSEDE


def test_misconception_retirement_is_damped_relative_to_mastery_update():
    prior = BeliefState(p_mastery=0.4, variance=0.04, n_evidence=2)
    policy = Policy()

    mastery_post = bayes_update(prior, Observation.CORRECT, policy, ConstructType.KNOWLEDGE_STATE)
    misconception_post = bayes_update(prior, Observation.CORRECT, policy, ConstructType.MISCONCEPTION)

    assert misconception_post.p_mastery < mastery_post.p_mastery
    assert misconception_post.p_mastery > prior.p_mastery


def test_behavioral_claim_confirms_without_probability_update():
    stored = make_claim(
        construct_type=ConstructType.PREFERENCE,
        construct_key="preference.visual_examples",
        claim_text="Student prefers visual examples.",
        belief=BeliefState(n_evidence=1),
    )
    new_ev = make_claim(
        claim_id="new-5",
        construct_type=ConstructType.PREFERENCE,
        construct_key="preference.visual_examples",
        claim_text="Student again requested a diagram.",
        belief=BeliefState(n_evidence=0),
    )

    decision = revise(stored, new_ev, context_for(new_ev), Policy())

    assert decision.regime is RevisionRegime.BEHAVIORAL_PREFERENCE
    assert decision.action is RevisionAction.CONFIRM
    assert decision.posterior_belief.n_evidence == 2


def test_sensitive_claim_does_not_enter_belief_update_under_full_router():
    stored = make_claim(
        construct_type=ConstructType.IDENTITY_OR_SENSITIVE,
        construct_key="identity.sensitive",
        claim_text="Student has a sensitive identity attribute.",
    )
    new_ev = make_claim(
        claim_id="new-6",
        construct_type=ConstructType.IDENTITY_OR_SENSITIVE,
        construct_key="identity.sensitive",
        claim_text="Sensitive identity evidence.",
    )

    decision = revise(stored, new_ev, context_for(new_ev), Policy.for_condition("C2"))

    assert decision.regime is RevisionRegime.SENSITIVE_NEED
    assert decision.action is RevisionAction.REJECT_OVERRIDE
    assert "sensitive_governance" in decision.reasons


def test_c2a_routes_sensitive_claim_to_uniform_quantifiable_regime():
    claim = make_claim(
        construct_type=ConstructType.IDENTITY_OR_SENSITIVE,
        construct_key="identity.sensitive",
        claim_text="Student has a sensitive identity attribute.",
    )

    assert regime_for(claim, Policy.for_condition("C2a")) is RevisionRegime.QUANTIFIABLE_KNOWLEDGE


def test_c2a_applies_uniform_quantifiable_update_to_preference_claim():
    stored = make_claim(
        construct_type=ConstructType.PREFERENCE,
        construct_key="preference.visual_examples",
        claim_text="Student prefers visual examples.",
        belief=BeliefState(n_evidence=1),
    )
    new_ev = make_claim(
        claim_id="new-c2a-pref",
        construct_type=ConstructType.PREFERENCE,
        construct_key="preference.visual_examples",
        claim_text="Student again showed correct performance with visual examples.",
        risk_flags=["obs:correct"],
    )

    ctx = context_for(new_ev)
    full = revise(stored, new_ev, ctx, Policy.for_condition("C2"))
    c2a = revise(stored, new_ev, ctx, Policy.for_condition("C2a"))

    assert full.regime is RevisionRegime.BEHAVIORAL_PREFERENCE
    assert full.action is RevisionAction.CONFIRM
    assert c2a.regime is RevisionRegime.QUANTIFIABLE_KNOWLEDGE
    assert c2a.action is not full.action
    assert c2a.posterior_belief.p_mastery is not None


def test_behavioral_revision_rejects_unsupported_new_evidence():
    stored = make_claim(
        construct_type=ConstructType.PREFERENCE,
        construct_key="preference.visual_examples",
        claim_text="Student prefers visual examples.",
        belief=BeliefState(n_evidence=1),
    )
    new_ev = make_claim(
        claim_id="new-unsupported-pref",
        construct_type=ConstructType.PREFERENCE,
        construct_key="preference.visual_examples",
        claim_text="Student again requested a diagram.",
        source_evidence_ids=[],
        evidence_span=None,
        belief=BeliefState(n_evidence=0),
    )

    decision = revise(stored, new_ev, GovernanceContext(), Policy.for_condition("C2"))

    assert decision.action is RevisionAction.REJECT_OVERRIDE
    assert "new_evidence_failed_admission" in decision.reasons


def test_c1_checks_admission_before_last_write_wins():
    stored = make_claim()
    new_ev = make_claim(claim_id="new-c1-unsupported", source_evidence_ids=[], evidence_span=None)

    decision = revise(stored, new_ev, GovernanceContext(), Policy.for_condition("C1"))

    assert decision.action is RevisionAction.REJECT_OVERRIDE
    assert "new_evidence_failed_admission" in decision.reasons


def test_decay_inflates_variance_without_moving_mean_and_can_expire():
    old_time = utc_now() - timedelta(days=10)
    claim = make_claim(
        belief=BeliefState(p_mastery=0.6, variance=0.04, n_evidence=3),
        updated_at=old_time,
    )
    policy = Policy(decay_rate=0.1, reconfirm_threshold=0.5)

    decision = apply_decay(claim, utc_now(), policy)

    assert decision is not None
    assert decision.action is RevisionAction.DECAY_EXPIRE
    assert decision.posterior_belief.p_mastery == claim.belief.p_mastery
    assert decision.posterior_belief.variance > claim.belief.variance
