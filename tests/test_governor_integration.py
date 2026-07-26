import pytest

from clear.context import Evidence, GovernanceContext
from clear.crosscut.audit import AuditTrail
from clear.governor import Governor, _hash_claim
from clear.models.claim import utc_now
from clear.models.learner_model import DuplicateClaimError, LearnerModel
from clear.models.policy import Policy
from clear.types import (
    AdmissionStatus,
    AuditEventType,
    ConditionId,
    ConstructType,
    Observation,
    ReviewStatus,
    RevisionAction,
    UseType,
)
from tests.factories import claim


def context_for_claims(*claims):
    return GovernanceContext(
        evidence_registry_ids={
            evidence_id
            for claim_obj in claims
            for evidence_id in claim_obj.source_evidence_ids
        }
    )


def test_governor_ingest_writes_store_and_audit_only_after_admission():
    audit = AuditTrail()
    governor = Governor(Policy.for_condition(ConditionId.C2), LearnerModel(), audit)
    ok = claim("ok-1", "Student has partially mastered adding fractions.")
    bad = claim("bad-1", "Unsupported mastery claim.", evidence_ids=[])

    decisions = governor.ingest([ok, bad], context_for_claims(ok))

    assert [decision.outcome for decision in decisions] == [
        AdmissionStatus.ACTIVE,
        AdmissionStatus.REJECTED,
    ]
    assert len(governor.store.all_active("learner-1")) == 1
    assert [event.event_type for event in audit.events] == [
        AuditEventType.ADMISSION,
        AuditEventType.ADMISSION,
    ]


def test_governor_request_use_logs_high_impact_denial_from_current_evidence():
    audit = AuditTrail()
    governor = Governor(Policy.for_condition("C2"), audit=audit)
    stored_claim = claim("ok-2", "Student has mastered adding fractions.").model_copy(
        update={"allowed_uses": [UseType.CHALLENGE_ESCALATION]}
    )
    governor.ingest([stored_claim], context_for_claims(stored_claim))

    decisions = governor.request_use(
        "learner-1",
        UseType.CHALLENGE_ESCALATION,
        [
            Evidence(
                evidence_id="turn-3",
                construct_type=ConstructType.KNOWLEDGE_STATE,
                construct_key="math.fractions.addition",
                observation=Observation.INCORRECT,
            )
        ],
    )

    assert decisions[0].authorized is False
    assert decisions[0].reason == "frozen_by_current_evidence"
    assert audit.events[-1].event_type is AuditEventType.USE_AUTHORIZATION
    assert audit.events[-1].decision == "denied"


def test_ingest_then_high_impact_knowledge_use_reaches_lower_bound_rule():
    audit = AuditTrail()
    governor = Governor(Policy.for_condition("C2"), audit=audit)
    stored_claim = claim(
        "ok-high-impact",
        "Student has mastered adding fractions.",
        evidence_ids=["turn-1", "turn-2"],
        p_mastery=0.95,
    )
    governor.ingest([stored_claim], context_for_claims(stored_claim))

    decisions = governor.request_use("learner-1", UseType.CHALLENGE_ESCALATION)

    assert decisions[0].authorized is True
    assert decisions[0].reason == "lower_bound_rule"
    assert audit.events[-1].permitted_uses_after
    assert UseType.CHALLENGE_ESCALATION in audit.events[0].permitted_uses_after


def test_governor_revision_updates_store_and_logs_revision():
    audit = AuditTrail()
    governor = Governor(Policy.for_condition("C2"), audit=audit)
    stored_claim = claim("ok-3", "Student has mastered adding fractions.")
    governor.ingest([stored_claim], context_for_claims(stored_claim))
    new_ev = claim(
        "new-ev-1",
        "Student made an error adding fractions.",
        evidence_ids=["turn-4"],
        risk_flags=["obs:incorrect"],
    )

    decisions = governor.revise_with("learner-1", new_ev, context_for_claims(new_ev))
    stored = governor.store.get("ok-3")

    assert decisions[0].action is RevisionAction.CHALLENGE
    assert stored.admission_status is AdmissionStatus.WEAK_CONTEXT
    assert audit.events[-1].event_type is AuditEventType.REVISION
    assert audit.events[-1].decision == "challenge"
    assert audit.events[-1].admission_status_after is stored.admission_status


def test_supersede_marks_old_claim_superseded_and_adds_new_claim():
    audit = AuditTrail()
    governor = Governor(Policy(supersede_delta=0.2, supersede_min_evidence=2), audit=audit)
    stored_claim = claim(
        "old-mastery",
        "Student has mastered adding fractions.",
        evidence_ids=["turn-1", "turn-2"],
        p_mastery=0.95,
    )
    governor.ingest([stored_claim], context_for_claims(stored_claim))
    new_ev = claim(
        "new-errors",
        "Student made repeated errors adding fractions.",
        evidence_ids=["turn-3", "turn-4"],
        risk_flags=["obs:incorrect"],
        p_mastery=0.2,
    )

    decisions = governor.revise_with("learner-1", new_ev, context_for_claims(new_ev))
    old = governor.store.get("old-mastery")
    replacement = governor.store.get("new-errors")
    active_ids = {claim_obj.claim_id for claim_obj in governor.store.all_active("learner-1")}

    assert decisions[0].action is RevisionAction.SUPERSEDE
    assert old.admission_status is AdmissionStatus.SUPERSEDED
    assert old.claim_text == "Student has mastered adding fractions."
    assert replacement.admission_status is AdmissionStatus.ACTIVE
    assert replacement.claim_text == "Student made repeated errors adding fractions."
    assert replacement.revision_of == "old-mastery"
    assert "old-mastery" in replacement.parent_claim_ids
    assert "old-mastery" not in active_ids
    assert "new-errors" in active_ids
    assert audit.events[-1].replacement_claim_id == "new-errors"
    assert audit.events[-1].admission_status_after is AdmissionStatus.SUPERSEDED


def test_weak_context_new_evidence_does_not_supersede_active_claim():
    audit = AuditTrail()
    governor = Governor(Policy(supersede_delta=0.2, supersede_min_evidence=2), audit=audit)
    stored_claim = claim(
        "old-struggle",
        "Student does not understand adding fractions.",
        evidence_ids=["turn-1", "turn-2"],
        p_mastery=0.1,
    )
    governor.ingest([stored_claim], context_for_claims(stored_claim))
    new_ev = claim(
        "new-weak-mastery",
        "Student has mastered adding fractions.",
        evidence_ids=["turn-3", "turn-4"],
        risk_flags=["obs:correct"],
        p_mastery=0.9,
    )
    conflicting_current = Evidence(
        evidence_id="turn-current",
        construct_type=ConstructType.KNOWLEDGE_STATE,
        construct_key="math.fractions.addition",
        observation=Observation.INCORRECT,
    )
    ctx = GovernanceContext(
        current_evidence=[conflicting_current],
        evidence_registry_ids={"turn-3", "turn-4", "turn-current"},
    )

    decisions = governor.revise_with("learner-1", new_ev, ctx)
    old = governor.store.get("old-struggle")
    replacement = governor.store.get("new-weak-mastery")

    assert decisions[0].action is RevisionAction.CHALLENGE
    assert "new_evidence_weak_context_not_superseding" in decisions[0].reasons
    assert old.admission_status is AdmissionStatus.WEAK_CONTEXT
    assert replacement is None
    assert audit.events[-1].replacement_claim_id is None


def test_revision_with_multiple_current_same_construct_does_not_duplicate_replacement():
    audit = AuditTrail()
    governor = Governor(Policy(supersede_delta=0.2, supersede_min_evidence=2), audit=audit)
    old_one = claim(
        "old-multi-1",
        "Student has mastered adding fractions.",
        evidence_ids=["turn-1", "turn-2"],
        p_mastery=0.85,
    )
    old_two = claim(
        "old-multi-2",
        "Student is confident adding fractions.",
        evidence_ids=["turn-3", "turn-4"],
        p_mastery=0.8,
    )
    governor.ingest([old_one, old_two], context_for_claims(old_one, old_two))
    new_ev = claim(
        "new-multi",
        "Student made repeated errors adding fractions.",
        evidence_ids=["turn-5", "turn-6"],
        risk_flags=["obs:incorrect"],
        p_mastery=0.2,
    )

    decisions = governor.revise_with("learner-1", new_ev, context_for_claims(new_ev))
    replacement = governor.store.get("new-multi")
    active_ids = {claim_obj.claim_id for claim_obj in governor.store.all_active("learner-1")}
    revision_events = [event for event in audit.events if event.event_type is AuditEventType.REVISION]

    assert [decision.action for decision in decisions] == [
        RevisionAction.SUPERSEDE,
        RevisionAction.SUPERSEDE,
    ]
    assert governor.store.get("old-multi-1").admission_status is AdmissionStatus.SUPERSEDED
    assert governor.store.get("old-multi-2").admission_status is AdmissionStatus.SUPERSEDED
    assert replacement is not None
    assert set(replacement.parent_claim_ids) >= {"old-multi-1", "old-multi-2"}
    assert active_ids == {"new-multi"}
    assert [event.replacement_claim_id for event in revision_events] == ["new-multi", "new-multi"]


def test_review_required_admission_sets_store_review_status():
    audit = AuditTrail()
    governor = Governor(Policy.for_condition("C2"), audit=audit)
    support_need = claim(
        "support-review",
        "Student needs accessible visual formatting.",
        construct_type=ConstructType.ACCESSIBILITY_NEED,
        construct_key="accessibility.visual_formatting",
        p_mastery=None,
    ).model_copy(update={"sensitive_flags": ["disability"]})

    governor.ingest([support_need], context_for_claims(support_need))
    stored = governor.store.get("support-review")

    assert stored.review_status is ReviewStatus.TEACHER_REVIEW_NEEDED
    assert "review_required" in stored.risk_flags
    assert audit.events[-1].review_required is True


def test_second_revision_after_supersede_does_not_duplicate_replacement():
    audit = AuditTrail()
    governor = Governor(Policy(supersede_delta=0.2, supersede_min_evidence=2), audit=audit)
    stored_claim = claim(
        "old-chain",
        "Student has mastered adding fractions.",
        evidence_ids=["turn-1", "turn-2"],
        p_mastery=0.85,
    )
    governor.ingest([stored_claim], context_for_claims(stored_claim))
    first_new = claim(
        "new-chain-1",
        "Student made repeated errors adding fractions.",
        evidence_ids=["turn-3", "turn-4"],
        risk_flags=["obs:incorrect"],
        p_mastery=0.2,
    )
    first_decisions = governor.revise_with("learner-1", first_new, context_for_claims(first_new))
    second_new = claim(
        "new-chain-2",
        "Student still made repeated errors adding fractions.",
        evidence_ids=["turn-5", "turn-6"],
        risk_flags=["obs:incorrect"],
        p_mastery=0.1,
    )

    second_decisions = governor.revise_with("learner-1", second_new, context_for_claims(second_new))

    assert first_decisions[0].action is RevisionAction.SUPERSEDE
    assert len(second_decisions) == 1
    assert governor.store.get("old-chain").admission_status is AdmissionStatus.SUPERSEDED
    assert governor.store.get("new-chain-2") is not None


def test_revision_lookup_allows_compatible_mastery_construct_types():
    audit = AuditTrail()
    governor = Governor(Policy.for_condition("C2"), audit=audit)
    stored_claim = claim(
        "partial-1",
        "Student has partially mastered adding fractions.",
        construct_type=ConstructType.PARTIAL_MASTERY,
        construct_key="math.fractions.addition",
    )
    governor.ingest([stored_claim], context_for_claims(stored_claim))
    new_ev = claim(
        "knowledge-1",
        "Student made an error adding fractions.",
        construct_type=ConstructType.KNOWLEDGE_STATE,
        construct_key="math.fractions.addition",
        evidence_ids=["turn-9"],
        risk_flags=["obs:incorrect"],
    )

    decisions = governor.revise_with("learner-1", new_ev, context_for_claims(new_ev))

    assert decisions
    assert decisions[0].action is not RevisionAction.NO_OP


def test_governor_confirm_revision_updates_store_belief():
    audit = AuditTrail()
    governor = Governor(Policy.for_condition("C2"), audit=audit)
    stored_claim = claim(
        "pref-1",
        "Student prefers visual examples.",
        construct_type=ConstructType.PREFERENCE,
        construct_key="preference.visual_examples",
        p_mastery=None,
    )
    governor.ingest([stored_claim], context_for_claims(stored_claim))
    new_ev = claim(
        "pref-2",
        "Student again requested a diagram.",
        construct_type=ConstructType.PREFERENCE,
        construct_key="preference.visual_examples",
        evidence_ids=["turn-7"],
        p_mastery=None,
    )

    decisions = governor.revise_with("learner-1", new_ev, context_for_claims(new_ev))
    stored = governor.store.get("pref-1")

    assert decisions[0].action is RevisionAction.CONFIRM
    assert stored.belief.n_evidence == 3
    assert audit.events[-1].belief_after.n_evidence == stored.belief.n_evidence


def test_learner_model_rejects_duplicate_claim_id():
    model = LearnerModel()
    stored_claim = claim("dup-1", "Student has partially mastered adding fractions.")

    model.add(stored_claim)

    with pytest.raises(DuplicateClaimError):
        model.add(stored_claim)


def test_candidate_input_hash_ignores_runtime_state_fields():
    original = claim("hash-1", "Student has partially mastered adding fractions.")
    changed_runtime_state = original.model_copy(
        update={
            "admission_status": AdmissionStatus.WEAK_CONTEXT,
            "risk_flags": ["challenged"],
            "updated_at": utc_now(),
        }
    )

    assert _hash_claim(original) == _hash_claim(changed_runtime_state)
