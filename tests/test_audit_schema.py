from clear.models.audit import GovernanceAuditEvent
from clear.models.claim import BeliefState
from clear.types import (
    AdmissionStatus,
    AuditEventType,
    ConditionId,
    ConstructType,
    SourceChannel,
)


def test_audit_event_has_stable_metric_fields_and_jsonl():
    event = GovernanceAuditEvent(
        event_type=AuditEventType.REVISION,
        learner_id="learner-1",
        claim_id="claim-1",
        construct_type=ConstructType.MISCONCEPTION,
        construct_key="linear_equations.negative_sign_transfer",
        source_channel=SourceChannel.TUTOR_INFERENCE,
        decision="challenge",
        reasons=["current_evidence_conflict"],
        admission_status_before=AdmissionStatus.ACTIVE,
        admission_status_after=AdmissionStatus.WEAK_CONTEXT,
        belief_before=BeliefState(p_mastery=0.72, variance=0.04, n_evidence=3),
        belief_after=BeliefState(p_mastery=0.48, variance=0.08, n_evidence=4),
        policy_version="clear-schema-batch1-v1",
        condition_id=ConditionId.C2,
        adapter_id="mock",
        input_hash="abc123",
        source_evidence_ids=["turn-3", "turn-4"],
    )

    dumped = event.model_dump(mode="json")
    assert dumped["construct_key"] == "linear_equations.negative_sign_transfer"
    assert dumped["policy_version"] == "clear-schema-batch1-v1"
    assert dumped["condition_id"] == "C2"
    assert dumped["belief_before"]["n_evidence"] == 3
    assert event.to_jsonl().endswith("\n")


def test_audit_event_forbids_unstable_payload_field():
    try:
        GovernanceAuditEvent(
            event_type=AuditEventType.ADMISSION,
            learner_id="learner-1",
            policy_version="clear-schema-batch1-v1",
            payload={"anything": "goes"},
        )
    except Exception as exc:
        assert "payload" in str(exc)
    else:
        raise AssertionError("payload field should be rejected")
