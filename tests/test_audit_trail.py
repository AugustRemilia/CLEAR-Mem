from clear.crosscut.audit import AuditTrail
from clear.models.audit import GovernanceAuditEvent
from clear.types import AuditEventType


def test_audit_trail_appends_and_reads_jsonl(tmp_path):
    path = tmp_path / "audit.jsonl"
    trail = AuditTrail(path)
    event = GovernanceAuditEvent(
        event_type=AuditEventType.ADMISSION,
        learner_id="learner-1",
        policy_version="test-policy",
        decision="active",
    )

    trail.log(event)
    loaded = AuditTrail.read_jsonl(path)

    assert len(loaded) == 1
    assert loaded[0].event_id == event.event_id
    assert loaded[0].decision == "active"
