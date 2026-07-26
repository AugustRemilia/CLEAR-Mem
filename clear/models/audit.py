"""Stable audit event schema for metrics reconstruction."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from clear.models.claim import BeliefState
from clear.types import (
    AdmissionStatus,
    AuditEventType,
    ConditionId,
    ConstructType,
    ImpactTier,
    SourceChannel,
    UseType,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_event_id() -> str:
    return uuid4().hex


class GovernanceAuditEvent(BaseModel):
    """Append-only audit event with stable fields used by CLEAR metrics."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    event_id: str = Field(default_factory=new_event_id, min_length=1)
    metric_key: str | None = None
    event_type: AuditEventType
    timestamp: datetime = Field(default_factory=utc_now)
    learner_id: str = Field(min_length=1)
    claim_id: str | None = None
    construct_type: ConstructType | None = None
    construct_key: str | None = None
    source_channel: SourceChannel | None = None
    decision: str | None = None
    reasons: list[str] = Field(default_factory=list)
    failed_criteria: list[str] = Field(default_factory=list)
    use_type: UseType | None = None
    impact_tier: ImpactTier | None = None
    admission_status_before: AdmissionStatus | None = None
    admission_status_after: AdmissionStatus | None = None
    belief_before: BeliefState | None = None
    belief_after: BeliefState | None = None
    policy_version: str = Field(min_length=1)
    condition_id: ConditionId | None = None
    adapter_id: str | None = None
    input_hash: str | None = None
    candidate_input_hash: str | None = None
    source_evidence_ids: list[str] = Field(default_factory=list)
    review_required: bool = False
    risk_flags_before: list[str] = Field(default_factory=list)
    risk_flags_after: list[str] = Field(default_factory=list)
    permitted_uses_after: list[UseType] = Field(default_factory=list)
    prohibited_uses_after: list[UseType] = Field(default_factory=list)
    replacement_claim_id: str | None = None
    replacement_admission_status_after: AdmissionStatus | None = None
    replacement_belief_after: BeliefState | None = None
    replacement_risk_flags_after: list[str] = Field(default_factory=list)
    replacement_permitted_uses_after: list[UseType] = Field(default_factory=list)
    replacement_prohibited_uses_after: list[UseType] = Field(default_factory=list)
    run_id: str | None = None
    probe_id: str | None = None
    case_id: str | None = None
    sequence_id: str | None = None
    random_seed: int | None = None

    def to_jsonl(self) -> str:
        return self.model_dump_json() + "\n"
