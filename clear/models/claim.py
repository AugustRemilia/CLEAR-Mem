"""Learner-state claim schema."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator

from clear.types import (
    AdmissionStatus,
    ConstructType,
    InferenceLevel,
    ReviewStatus,
    ScopeGranularity,
    SourceChannel,
    TemporalStatus,
    UseType,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class BeliefState(BaseModel):
    """Quantifiable belief state used only for knowledge-like claims."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    p_mastery: float | None = Field(default=None, ge=0.0, le=1.0)
    variance: float | None = Field(default=None, ge=0.0)
    n_evidence: int = Field(default=0, ge=0)


class LearnerStateClaim(BaseModel):
    """Smallest governed unit in CLEAR: one evidence-checkable learner claim."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    claim_id: str = Field(min_length=1)
    learner_id: str = Field(min_length=1)
    session_id: str | None = None
    claim_text: str = Field(min_length=1)
    construct_type: ConstructType
    construct_key: str = Field(min_length=1)
    source_channel: SourceChannel
    source_evidence_ids: list[str] = Field(default_factory=list)
    evidence_span: str | None = None
    inference_level: InferenceLevel
    scope_granularity: ScopeGranularity
    scope_domain: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    belief: BeliefState = Field(default_factory=BeliefState)
    temporal_status: TemporalStatus = TemporalStatus.CURRENT
    valid_until: datetime | None = None
    sensitive_flags: list[str] = Field(default_factory=list)
    allowed_uses: list[UseType] = Field(default_factory=list)
    prohibited_uses: list[UseType] = Field(default_factory=list)
    admission_status: AdmissionStatus = AdmissionStatus.CANDIDATE
    review_status: ReviewStatus = ReviewStatus.NONE
    parent_claim_ids: list[str] = Field(default_factory=list)
    derived_from_summary_id: str | None = None
    revision_of: str | None = None
    risk_flags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("claim_id", "learner_id", "claim_text", "construct_key")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("field must not be blank")
        return stripped

    @field_validator("source_evidence_ids", "sensitive_flags", "risk_flags")
    @classmethod
    def strip_string_lists(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]
