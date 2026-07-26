"""Decision schemas returned by pure gate and update functions."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from clear.models.claim import BeliefState
from clear.types import (
    AdmissionStatus,
    ImpactTier,
    RevisionAction,
    RevisionRegime,
    SourceChannel,
    UseType,
)


class AdmissionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    claim_id: str
    outcome: AdmissionStatus
    failed_criteria: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    channel: SourceChannel
    permitted_uses_initial: list[UseType] = Field(default_factory=list)
    review_required: bool = False
    risk_flags: list[str] = Field(default_factory=list)


class UseDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    claim_id: str
    use_type: UseType
    impact_tier: ImpactTier
    authorized: bool
    reason: str
    required_lower_bound: float | None = None
    belief_lower_bound: float | None = None


class RevisionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    claim_id: str
    regime: RevisionRegime
    action: RevisionAction
    prior_belief: BeliefState | None = None
    posterior_belief: BeliefState | None = None
    reasons: list[str] = Field(default_factory=list)
