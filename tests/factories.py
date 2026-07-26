"""Deterministic claim factory used by integration tests."""

from clear.models.claim import BeliefState, LearnerStateClaim
from clear.types import (
    AdmissionStatus,
    ConstructType,
    InferenceLevel,
    ScopeGranularity,
    SourceChannel,
)


def claim(
    claim_id: str,
    text: str,
    construct_type: ConstructType = ConstructType.KNOWLEDGE_STATE,
    construct_key: str = "math.fractions.addition",
    evidence_ids: list[str] | None = None,
    confidence: float = 0.8,
    status: AdmissionStatus = AdmissionStatus.CANDIDATE,
    p_mastery: float | None = 0.8,
    risk_flags: list[str] | None = None,
) -> LearnerStateClaim:
    return LearnerStateClaim(
        claim_id=claim_id,
        learner_id="learner-1",
        claim_text=text,
        construct_type=construct_type,
        construct_key=construct_key,
        source_channel=SourceChannel.TUTOR_INFERENCE,
        source_evidence_ids=evidence_ids if evidence_ids is not None else ["turn-1"],
        inference_level=InferenceLevel.LOCAL_INFERENCE,
        scope_granularity=ScopeGranularity.SKILL,
        scope_domain="math",
        confidence=confidence,
        admission_status=status,
        belief=BeliefState(p_mastery=p_mastery, variance=0.04, n_evidence=2),
        risk_flags=risk_flags or [],
    )
