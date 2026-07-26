"""Minimal admission and instructional-use example."""

from clear.context import Evidence, GovernanceContext
from clear.governor import Governor
from clear.models.claim import BeliefState, LearnerStateClaim
from clear.models.policy import Policy
from clear.types import (
    ConstructType,
    InferenceLevel,
    Observation,
    ScopeGranularity,
    SourceChannel,
    UseType,
)


evidence = Evidence(
    evidence_id="turn-7",
    construct_type=ConstructType.MISCONCEPTION,
    construct_key="math.equations.inverse_operation",
    observation=Observation.INCORRECT,
    text="The learner changed 3x - 5 = 10 to 3x = 10 - 5.",
    scope_domain="mathematics",
)

claim = LearnerStateClaim(
    claim_id="claim-1",
    learner_id="learner-1",
    session_id="session-1",
    claim_text="The learner currently confuses the direction of inverse operations.",
    construct_type=ConstructType.MISCONCEPTION,
    construct_key="math.equations.inverse_operation",
    source_channel=SourceChannel.TUTOR_INFERENCE,
    source_evidence_ids=[evidence.evidence_id],
    evidence_span=evidence.text,
    inference_level=InferenceLevel.LOCAL_INFERENCE,
    scope_granularity=ScopeGranularity.SKILL,
    scope_domain="mathematics",
    confidence=0.82,
    belief=BeliefState(p_mastery=0.25, variance=0.04, n_evidence=1),
)

context = GovernanceContext(
    current_evidence=[evidence],
    evidence_registry_ids={evidence.evidence_id},
)
governor = Governor(Policy())

admission = governor.ingest([claim], context)[0]
use = governor.request_use(
    learner_id=claim.learner_id,
    use_type=UseType.CORRECTIVE_FEEDBACK,
    current_evidence=[evidence],
)[0]

print(f"admission={admission.outcome.value}")
print(f"corrective_feedback_authorized={use.authorized}")
print(f"audit_events={len(governor.audit.events)}")
