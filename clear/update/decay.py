"""Temporal decay for quantifiable knowledge claims."""

from __future__ import annotations

from datetime import datetime

from clear.models.claim import LearnerStateClaim
from clear.models.decision import RevisionDecision
from clear.models.policy import Policy
from clear.types import ConstructType, RevisionAction, RevisionRegime
from clear.update.decision_layer import belief_lower_bound

QUANTIFIABLE = {
    ConstructType.KNOWLEDGE_STATE,
    ConstructType.PARTIAL_MASTERY,
    ConstructType.MISCONCEPTION,
}


def apply_decay(claim: LearnerStateClaim, now: datetime, policy: Policy) -> RevisionDecision | None:
    if claim.construct_type not in QUANTIFIABLE:
        return None
    age_seconds = max((now - claim.updated_at).total_seconds(), 0.0)
    age_days = age_seconds / 86400.0
    if age_days <= 0:
        return None
    prior = claim.belief
    variance = prior.variance if prior.variance is not None else policy.init_variance
    inflated = min(1.0, variance + policy.decay_rate * age_days)
    posterior = prior.model_copy(update={"variance": inflated})
    if belief_lower_bound(posterior, policy) < policy.reconfirm_threshold:
        return RevisionDecision(
            claim_id=claim.claim_id,
            regime=RevisionRegime.QUANTIFIABLE_KNOWLEDGE,
            action=RevisionAction.DECAY_EXPIRE,
            prior_belief=prior,
            posterior_belief=posterior,
            reasons=["variance_inflation_needs_reconfirmation"],
        )
    return None
