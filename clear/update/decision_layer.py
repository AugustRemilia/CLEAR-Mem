"""Decision-layer helpers shared by use authorization and revision."""

from __future__ import annotations

from clear.models.claim import BeliefState
from clear.models.policy import Policy


def belief_lower_bound(belief: BeliefState, policy: Policy) -> float:
    if belief.p_mastery is None:
        return 0.0
    variance = belief.variance if belief.variance is not None else policy.init_variance
    sd = variance**0.5
    return max(0.0, belief.p_mastery - policy.z_lower_bound * sd)
