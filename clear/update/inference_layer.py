"""Simplified BKT-style inference layer for quantifiable knowledge claims."""

from __future__ import annotations

from clear.models.claim import BeliefState, LearnerStateClaim
from clear.models.policy import Policy
from clear.types import ConstructType, Observation


def observation_from(new_ev: LearnerStateClaim) -> Observation:
    flags = set(new_ev.risk_flags)
    text = new_ev.claim_text.lower()
    if "obs:incorrect" in flags or "incorrect" in text or "error" in text or "mistake" in text:
        return Observation.INCORRECT
    if "obs:correct" in flags or "correct" in text or "mastered" in text or "success" in text:
        return Observation.CORRECT
    return Observation.NEUTRAL


def bayes_update(
    prior: BeliefState,
    obs: Observation,
    policy: Policy,
    construct: ConstructType,
) -> BeliefState:
    p = prior.p_mastery if prior.p_mastery is not None else policy.init_p_mastery
    if obs is Observation.NEUTRAL:
        return BeliefState(
            p_mastery=p,
            variance=prior.variance if prior.variance is not None else policy.init_variance,
            n_evidence=prior.n_evidence,
        )
    if obs is Observation.CORRECT:
        num = (1 - policy.slip) * p
        den = num + policy.guess * (1 - p)
    else:
        num = policy.slip * p
        den = num + (1 - policy.guess) * (1 - p)
    p_post = num / den if den > 0 else p
    if construct is ConstructType.MISCONCEPTION and obs is Observation.CORRECT:
        p_post = p + policy.misconception_damping * (p_post - p)
    prior_variance = prior.variance if prior.variance is not None else policy.init_variance
    n_next = prior.n_evidence + 1
    variance = prior_variance * max(prior.n_evidence, 1) / max(n_next, 1)
    return BeliefState(p_mastery=p_post, variance=variance, n_evidence=n_next)


def strong_enough(
    prior: BeliefState,
    post: BeliefState,
    policy: Policy,
    new_evidence_count: int,
) -> bool:
    prior_p = prior.p_mastery if prior.p_mastery is not None else policy.init_p_mastery
    post_p = post.p_mastery if post.p_mastery is not None else prior_p
    moved = abs(post_p - prior_p) >= policy.supersede_delta
    enough = new_evidence_count >= policy.supersede_min_evidence
    return moved and enough
