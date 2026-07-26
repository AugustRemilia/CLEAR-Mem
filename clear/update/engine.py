"""Pure revision engine."""

from __future__ import annotations

from clear.context import GovernanceContext
from clear.gates.admission import run_admission
from clear.models.claim import LearnerStateClaim
from clear.models.decision import AdmissionDecision, RevisionDecision
from clear.models.policy import Policy
from clear.types import AdmissionStatus, RevisionAction, RevisionRegime
from clear.update.inference_layer import bayes_update, observation_from, strong_enough
from clear.update.router import regime_for, revision_match


def revise(
    stored_claim: LearnerStateClaim,
    new_evidence_claim: LearnerStateClaim,
    ctx: GovernanceContext | None,
    policy: Policy,
) -> RevisionDecision:
    regime = regime_for(stored_claim, policy)
    if not policy.structured_adjudication:
        return RevisionDecision(
            claim_id=stored_claim.claim_id,
            regime=regime,
            action=RevisionAction.SUPERSEDE,
            prior_belief=stored_claim.belief,
            posterior_belief=new_evidence_claim.belief,
            reasons=["no_governance_supersede"],
        )
    if not revision_match(stored_claim, new_evidence_claim, policy):
        return RevisionDecision(
            claim_id=stored_claim.claim_id,
            regime=regime,
            action=RevisionAction.NO_OP,
            prior_belief=stored_claim.belief,
            reasons=["construct_key_or_scope_mismatch"],
        )
    admission = run_admission(new_evidence_claim, ctx or GovernanceContext(), policy)
    if _failed_revision_admission(admission):
        return RevisionDecision(
            claim_id=stored_claim.claim_id,
            regime=regime,
            action=RevisionAction.REJECT_OVERRIDE,
            prior_belief=stored_claim.belief,
            posterior_belief=stored_claim.belief,
            reasons=["new_evidence_failed_admission", *admission.reasons],
        )
    if admission.outcome is AdmissionStatus.WEAK_CONTEXT:
        return RevisionDecision(
            claim_id=stored_claim.claim_id,
            regime=regime,
            action=RevisionAction.CHALLENGE,
            prior_belief=stored_claim.belief,
            posterior_belief=stored_claim.belief,
            reasons=["new_evidence_weak_context_not_superseding", *admission.reasons],
        )
    if not policy.use_bayesian:
        return RevisionDecision(
            claim_id=stored_claim.claim_id,
            regime=regime,
            action=RevisionAction.SUPERSEDE,
            prior_belief=stored_claim.belief,
            posterior_belief=new_evidence_claim.belief,
            reasons=["naive_last_write_wins"],
        )
    if regime is RevisionRegime.QUANTIFIABLE_KNOWLEDGE:
        return revise_quantifiable(stored_claim, new_evidence_claim, policy)
    if regime is RevisionRegime.BEHAVIORAL_PREFERENCE:
        return revise_behavioral(stored_claim, new_evidence_claim, policy)
    return revise_sensitive(stored_claim)


def revise_quantifiable(
    stored: LearnerStateClaim,
    new_ev: LearnerStateClaim,
    policy: Policy,
) -> RevisionDecision:
    prior = stored.belief
    obs = observation_from(new_ev)
    post = bayes_update(prior, obs, policy, stored.construct_type)
    if strong_enough(prior, post, policy, new_evidence_count=len(new_ev.source_evidence_ids)):
        return RevisionDecision(
            claim_id=stored.claim_id,
            regime=RevisionRegime.QUANTIFIABLE_KNOWLEDGE,
            action=RevisionAction.SUPERSEDE,
            prior_belief=prior,
            posterior_belief=post,
            reasons=["strong_evidence_supersede"],
        )
    return RevisionDecision(
        claim_id=stored.claim_id,
        regime=RevisionRegime.QUANTIFIABLE_KNOWLEDGE,
        action=RevisionAction.CHALLENGE,
        prior_belief=prior,
        posterior_belief=post,
        reasons=["insufficient_to_supersede"],
    )


def revise_behavioral(
    stored: LearnerStateClaim,
    new_ev: LearnerStateClaim,
    _policy: Policy,
) -> RevisionDecision:
    if _scope_creep_to_trait(new_ev):
        return RevisionDecision(
            claim_id=stored.claim_id,
            regime=RevisionRegime.BEHAVIORAL_PREFERENCE,
            action=RevisionAction.REJECT_OVERRIDE,
            prior_belief=stored.belief,
            posterior_belief=stored.belief,
            reasons=["scope_creep_to_trait_label"],
        )
    updated = stored.belief.model_copy(update={"n_evidence": stored.belief.n_evidence + 1})
    return RevisionDecision(
        claim_id=stored.claim_id,
        regime=RevisionRegime.BEHAVIORAL_PREFERENCE,
        action=RevisionAction.CONFIRM,
        prior_belief=stored.belief,
        posterior_belief=updated,
        reasons=["behavioral_stability_counted"],
    )


def revise_sensitive(stored: LearnerStateClaim) -> RevisionDecision:
    return RevisionDecision(
        claim_id=stored.claim_id,
        regime=RevisionRegime.SENSITIVE_NEED,
        action=RevisionAction.NO_OP,
        prior_belief=stored.belief,
        posterior_belief=stored.belief,
        reasons=["sensitive_no_belief_update"],
    )


def _scope_creep_to_trait(new_ev: LearnerStateClaim) -> bool:
    text = new_ev.claim_text.lower()
    return "learning style" in text or "type of learner" in text or "fixed trait" in text


def _failed_revision_admission(admission: AdmissionDecision) -> bool:
    return admission.outcome in {
        AdmissionStatus.REJECTED,
        AdmissionStatus.QUARANTINED,
        AdmissionStatus.AUDIT_ONLY,
        AdmissionStatus.EXPIRED,
    }
