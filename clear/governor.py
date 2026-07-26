"""CLEAR governor: the only layer that writes store and audit side effects."""

from __future__ import annotations

import hashlib
import json

from clear.context import Evidence, GovernanceContext
from clear.crosscut.audit import AuditTrail
from clear.gates.admission import run_admission
from clear.gates.use_authorization import authorize_use
from clear.models.audit import GovernanceAuditEvent
from clear.models.claim import LearnerStateClaim, utc_now
from clear.models.decision import AdmissionDecision, RevisionDecision, UseDecision
from clear.models.learner_model import LearnerModel
from clear.models.policy import Policy
from clear.types import (
    AdmissionStatus,
    AuditEventType,
    ReviewStatus,
    RevisionAction,
    UseType,
)
from clear.update.engine import revise


REVISION_ELIGIBLE_STATUSES = {
    AdmissionStatus.ACTIVE,
    AdmissionStatus.WEAK_CONTEXT,
}


class Governor:
    def __init__(
        self,
        policy: Policy,
        store: LearnerModel | None = None,
        audit: AuditTrail | None = None,
        adapter_id: str = "mock",
    ) -> None:
        self.policy = policy
        self.store = store or LearnerModel()
        self.audit = audit or AuditTrail()
        self.adapter_id = adapter_id

    def ingest(
        self,
        claims: list[LearnerStateClaim],
        ctx: GovernanceContext | None = None,
    ) -> list[AdmissionDecision]:
        decisions: list[AdmissionDecision] = []
        for claim in claims:
            decision = run_admission(claim, ctx or GovernanceContext(), self.policy)
            self.audit.log(self._admission_event(claim, decision))
            if decision.outcome in {AdmissionStatus.ACTIVE, AdmissionStatus.WEAK_CONTEXT}:
                risk_flags = list(dict.fromkeys(claim.risk_flags + decision.risk_flags))
                if decision.review_required:
                    risk_flags = list(dict.fromkeys(risk_flags + ["review_required"]))
                stored = claim.model_copy(
                    update={
                        "admission_status": decision.outcome,
                        "allowed_uses": decision.permitted_uses_initial,
                        "review_status": _review_status_after_admission(decision),
                        "risk_flags": risk_flags,
                    },
                    deep=True,
                )
                self.store.add(stored)
            decisions.append(decision)
        return decisions

    def request_use(
        self,
        learner_id: str,
        use_type: UseType,
        current_evidence: list[Evidence] | None = None,
    ) -> list[UseDecision]:
        decisions: list[UseDecision] = []
        for claim in self.store.all_active(learner_id):
            decision = authorize_use(claim, use_type, current_evidence or [], self.policy)
            self.audit.log(self._use_event(claim, decision))
            decisions.append(decision)
        return decisions

    def revise_with(
        self,
        learner_id: str,
        new_evidence_claim: LearnerStateClaim,
        ctx: GovernanceContext | None = None,
    ) -> list[RevisionDecision]:
        decisions: list[RevisionDecision] = []
        matches = self.store.find(
            learner_id,
            construct_key=new_evidence_claim.construct_key,
        )
        eligible_claims = [
            claim for claim in matches if claim.admission_status in REVISION_ELIGIBLE_STATUSES
        ]
        context = ctx or GovernanceContext()
        revision_pairs = [
            (claim, revise(claim, new_evidence_claim, context, self.policy))
            for claim in eligible_claims
        ]
        supersede_pairs = [
            (claim, decision)
            for claim, decision in revision_pairs
            if decision.action is RevisionAction.SUPERSEDE and decision.posterior_belief
        ]
        if supersede_pairs:
            transaction_results = self._apply_supersede_transaction(
                supersede_pairs,
                new_evidence_claim,
                context,
            )
            after_by_claim_id = {
                before.claim_id: (after, replacement)
                for before, after, replacement in transaction_results
            }
        else:
            after_by_claim_id = {}

        for claim, decision in revision_pairs:
            if claim.claim_id in after_by_claim_id:
                after_claim, replacement_claim = after_by_claim_id[claim.claim_id]
            else:
                after_claim, replacement_claim = self._apply_revision(
                    claim,
                    decision,
                    new_evidence_claim,
                    context,
                )
            self.audit.log(
                self._revision_event(
                    claim,
                    after_claim,
                    replacement_claim,
                    decision,
                    new_evidence_claim,
                )
            )
            decisions.append(decision)
        return decisions

    def _apply_supersede_transaction(
        self,
        supersede_pairs: list[tuple[LearnerStateClaim, RevisionDecision]],
        new_evidence_claim: LearnerStateClaim,
        ctx: GovernanceContext,
    ) -> list[tuple[LearnerStateClaim, LearnerStateClaim, LearnerStateClaim | None]]:
        first_claim, first_decision = supersede_pairs[0]
        now = utc_now()
        admission = run_admission(new_evidence_claim, ctx, self.policy)
        all_parent_ids = list(
            dict.fromkeys(
                new_evidence_claim.parent_claim_ids
                + [claim.claim_id for claim, _decision in supersede_pairs]
            )
        )
        replacement = new_evidence_claim.model_copy(
            update={
                "belief": first_decision.posterior_belief,
                "admission_status": admission.outcome,
                "allowed_uses": admission.permitted_uses_initial,
                "revision_of": first_claim.claim_id,
                "parent_claim_ids": all_parent_ids,
                "review_status": _review_status_after_admission(admission),
                "risk_flags": _stored_risk_flags(new_evidence_claim, admission),
                "updated_at": now,
            },
            deep=True,
        )
        results: list[tuple[LearnerStateClaim, LearnerStateClaim, LearnerStateClaim | None]] = []
        for claim, _decision in supersede_pairs:
            old_claim = claim.model_copy(
                update={
                    "admission_status": AdmissionStatus.SUPERSEDED,
                    "updated_at": now,
                },
                deep=True,
            )
            self.store.update(old_claim)
            results.append((claim, old_claim, replacement))
        self.store.add(replacement)
        return results

    def _apply_revision(
        self,
        claim: LearnerStateClaim,
        decision: RevisionDecision,
        new_evidence_claim: LearnerStateClaim,
        ctx: GovernanceContext,
    ) -> tuple[LearnerStateClaim, LearnerStateClaim | None]:
        if decision.action is RevisionAction.SUPERSEDE and decision.posterior_belief:
            now = utc_now()
            old_claim = claim.model_copy(
                update={
                    "admission_status": AdmissionStatus.SUPERSEDED,
                    "updated_at": now,
                },
                deep=True,
            )
            admission = run_admission(new_evidence_claim, ctx, self.policy)
            replacement = new_evidence_claim.model_copy(
                update={
                    "belief": decision.posterior_belief,
                    "admission_status": admission.outcome,
                    "allowed_uses": admission.permitted_uses_initial,
                    "revision_of": claim.claim_id,
                    "parent_claim_ids": list(
                        dict.fromkeys(new_evidence_claim.parent_claim_ids + [claim.claim_id])
                    ),
                    "review_status": _review_status_after_admission(admission),
                    "risk_flags": _stored_risk_flags(new_evidence_claim, admission),
                    "updated_at": now,
                },
                deep=True,
            )
            self.store.update(old_claim)
            self.store.add(replacement)
            return old_claim, replacement
        elif decision.action is RevisionAction.CHALLENGE and decision.posterior_belief:
            flags = list(dict.fromkeys(claim.risk_flags + ["challenged"]))
            updated = claim.model_copy(
                update={
                    "belief": decision.posterior_belief,
                    "admission_status": AdmissionStatus.WEAK_CONTEXT,
                    "risk_flags": flags,
                },
                deep=True,
            )
            self.store.update(updated)
            return updated, None
        elif decision.action is RevisionAction.CONFIRM and decision.posterior_belief:
            updated = claim.model_copy(update={"belief": decision.posterior_belief}, deep=True)
            self.store.update(updated)
            return updated, None
        elif decision.action is RevisionAction.DECAY_EXPIRE and decision.posterior_belief:
            updated = claim.model_copy(
                update={
                    "belief": decision.posterior_belief,
                    "admission_status": AdmissionStatus.EXPIRED,
                },
                deep=True,
            )
            self.store.update(updated)
            return updated, None
        return claim, None

    def _base_event_kwargs(self, claim: LearnerStateClaim) -> dict:
        return {
            "learner_id": claim.learner_id,
            "claim_id": claim.claim_id,
            "construct_type": claim.construct_type,
            "construct_key": claim.construct_key,
            "source_channel": claim.source_channel,
            "policy_version": self.policy.policy_version,
            "condition_id": self.policy.condition_id,
            "adapter_id": self.adapter_id,
            "input_hash": _hash_claim(claim),
            "source_evidence_ids": claim.source_evidence_ids,
        }

    def _admission_event(
        self,
        claim: LearnerStateClaim,
        decision: AdmissionDecision,
    ) -> GovernanceAuditEvent:
        return GovernanceAuditEvent(
            event_type=AuditEventType.ADMISSION,
            decision=decision.outcome.value,
            reasons=decision.reasons,
            failed_criteria=decision.failed_criteria,
            admission_status_before=AdmissionStatus.CANDIDATE,
            admission_status_after=decision.outcome,
            review_required=decision.review_required,
            risk_flags_before=claim.risk_flags,
            risk_flags_after=list(dict.fromkeys(claim.risk_flags + decision.risk_flags)),
            permitted_uses_after=decision.permitted_uses_initial,
            prohibited_uses_after=claim.prohibited_uses,
            **self._base_event_kwargs(claim),
        )

    def _use_event(self, claim: LearnerStateClaim, decision: UseDecision) -> GovernanceAuditEvent:
        return GovernanceAuditEvent(
            event_type=AuditEventType.USE_AUTHORIZATION,
            decision="authorized" if decision.authorized else "denied",
            reasons=[decision.reason],
            use_type=decision.use_type,
            impact_tier=decision.impact_tier,
            admission_status_before=claim.admission_status,
            admission_status_after=claim.admission_status,
            belief_before=claim.belief,
            risk_flags_before=claim.risk_flags,
            risk_flags_after=claim.risk_flags,
            permitted_uses_after=claim.allowed_uses,
            prohibited_uses_after=claim.prohibited_uses,
            **self._base_event_kwargs(claim),
        )

    def _revision_event(
        self,
        claim: LearnerStateClaim,
        after_claim: LearnerStateClaim,
        replacement_claim: LearnerStateClaim | None,
        decision: RevisionDecision,
        new_evidence_claim: LearnerStateClaim,
    ) -> GovernanceAuditEvent:
        return GovernanceAuditEvent(
            event_type=AuditEventType.REVISION,
            decision=decision.action.value,
            reasons=decision.reasons,
            admission_status_before=claim.admission_status,
            admission_status_after=after_claim.admission_status,
            belief_before=decision.prior_belief,
            belief_after=after_claim.belief,
            candidate_input_hash=_hash_claim(new_evidence_claim),
            risk_flags_before=claim.risk_flags,
            risk_flags_after=after_claim.risk_flags,
            permitted_uses_after=after_claim.allowed_uses,
            prohibited_uses_after=after_claim.prohibited_uses,
            replacement_claim_id=replacement_claim.claim_id if replacement_claim else None,
            replacement_admission_status_after=(
                replacement_claim.admission_status if replacement_claim else None
            ),
            replacement_belief_after=replacement_claim.belief if replacement_claim else None,
            replacement_risk_flags_after=replacement_claim.risk_flags if replacement_claim else [],
            replacement_permitted_uses_after=(
                replacement_claim.allowed_uses if replacement_claim else []
            ),
            replacement_prohibited_uses_after=(
                replacement_claim.prohibited_uses if replacement_claim else []
            ),
            source_evidence_ids=new_evidence_claim.source_evidence_ids,
            **{
                key: value
                for key, value in self._base_event_kwargs(claim).items()
                if key != "source_evidence_ids"
            },
        )


def _review_status_after_admission(decision: AdmissionDecision) -> ReviewStatus:
    return ReviewStatus.TEACHER_REVIEW_NEEDED if decision.review_required else ReviewStatus.NONE


def _stored_risk_flags(claim: LearnerStateClaim, decision: AdmissionDecision) -> list[str]:
    flags = list(dict.fromkeys(claim.risk_flags + decision.risk_flags))
    if decision.review_required:
        flags = list(dict.fromkeys(flags + ["review_required"]))
    return flags


def _hash_claim(claim: LearnerStateClaim) -> str:
    claim_data = claim.model_dump(
        mode="json",
        exclude={
            "created_at",
            "updated_at",
            "admission_status",
            "review_status",
            "risk_flags",
        },
    )
    raw = json.dumps(claim_data, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
