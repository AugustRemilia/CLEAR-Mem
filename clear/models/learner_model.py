"""In-memory learner model store with stable query surface."""

from __future__ import annotations

from clear.models.claim import LearnerStateClaim
from clear.types import AdmissionStatus, ConstructType


class DuplicateClaimError(ValueError):
    pass


class LearnerModel:
    def __init__(self) -> None:
        self._claims: dict[str, LearnerStateClaim] = {}

    def add(self, claim: LearnerStateClaim) -> None:
        if claim.claim_id in self._claims:
            raise DuplicateClaimError(claim.claim_id)
        self._claims[claim.claim_id] = claim.model_copy(deep=True)

    def get(self, claim_id: str) -> LearnerStateClaim | None:
        claim = self._claims.get(claim_id)
        return claim.model_copy(deep=True) if claim else None

    def find(
        self,
        learner_id: str,
        construct_type: ConstructType | None = None,
        construct_key: str | None = None,
        status: AdmissionStatus | None = None,
    ) -> list[LearnerStateClaim]:
        out: list[LearnerStateClaim] = []
        for claim in self._claims.values():
            if claim.learner_id != learner_id:
                continue
            if construct_type is not None and claim.construct_type != construct_type:
                continue
            if construct_key is not None and claim.construct_key != construct_key:
                continue
            if status is not None and claim.admission_status != status:
                continue
            out.append(claim.model_copy(deep=True))
        return out

    def update(self, claim: LearnerStateClaim) -> None:
        if claim.claim_id not in self._claims:
            raise KeyError(claim.claim_id)
        self._claims[claim.claim_id] = claim.model_copy(deep=True)

    def all_claims(self) -> list[LearnerStateClaim]:
        return [claim.model_copy(deep=True) for claim in self._claims.values()]

    def all_active(self, learner_id: str) -> list[LearnerStateClaim]:
        return [
            claim.model_copy(deep=True)
            for claim in self._claims.values()
            if claim.learner_id == learner_id
            and claim.admission_status in {AdmissionStatus.ACTIVE, AdmissionStatus.WEAK_CONTEXT}
        ]
