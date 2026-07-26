"""Deterministic mock adapter used before live LLM integration."""

from __future__ import annotations

from clear.models.claim import LearnerStateClaim


class MockAdapter:
    adapter_id = "mock"

    def propose_claims(self, claims: list[LearnerStateClaim]) -> list[LearnerStateClaim]:
        return [claim.model_copy(deep=True) for claim in claims]
