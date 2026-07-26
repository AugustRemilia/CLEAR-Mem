"""Runtime context objects passed into pure CLEAR gates."""

from __future__ import annotations

from dataclasses import dataclass, field

from clear.types import ConstructType, Observation


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    construct_type: ConstructType
    construct_key: str
    observation: Observation = Observation.NEUTRAL
    text: str = ""
    scope_domain: str | None = None


@dataclass(frozen=True)
class GovernanceContext:
    current_evidence: list[Evidence] = field(default_factory=list)
    evidence_registry_ids: set[str] = field(default_factory=set)
    now_iso: str | None = None
