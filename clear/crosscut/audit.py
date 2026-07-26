"""Append-only JSONL audit trail."""

from __future__ import annotations

from pathlib import Path

from clear.models.audit import GovernanceAuditEvent


class AuditTrail:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.events: list[GovernanceAuditEvent] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: GovernanceAuditEvent) -> None:
        self.events.append(event)
        if self.path:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(event.to_jsonl())

    @classmethod
    def read_jsonl(cls, path: str | Path) -> list[GovernanceAuditEvent]:
        events: list[GovernanceAuditEvent] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    events.append(GovernanceAuditEvent.model_validate_json(line))
        return events
