"""Typed witness closure and the no-compounding promotion gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .events import Event, SpeechAct


class GroundingError(RuntimeError):
    pass


def ledger_witness_id(witness: str) -> str | None:
    return witness.removeprefix("event:") if witness.startswith("event:") else None


def external_witnesses(event: Event) -> tuple[str, ...]:
    return tuple(
        witness
        for witness in event.provenance_witnesses
        if witness.startswith("external:")
    )


def is_promotion(event: Event) -> bool:
    marker = event.payload.get("promotion")
    return marker is True or isinstance(marker, dict)


@dataclass(frozen=True)
class GroundingReport:
    event_id: str
    closure_event_ids: tuple[str, ...]
    observed_root_ids: tuple[str, ...]
    external_witness_ids: tuple[str, ...]

    @property
    def grounded(self) -> bool:
        return bool(self.observed_root_ids)


class GroundingAuditor:
    def __init__(self, events: Iterable[Event]) -> None:
        items = list(events)
        self.events = {event.event_id: event for event in items}
        if len(self.events) != len(items):
            raise GroundingError("grounding audit cannot contain duplicate event IDs")

    def report(self, event_id: str) -> GroundingReport:
        if event_id not in self.events:
            raise GroundingError(f"unknown event {event_id!r}")
        seen: set[str] = set()
        pending = [event_id]
        external: set[str] = set()
        while pending:
            current_id = pending.pop()
            if current_id in seen:
                continue
            current = self.events.get(current_id)
            if current is None:
                raise GroundingError(
                    f"event {event_id!r} has missing provenance event {current_id!r}"
                )
            seen.add(current_id)
            external.update(external_witnesses(current))
            dependencies = set(current.parent_ids)
            dependencies.update(
                witness_id
                for witness in current.provenance_witnesses
                if (witness_id := ledger_witness_id(witness)) is not None
            )
            pending.extend(sorted(dependencies - seen))
        # A promotion cannot ground itself merely by labeling its own speech act
        # observed. It needs an earlier observation in its provenance closure.
        observed = {
            candidate_id
            for candidate_id in seen - {event_id}
            if self.events[candidate_id].speech_act == SpeechAct.OBSERVED
        }
        return GroundingReport(
            event_id=event_id,
            closure_event_ids=tuple(sorted(seen)),
            observed_root_ids=tuple(sorted(observed)),
            external_witness_ids=tuple(sorted(external)),
        )

    def promotion_reports(self) -> tuple[GroundingReport, ...]:
        return tuple(
            self.report(event.event_id)
            for event in self.events.values()
            if is_promotion(event)
        )

    def assert_promotions_grounded(self, event_ids: Iterable[str]) -> None:
        for event_id in event_ids:
            event = self.events[event_id]
            if not is_promotion(event):
                continue
            report = self.report(event_id)
            if not report.grounded:
                raise GroundingError(
                    f"promotion {event_id!r} has no observed provenance root"
                )


def validate_typed_witnesses(events: Iterable[Event]) -> None:
    for event in events:
        for witness in event.provenance_witnesses:
            if witness.startswith("event:"):
                witness_id = ledger_witness_id(witness)
                if not witness_id or witness_id == event.event_id:
                    raise GroundingError(
                        f"{event.event_id}: invalid ledger witness {witness!r}"
                    )
            elif witness.startswith("external:"):
                parts = witness.split(":", 2)
                if len(parts) != 3 or not parts[1] or not parts[2]:
                    raise GroundingError(
                        f"{event.event_id}: invalid external witness {witness!r}"
                    )
            else:
                raise GroundingError(
                    f"{event.event_id}: witness must use event: or external: typing"
                )
