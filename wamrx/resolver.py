"""Deterministic, contradiction-preserving replay of version-1 events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .canonical import sha256_json
from .events import CONTENT_ACTS, Event, SpeechAct
from .store import AppendOnlyEventStore, LedgerFrontier

RESOLVER_VERSION = "wamrx-resolver-v1"

_PRECEDENCE = {
    SpeechAct.VERIFIED: 1,
    SpeechAct.REFUTED: 2,
    SpeechAct.RETRACTED: 3,
    SpeechAct.TOMBSTONE: 4,
}


@dataclass(frozen=True)
class ResolvedRecord:
    event_id: str
    status: str
    payload: dict[str, Any]
    region: str
    actor: str
    source_id: str
    verifier_id: str | None
    verifier_class: str
    confidence: float | None
    modality: str
    ontology_version: str
    valid_from: str
    valid_to: str | None
    control_event_ids: tuple[str, ...]

    @property
    def usable(self) -> bool:
        return self.status in {"observed", "asserted", "inferred", "verified"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "status": self.status,
            "payload": self.payload,
            "region": self.region,
            "actor": self.actor,
            "source_id": self.source_id,
            "verifier_id": self.verifier_id,
            "verifier_class": self.verifier_class,
            "confidence": self.confidence,
            "modality": self.modality,
            "ontology_version": self.ontology_version,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "control_event_ids": list(self.control_event_ids),
        }


@dataclass(frozen=True)
class ResolvedSnapshot:
    valid_at: str
    records: tuple[ResolvedRecord, ...]
    frontier: LedgerFrontier
    resolver_version: str = RESOLVER_VERSION

    @property
    def snapshot_hash(self) -> str:
        # Frontier identity is reported separately.  This hash represents the
        # resolved state, so equivalent replays can be compared directly.
        return sha256_json(
            {
                "valid_at": self.valid_at,
                "resolver_version": self.resolver_version,
                "records": [record.to_dict() for record in self.records],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid_at": self.valid_at,
            "resolver_version": self.resolver_version,
            "frontier": self.frontier.to_dict(),
            "snapshot_hash": self.snapshot_hash,
            "records": [record.to_dict() for record in self.records],
        }

    def record(self, event_id: str) -> ResolvedRecord | None:
        return next((record for record in self.records if record.event_id == event_id), None)

    def usable_event_ids(self) -> set[str]:
        return {record.event_id for record in self.records if record.usable}


def _resolve_events(
    events: Iterable[Event], valid_at: str, frontier: LedgerFrontier
) -> ResolvedSnapshot:
    # The iterable order is the ledger transaction sequence. Caller-supplied
    # timestamps are canonical metadata and never reorder an append.
    ordered = list(events)
    controls: dict[str, list[tuple[int, Event]]] = {}
    content = []
    for sequence, event in enumerate(ordered, start=1):
        if event.speech_act in CONTENT_ACTS:
            content.append(event)
        elif event.is_valid_at(valid_at):
            for target in event.target_event_ids:
                controls.setdefault(target, []).append((sequence, event))

    records = []
    for event in content:
        if not event.is_valid_at(valid_at):
            status = "out_of_validity"
            applicable: list[Event] = []
        else:
            applicable = controls.get(event.event_id, [])
            if applicable:
                _, strongest = max(
                    applicable,
                    key=lambda item: (
                        _PRECEDENCE[item[1].speech_act],
                        item[0],
                    ),
                )
                status = strongest.speech_act.value
            else:
                status = event.speech_act.value
        records.append(
            ResolvedRecord(
                event_id=event.event_id,
                status=status,
                payload=event.payload,
                region=str(event.payload.get("region", "unclassified")),
                actor=event.actor,
                source_id=event.source_id,
                verifier_id=event.verifier_id,
                verifier_class=event.verifier_class,
                confidence=event.confidence,
                modality=event.modality,
                ontology_version=event.ontology_version,
                valid_from=event.valid_from,
                valid_to=event.valid_to,
                control_event_ids=tuple(
                    control.event_id for _, control in applicable
                ),
            )
        )
    return ResolvedSnapshot(
        valid_at=valid_at,
        records=tuple(sorted(records, key=lambda record: record.event_id)),
        frontier=frontier,
    )


def resolve(store: AppendOnlyEventStore, *, valid_at: str) -> ResolvedSnapshot:
    return _resolve_events(store.events(), valid_at, store.frontier())


def replay(events: Iterable[Event], *, valid_at: str) -> ResolvedSnapshot:
    items = list(events)
    return _resolve_events(
        items,
        valid_at,
        LedgerFrontier(sequence=len(items), chain_hash="replay"),
    )
