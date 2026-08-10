"""Version-1 WAM-RX event contract.

Corrections never mutate an earlier row.  They are control events targeting the
earlier event.  A correction normally consists of a retraction plus a new
observed/asserted event in one atomic batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from .canonical import canonical_json, sha256_json

EVENT_SCHEMA_VERSION = 1


class SpeechAct(str, Enum):
    OBSERVED = "observed"
    ASSERTED = "asserted"
    INFERRED = "inferred"
    VERIFIED = "verified"
    REFUTED = "refuted"
    RETRACTED = "retracted"
    TOMBSTONE = "tombstone"


CONTENT_ACTS = frozenset(
    {SpeechAct.OBSERVED, SpeechAct.ASSERTED, SpeechAct.INFERRED}
)
CONTROL_ACTS = frozenset(
    {
        SpeechAct.VERIFIED,
        SpeechAct.REFUTED,
        SpeechAct.RETRACTED,
        SpeechAct.TOMBSTONE,
    }
)


def _parse_timestamp(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


@dataclass(frozen=True)
class Event:
    event_id: str
    parent_ids: tuple[str, ...]
    transaction_time: str
    valid_from: str
    valid_to: str | None
    actor: str
    source_id: str
    verifier_id: str | None
    modality: str
    speech_act: SpeechAct
    payload: dict[str, Any]
    target_event_ids: tuple[str, ...]
    confidence: float | None
    verifier_class: str
    provenance_witnesses: tuple[str, ...]
    access_policy: str
    retention_policy: str
    ontology_version: str
    resolver_version: str
    signature: str | None
    content_hash: str
    schema_version: int = EVENT_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        event_id: str,
        transaction_time: str,
        valid_from: str,
        actor: str,
        source_id: str,
        modality: str,
        speech_act: SpeechAct | str,
        payload: dict[str, Any] | None = None,
        parent_ids: tuple[str, ...] | list[str] = (),
        target_event_ids: tuple[str, ...] | list[str] = (),
        valid_to: str | None = None,
        confidence: float | None = None,
        verifier_class: str = "unverified",
        verifier_id: str | None = None,
        provenance_witnesses: tuple[str, ...] | list[str] = (),
        access_policy: str = "default",
        retention_policy: str = "retain",
        ontology_version: str = "v1",
        resolver_version: str = "v1",
        signature: str | None = None,
    ) -> "Event":
        act = SpeechAct(speech_act)
        body = dict(payload or {})
        event = cls(
            event_id=event_id,
            parent_ids=tuple(sorted(set(parent_ids))),
            transaction_time=transaction_time,
            valid_from=valid_from,
            valid_to=valid_to,
            actor=actor,
            source_id=source_id,
            verifier_id=verifier_id,
            modality=modality,
            speech_act=act,
            payload=body,
            target_event_ids=tuple(sorted(set(target_event_ids))),
            confidence=confidence,
            verifier_class=verifier_class,
            provenance_witnesses=tuple(sorted(set(provenance_witnesses))),
            access_policy=access_policy,
            retention_policy=retention_policy,
            ontology_version=ontology_version,
            resolver_version=resolver_version,
            signature=signature,
            content_hash=sha256_json(body),
        )
        event.validate()
        return event

    def validate(self) -> None:
        if self.schema_version != EVENT_SCHEMA_VERSION:
            raise ValueError(f"unsupported event schema {self.schema_version}")
        required = {
            "event_id": self.event_id,
            "actor": self.actor,
            "source_id": self.source_id,
            "modality": self.modality,
            "verifier_class": self.verifier_class,
            "access_policy": self.access_policy,
            "retention_policy": self.retention_policy,
            "ontology_version": self.ontology_version,
            "resolver_version": self.resolver_version,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"event fields cannot be empty: {', '.join(missing)}")
        if self.verifier_class != "unverified" and not self.verifier_id:
            raise ValueError(
                "verified/grounded events require an explicit verifier_id"
            )
        transaction = _parse_timestamp(self.transaction_time, "transaction_time")
        valid_from = _parse_timestamp(self.valid_from, "valid_from")
        if self.valid_to is not None:
            valid_to = _parse_timestamp(self.valid_to, "valid_to")
            if valid_to <= valid_from:
                raise ValueError("valid_to must be later than valid_from")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.event_id in self.parent_ids or self.event_id in self.target_event_ids:
            raise ValueError("an event cannot parent or target itself")
        if self.speech_act in CONTENT_ACTS and self.target_event_ids:
            raise ValueError("content events cannot target earlier events")
        if self.speech_act in CONTROL_ACTS and not self.target_event_ids:
            raise ValueError(f"{self.speech_act.value} must target at least one event")
        if self.content_hash != sha256_json(self.payload):
            raise ValueError("payload does not match content_hash")
        # Parsed solely to reject platform-dependent/non-ISO strings.
        del transaction

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "parent_ids": list(self.parent_ids),
            "transaction_time": self.transaction_time,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "actor": self.actor,
            "source_id": self.source_id,
            "verifier_id": self.verifier_id,
            "modality": self.modality,
            "speech_act": self.speech_act.value,
            "payload": self.payload,
            "target_event_ids": list(self.target_event_ids),
            "confidence": self.confidence,
            "verifier_class": self.verifier_class,
            "provenance_witnesses": list(self.provenance_witnesses),
            "access_policy": self.access_policy,
            "retention_policy": self.retention_policy,
            "ontology_version": self.ontology_version,
            "resolver_version": self.resolver_version,
            "signature": self.signature,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Event":
        event = cls(
            schema_version=int(value["schema_version"]),
            event_id=str(value["event_id"]),
            parent_ids=tuple(value.get("parent_ids", ())),
            transaction_time=str(value["transaction_time"]),
            valid_from=str(value["valid_from"]),
            valid_to=value.get("valid_to"),
            actor=str(value["actor"]),
            source_id=str(value["source_id"]),
            verifier_id=value.get("verifier_id"),
            modality=str(value["modality"]),
            speech_act=SpeechAct(value["speech_act"]),
            payload=dict(value.get("payload", {})),
            target_event_ids=tuple(value.get("target_event_ids", ())),
            confidence=value.get("confidence"),
            verifier_class=str(value["verifier_class"]),
            provenance_witnesses=tuple(value.get("provenance_witnesses", ())),
            access_policy=str(value["access_policy"]),
            retention_policy=str(value["retention_policy"]),
            ontology_version=str(value["ontology_version"]),
            resolver_version=str(value["resolver_version"]),
            signature=value.get("signature"),
            content_hash=str(value["content_hash"]),
        )
        event.validate()
        return event

    @property
    def event_hash(self) -> str:
        return sha256_json(self.to_dict())

    @property
    def canonical(self) -> str:
        return canonical_json(self.to_dict())

    def is_valid_at(self, valid_at: str) -> bool:
        point = _parse_timestamp(valid_at, "valid_at")
        start = _parse_timestamp(self.valid_from, "valid_from")
        end = _parse_timestamp(self.valid_to, "valid_to") if self.valid_to else None
        return start <= point and (end is None or point < end)
