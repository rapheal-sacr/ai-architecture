"""Authority-limited, fixed-capacity native working-memory boundary.

The implementation is intentionally non-neural.  A later learned gate may choose
the registered remember/update/merge/forget operations, but this module owns the
authority, evidence, session, capacity, version, and accounting checks that the
gate is not allowed to bypass.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
from typing import Any, Mapping

from .artifacts import SupportManifest
from .canonical import canonical_json, sha256_json
from .events import Event, SpeechAct, canonical_timestamp
from .resolver import resolve
from .store import AppendOnlyEventStore


NATIVE_MEMORY_SCHEMA_VERSION = 1
NATIVE_MEMORY_COMPONENT_VERSION = "wamrx-native-memory-boundary-v1"
CHECKPOINT_FORMAT = "wamrx-native-memory-checkpoint-v1"
REGISTERED_OPERATIONS = ("remember", "update", "merge", "forget")
REGISTERED_FAILURE_STATUSES = ("NOT_RUN", "INCOMPLETE", "INVALID", "PASS", "FAIL")

# Exact, deliberately simple analytical accounting for the non-neural boundary.
# The future comparison must add model FLOPs separately.
OPERATION_BASE_FLOPS = {
    "remember": 64,
    "update": 72,
    "merge": 96,
    "forget": 24,
    "model_call": 48,
    "durable_candidate": 56,
    "checkpoint": 40,
    "restore": 40,
    "reset": 16,
}
FLOPS_PER_CANONICAL_INPUT_BYTE = 2
FLOPS_PER_SCANNED_SLOT = 8


class NativeMemoryError(RuntimeError):
    """Base class for a fail-closed native-memory operation."""


class AuthorityViolation(NativeMemoryError):
    pass


class AccessViolation(NativeMemoryError):
    pass


class SessionExpired(NativeMemoryError):
    pass


class CapacityExceeded(NativeMemoryError):
    pass


class MemoryCollision(NativeMemoryError):
    pass


class EvidenceViolation(NativeMemoryError):
    pass


class IncompatibleCheckpoint(NativeMemoryError):
    pass


class InterruptedCheckpoint(NativeMemoryError):
    pass


class ProtectedRetentionViolation(NativeMemoryError):
    pass


def _nonempty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise NativeMemoryError(f"{label} must be a non-empty string")
    return value


def _canonical_size(value: Any) -> int:
    return len(canonical_json(value).encode("utf-8"))


def _at_or_after(left: str, right: str) -> bool:
    return datetime.fromisoformat(left) >= datetime.fromisoformat(right)


@dataclass(frozen=True)
class NativeMemoryConfig:
    session_id: str
    task_id: str
    owner_id: str
    maximum_slots: int
    maximum_serialized_bytes: int
    expires_at: str
    base_weight_version: str
    ontology_version: str
    selected_core_id: str = "fixed-depth-v1"
    selected_macro_depth: int = 4
    component_version: str = NATIVE_MEMORY_COMPONENT_VERSION
    protected_regions: tuple[str, ...] = ("finance", "operations", "rare-protected")

    def __post_init__(self) -> None:
        for label in (
            "session_id",
            "task_id",
            "owner_id",
            "base_weight_version",
            "ontology_version",
            "selected_core_id",
            "component_version",
        ):
            _nonempty(str(getattr(self, label)), label)
        if self.maximum_slots < 1:
            raise NativeMemoryError("maximum_slots must be positive")
        if self.maximum_serialized_bytes < 1:
            raise NativeMemoryError("maximum_serialized_bytes must be positive")
        if self.selected_core_id != "fixed-depth-v1" or self.selected_macro_depth != 4:
            raise NativeMemoryError("native memory is bound to fixed-depth-v1 at depth 4")
        if self.component_version != NATIVE_MEMORY_COMPONENT_VERSION:
            raise NativeMemoryError("unsupported native-memory component version")
        object.__setattr__(
            self,
            "expires_at",
            canonical_timestamp(self.expires_at, "expires_at"),
        )
        if not self.protected_regions or any(not item for item in self.protected_regions):
            raise NativeMemoryError("protected_regions must be non-empty strings")

    def identity_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "owner_id": self.owner_id,
            "maximum_slots": self.maximum_slots,
            "maximum_serialized_bytes": self.maximum_serialized_bytes,
            "expires_at": self.expires_at,
            "base_weight_version": self.base_weight_version,
            "ontology_version": self.ontology_version,
            "selected_core_id": self.selected_core_id,
            "selected_macro_depth": self.selected_macro_depth,
            "component_version": self.component_version,
            "protected_regions": list(self.protected_regions),
        }


@dataclass(frozen=True)
class MemoryAccess:
    session_id: str
    task_id: str
    owner_id: str
    session_epoch: int


@dataclass(frozen=True)
class MemoryEvidenceBundle:
    event_ids: tuple[str, ...]
    valid_at: str
    ontology_version: str
    ledger_frontier_sequence: int
    ledger_frontier_hash: str
    resolved_snapshot_hash: str
    event_content_hashes: dict[str, str]

    @classmethod
    def create(
        cls,
        store: AppendOnlyEventStore,
        *,
        event_ids: tuple[str, ...] | list[str],
        valid_at: str,
        ontology_version: str,
    ) -> "MemoryEvidenceBundle":
        ids = tuple(sorted(set(event_ids)))
        if not ids:
            raise EvidenceViolation("every model call requires reinjected ledger evidence")
        canonical_valid_at = canonical_timestamp(valid_at, "valid_at")
        snapshot = resolve(store, valid_at=canonical_valid_at)
        records = {record.event_id: record for record in snapshot.records}
        hashes: dict[str, str] = {}
        for event_id in ids:
            event = store.get_event(event_id)
            record = records.get(event_id)
            if event is None or record is None:
                raise EvidenceViolation(f"missing evidence event {event_id!r}")
            if not record.usable:
                raise EvidenceViolation(f"evidence event {event_id!r} is not usable")
            if record.ontology_version != ontology_version:
                raise EvidenceViolation(f"evidence event {event_id!r} has stale ontology")
            if record.verifier_class == "unverified":
                raise EvidenceViolation(f"evidence event {event_id!r} is unverified")
            hashes[event_id] = event.content_hash
        return cls(
            event_ids=ids,
            valid_at=canonical_valid_at,
            ontology_version=ontology_version,
            ledger_frontier_sequence=snapshot.frontier.sequence,
            ledger_frontier_hash=snapshot.frontier.chain_hash,
            resolved_snapshot_hash=snapshot.snapshot_hash,
            event_content_hashes=dict(sorted(hashes.items())),
        )

    @property
    def bundle_hash(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_ids": list(self.event_ids),
            "valid_at": self.valid_at,
            "ontology_version": self.ontology_version,
            "ledger_frontier_sequence": self.ledger_frontier_sequence,
            "ledger_frontier_hash": self.ledger_frontier_hash,
            "resolved_snapshot_hash": self.resolved_snapshot_hash,
            "event_content_hashes": self.event_content_hashes,
        }

    def revalidate(self, store: AppendOnlyEventStore) -> None:
        rebuilt = MemoryEvidenceBundle.create(
            store,
            event_ids=self.event_ids,
            valid_at=self.valid_at,
            ontology_version=self.ontology_version,
        )
        if rebuilt.to_dict() != self.to_dict():
            raise EvidenceViolation("evidence bundle is stale or was substituted")


@dataclass(frozen=True)
class MemorySlot:
    slot_id: str
    content_key: str
    value: Any
    support_event_ids: tuple[str, ...]
    superseded_support_event_ids: tuple[str, ...]
    protected_regions: tuple[str, ...]
    created_turn: int
    updated_turn: int
    active: bool = True
    disabled_reason: str | None = None

    @property
    def content_hash(self) -> str:
        return sha256_json(
            {
                "content_key": self.content_key,
                "value": self.value,
                "support_event_ids": list(self.support_event_ids),
                "superseded_support_event_ids": list(self.superseded_support_event_ids),
                "protected_regions": list(self.protected_regions),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "content_key": self.content_key,
            "value": self.value,
            "support_event_ids": list(self.support_event_ids),
            "superseded_support_event_ids": list(self.superseded_support_event_ids),
            "protected_regions": list(self.protected_regions),
            "created_turn": self.created_turn,
            "updated_turn": self.updated_turn,
            "active": self.active,
            "disabled_reason": self.disabled_reason,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MemorySlot":
        slot = cls(
            slot_id=str(value["slot_id"]),
            content_key=str(value["content_key"]),
            value=value["value"],
            support_event_ids=tuple(value["support_event_ids"]),
            superseded_support_event_ids=tuple(value["superseded_support_event_ids"]),
            protected_regions=tuple(value["protected_regions"]),
            created_turn=int(value["created_turn"]),
            updated_turn=int(value["updated_turn"]),
            active=bool(value["active"]),
            disabled_reason=value.get("disabled_reason"),
        )
        if value.get("content_hash") != slot.content_hash:
            raise IncompatibleCheckpoint("checkpoint slot content hash mismatch")
        return slot


@dataclass(frozen=True)
class MemoryComputeRecord:
    operation_counts: dict[str, int]
    realized_memory_flops: int
    occupied_slots: int
    active_slots: int
    maximum_slots: int
    serialized_slot_bytes: int
    maximum_serialized_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_counts": dict(sorted(self.operation_counts.items())),
            "realized_memory_flops": self.realized_memory_flops,
            "occupied_slots": self.occupied_slots,
            "active_slots": self.active_slots,
            "maximum_slots": self.maximum_slots,
            "serialized_slot_bytes": self.serialized_slot_bytes,
            "maximum_serialized_bytes": self.maximum_serialized_bytes,
        }


@dataclass(frozen=True)
class MemoryRead:
    session_id: str
    task_id: str
    owner_id: str
    session_epoch: int
    state_hash: str
    evidence_bundle_hash: str
    ledger_frontier_sequence: int
    ledger_frontier_hash: str
    active_slots: tuple[MemorySlot, ...]
    disabled_slot_ids: tuple[str, ...]
    compute: MemoryComputeRecord


@dataclass(frozen=True)
class DurableCandidate:
    proposal_id: str
    session_id: str
    task_id: str
    owner_id: str
    payload: dict[str, Any]
    support: SupportManifest
    ledger_frontier_sequence: int
    ledger_frontier_hash: str
    authority: str = "candidate-only"
    direct_ledger_write_authorized: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "owner_id": self.owner_id,
            "payload": self.payload,
            "support": self.support.to_dict(),
            "ledger_frontier_sequence": self.ledger_frontier_sequence,
            "ledger_frontier_hash": self.ledger_frontier_hash,
            "authority": self.authority,
            "direct_ledger_write_authorized": self.direct_ledger_write_authorized,
        }


@dataclass(frozen=True)
class ResetReceipt:
    reason: str
    cleared_slots: int
    prior_epoch: int
    new_access: MemoryAccess
    empty_payload_hash: str


class SessionMemory:
    """Fixed-slot session memory with no reference to a ledger append method."""

    def __init__(self, config: NativeMemoryConfig) -> None:
        self.config = config
        self._epoch = 0
        self._turn = 0
        self._next_slot = 0
        self._slots: dict[str, MemorySlot] = {}
        self._operation_counts: dict[str, int] = {}
        self._flops = 0
        self._expired = False

    @property
    def access(self) -> MemoryAccess:
        return MemoryAccess(
            self.config.session_id,
            self.config.task_id,
            self.config.owner_id,
            self._epoch,
        )

    @property
    def slots(self) -> tuple[MemorySlot, ...]:
        return tuple(self._slots[key] for key in sorted(self._slots))

    @property
    def payload_state_hash(self) -> str:
        return sha256_json([slot.to_dict() for slot in self.slots])

    @property
    def state_hash(self) -> str:
        return sha256_json(
            {
                "identity": self.config.identity_dict(),
                "session_epoch": self._epoch,
                "turn": self._turn,
                "slots": [slot.to_dict() for slot in self.slots],
            }
        )

    def _slot_bytes(self, slots: Mapping[str, MemorySlot] | None = None) -> int:
        values = slots if slots is not None else self._slots
        return _canonical_size([values[key].to_dict() for key in sorted(values)])

    def compute_record(self) -> MemoryComputeRecord:
        return MemoryComputeRecord(
            operation_counts=dict(sorted(self._operation_counts.items())),
            realized_memory_flops=self._flops,
            occupied_slots=len(self._slots),
            active_slots=sum(slot.active for slot in self._slots.values()),
            maximum_slots=self.config.maximum_slots,
            serialized_slot_bytes=self._slot_bytes(),
            maximum_serialized_bytes=self.config.maximum_serialized_bytes,
        )

    def _charge(self, operation: str, payload: Any) -> None:
        if operation not in OPERATION_BASE_FLOPS:
            raise NativeMemoryError(f"unregistered operation {operation!r}")
        self._operation_counts[operation] = self._operation_counts.get(operation, 0) + 1
        self._flops += (
            OPERATION_BASE_FLOPS[operation]
            + FLOPS_PER_CANONICAL_INPUT_BYTE * _canonical_size(payload)
            + FLOPS_PER_SCANNED_SLOT * len(self._slots)
        )

    def _clear(self, reason: str) -> ResetReceipt:
        prior = self._epoch
        cleared = len(self._slots)
        self._slots = {}
        self._turn = 0
        self._next_slot = 0
        self._epoch += 1
        self._charge("reset", {"reason": reason, "cleared_slots": cleared})
        return ResetReceipt(
            reason=reason,
            cleared_slots=cleared,
            prior_epoch=prior,
            new_access=self.access,
            empty_payload_hash=sha256_json([]),
        )

    def _authorize(self, access: MemoryAccess, now: str) -> str:
        canonical_now = canonical_timestamp(now, "now")
        if self._expired:
            raise SessionExpired("native-memory session has expired")
        expected = self.access
        if access != expected:
            raise AccessViolation("memory access is not bound to this owner/session/task epoch")
        if _at_or_after(canonical_now, self.config.expires_at):
            self._clear("expiry")
            self._expired = True
            raise SessionExpired("native-memory session expired and was cleared")
        return canonical_now

    def _validate_bundle(
        self,
        store: AppendOnlyEventStore,
        bundle: MemoryEvidenceBundle,
    ) -> None:
        if bundle.ontology_version != self.config.ontology_version:
            raise EvidenceViolation("evidence bundle has an incompatible ontology")
        bundle.revalidate(store)

    def _require_support(
        self,
        support_event_ids: tuple[str, ...] | list[str],
        bundle: MemoryEvidenceBundle,
    ) -> tuple[str, ...]:
        support = tuple(sorted(set(support_event_ids)))
        if not support:
            raise AuthorityViolation("latent memory cannot be its own evidence root")
        if any(item.startswith(("memory:", "slot:")) for item in support):
            raise AuthorityViolation("memory identifiers cannot be cited as evidence")
        missing = set(support) - set(bundle.event_ids)
        if missing:
            raise EvidenceViolation(f"memory support was not reinjected: {sorted(missing)}")
        return support

    def _commit_slots(self, slots: dict[str, MemorySlot]) -> None:
        if len(slots) > self.config.maximum_slots:
            raise CapacityExceeded("fixed native-memory slot capacity exceeded")
        used = self._slot_bytes(slots)
        if used > self.config.maximum_serialized_bytes:
            raise CapacityExceeded(
                f"serialized native memory exceeds {self.config.maximum_serialized_bytes} bytes"
            )
        self._slots = slots

    def remember(
        self,
        store: AppendOnlyEventStore,
        *,
        access: MemoryAccess,
        now: str,
        bundle: MemoryEvidenceBundle,
        content_key: str,
        value: Any,
        support_event_ids: tuple[str, ...] | list[str],
        protected_regions: tuple[str, ...] | list[str],
    ) -> MemorySlot:
        self._authorize(access, now)
        self._validate_bundle(store, bundle)
        key = _nonempty(content_key, "content_key")
        if any(slot.content_key == key for slot in self._slots.values()):
            raise MemoryCollision("remember cannot silently overwrite a colliding key")
        support = self._require_support(support_event_ids, bundle)
        regions = tuple(sorted(set(protected_regions)))
        if not regions:
            raise NativeMemoryError("every slot requires at least one region")
        self._turn += 1
        self._next_slot += 1
        slot = MemorySlot(
            slot_id=f"memory:{self.config.session_id}:{self._epoch}:{self._next_slot}",
            content_key=key,
            value=value,
            support_event_ids=support,
            superseded_support_event_ids=(),
            protected_regions=regions,
            created_turn=self._turn,
            updated_turn=self._turn,
        )
        proposed = dict(self._slots)
        proposed[slot.slot_id] = slot
        self._commit_slots(proposed)
        self._charge("remember", slot.to_dict())
        return slot

    def update(
        self,
        store: AppendOnlyEventStore,
        *,
        access: MemoryAccess,
        now: str,
        bundle: MemoryEvidenceBundle,
        slot_id: str,
        value: Any,
        support_event_ids: tuple[str, ...] | list[str],
    ) -> MemorySlot:
        self._authorize(access, now)
        self._validate_bundle(store, bundle)
        old = self._slots.get(slot_id)
        if old is None or not old.active:
            raise NativeMemoryError("only an active slot can be updated")
        support = self._require_support(support_event_ids, bundle)
        self._turn += 1
        updated = replace(
            old,
            value=value,
            support_event_ids=support,
            superseded_support_event_ids=tuple(
                sorted(set((*old.superseded_support_event_ids, *old.support_event_ids)))
            ),
            updated_turn=self._turn,
        )
        proposed = dict(self._slots)
        proposed[slot_id] = updated
        self._commit_slots(proposed)
        self._charge("update", updated.to_dict())
        return updated

    def merge(
        self,
        store: AppendOnlyEventStore,
        *,
        access: MemoryAccess,
        now: str,
        bundle: MemoryEvidenceBundle,
        slot_ids: tuple[str, ...] | list[str],
        content_key: str,
        value: Any,
    ) -> MemorySlot:
        self._authorize(access, now)
        self._validate_bundle(store, bundle)
        ids = tuple(sorted(set(slot_ids)))
        if len(ids) < 2:
            raise NativeMemoryError("merge requires at least two distinct slots")
        sources = [self._slots.get(slot_id) for slot_id in ids]
        if any(slot is None or not slot.active for slot in sources):
            raise NativeMemoryError("merge sources must all be active")
        typed_sources = [slot for slot in sources if slot is not None]
        support = self._require_support(
            tuple(event_id for slot in typed_sources for event_id in slot.support_event_ids),
            bundle,
        )
        key = _nonempty(content_key, "content_key")
        if any(
            slot.content_key == key and slot.slot_id not in ids
            for slot in self._slots.values()
        ):
            raise MemoryCollision("merged content key collides with another slot")
        self._turn += 1
        self._next_slot += 1
        merged = MemorySlot(
            slot_id=f"memory:{self.config.session_id}:{self._epoch}:{self._next_slot}",
            content_key=key,
            value=value,
            support_event_ids=support,
            superseded_support_event_ids=tuple(
                sorted(
                    {
                        event_id
                        for slot in typed_sources
                        for event_id in slot.superseded_support_event_ids
                    }
                )
            ),
            protected_regions=tuple(
                sorted({region for slot in typed_sources for region in slot.protected_regions})
            ),
            created_turn=self._turn,
            updated_turn=self._turn,
        )
        proposed = {key_id: slot for key_id, slot in self._slots.items() if key_id not in ids}
        proposed[merged.slot_id] = merged
        self._commit_slots(proposed)
        self._charge("merge", {"source_slot_ids": list(ids), "merged": merged.to_dict()})
        return merged

    def forget(
        self,
        *,
        access: MemoryAccess,
        now: str,
        slot_id: str,
        reason: str,
    ) -> None:
        self._authorize(access, now)
        slot = self._slots.get(slot_id)
        if slot is None:
            raise NativeMemoryError("cannot forget an unknown slot")
        if reason not in {"explicit-user", "correction", "invalidation", "decay", "capacity"}:
            raise NativeMemoryError("forget reason is not registered")
        protected = set(slot.protected_regions) & set(self.config.protected_regions)
        if reason in {"decay", "capacity"} and protected:
            raise ProtectedRetentionViolation(
                "automatic forgetting cannot delete protected-region state"
            )
        proposed = dict(self._slots)
        del proposed[slot_id]
        self._commit_slots(proposed)
        self._turn += 1
        self._charge("forget", {"slot_id": slot_id, "reason": reason})

    def _refresh_support(self, store: AppendOnlyEventStore, valid_at: str) -> None:
        snapshot = resolve(store, valid_at=valid_at)
        records = {record.event_id: record for record in snapshot.records}
        changed = dict(self._slots)
        for slot_id, slot in self._slots.items():
            if not slot.active:
                continue
            invalid = sorted(
                event_id
                for event_id in slot.support_event_ids
                if (record := records.get(event_id)) is None
                or not record.usable
                or record.ontology_version != self.config.ontology_version
                or record.verifier_class == "unverified"
            )
            if invalid:
                changed[slot_id] = replace(
                    slot,
                    active=False,
                    disabled_reason=f"invalid-support:{','.join(invalid)}",
                )
        self._slots = changed

    def prepare_model_call(
        self,
        store: AppendOnlyEventStore,
        *,
        access: MemoryAccess,
        now: str,
        bundle: MemoryEvidenceBundle,
    ) -> MemoryRead:
        self._authorize(access, now)
        self._refresh_support(store, bundle.valid_at)
        self._validate_bundle(store, bundle)
        required = {
            event_id
            for slot in self._slots.values()
            if slot.active
            for event_id in slot.support_event_ids
        }
        missing = sorted(required - set(bundle.event_ids))
        if missing:
            raise EvidenceViolation(
                f"model call omitted active memory evidence: {missing}"
            )
        self._charge(
            "model_call",
            {"bundle_hash": bundle.bundle_hash, "active_support": sorted(required)},
        )
        return MemoryRead(
            session_id=self.config.session_id,
            task_id=self.config.task_id,
            owner_id=self.config.owner_id,
            session_epoch=self._epoch,
            state_hash=self.state_hash,
            evidence_bundle_hash=bundle.bundle_hash,
            ledger_frontier_sequence=bundle.ledger_frontier_sequence,
            ledger_frontier_hash=bundle.ledger_frontier_hash,
            active_slots=tuple(slot for slot in self.slots if slot.active),
            disabled_slot_ids=tuple(slot.slot_id for slot in self.slots if not slot.active),
            compute=self.compute_record(),
        )

    def answer_manifest(
        self,
        read: MemoryRead,
        *,
        slot_ids: tuple[str, ...] | list[str],
    ) -> SupportManifest:
        if read.state_hash != self.state_hash or read.session_epoch != self._epoch:
            raise EvidenceViolation("memory read is no longer current")
        active = {slot.slot_id: slot for slot in read.active_slots}
        ids = tuple(sorted(set(slot_ids)))
        if not ids or any(slot_id not in active for slot_id in ids):
            raise EvidenceViolation("answer cites an absent or disabled memory slot")
        support = tuple(
            sorted({event_id for slot_id in ids for event_id in active[slot_id].support_event_ids})
        )
        if any(item.startswith(("memory:", "slot:")) for item in support):
            raise AuthorityViolation("memory state cannot become independent evidence")
        return SupportManifest.create(
            supporting_event_ids=support,
            candidate_event_ids=support,
        )

    def durable_candidate(
        self,
        store: AppendOnlyEventStore,
        *,
        access: MemoryAccess,
        now: str,
        read: MemoryRead,
        proposal_id: str,
        payload: dict[str, Any],
        slot_ids: tuple[str, ...] | list[str],
    ) -> DurableCandidate:
        self._authorize(access, now)
        current = store.frontier()
        if (
            current.sequence != read.ledger_frontier_sequence
            or current.chain_hash != read.ledger_frontier_hash
        ):
            raise EvidenceViolation("durable candidate was derived from a stale ledger frontier")
        manifest = self.answer_manifest(read, slot_ids=slot_ids)
        events = {event.event_id: event for event in store.events()}
        supports = [events.get(event_id) for event_id in manifest.supporting_event_ids]
        if any(event is None for event in supports):
            raise AuthorityViolation("durable proposal names missing ledger evidence")
        if not any(_reaches_observation(event, events) for event in supports if event):
            raise AuthorityViolation("durable proposal has no observed ledger root")
        self._charge(
            "durable_candidate",
            {"proposal_id": proposal_id, "payload": payload, "support": manifest.to_dict()},
        )
        return DurableCandidate(
            proposal_id=_nonempty(proposal_id, "proposal_id"),
            session_id=self.config.session_id,
            task_id=self.config.task_id,
            owner_id=self.config.owner_id,
            payload=dict(payload),
            support=manifest,
            ledger_frontier_sequence=current.sequence,
            ledger_frontier_hash=current.chain_hash,
        )

    def reset(self, *, access: MemoryAccess, now: str) -> ResetReceipt:
        self._authorize(access, now)
        return self._clear("explicit-reset")

    def checkpoint(
        self,
        *,
        access: MemoryAccess,
        now: str,
        interrupt_after_slots: int | None = None,
    ) -> bytes:
        self._authorize(access, now)
        slot_rows = []
        for index, slot in enumerate(self.slots, start=1):
            slot_rows.append(slot.to_dict())
            if interrupt_after_slots is not None and index >= interrupt_after_slots:
                raise InterruptedCheckpoint("memory checkpoint interrupted before commit")
        body = {
            "schema_version": NATIVE_MEMORY_SCHEMA_VERSION,
            "format": CHECKPOINT_FORMAT,
            "config": self.config.identity_dict(),
            "session_epoch": self._epoch,
            "turn": self._turn,
            "next_slot": self._next_slot,
            "slots": slot_rows,
            "operation_counts": dict(sorted(self._operation_counts.items())),
            "realized_memory_flops": self._flops,
            "payload_state_hash": self.payload_state_hash,
        }
        self._charge("checkpoint", body)
        # Include the checkpoint charge in the committed record.
        body["operation_counts"] = dict(sorted(self._operation_counts.items()))
        body["realized_memory_flops"] = self._flops
        envelope = {"checkpoint": body, "checkpoint_sha256": sha256_json(body)}
        return canonical_json(envelope).encode("utf-8")

    @classmethod
    def restore(
        cls,
        payload: bytes,
        *,
        expected_config: NativeMemoryConfig,
        expected_access: MemoryAccess,
    ) -> "SessionMemory":
        try:
            decoded = json.loads(payload.decode("utf-8"))
            body = decoded["checkpoint"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise IncompatibleCheckpoint("malformed native-memory checkpoint") from exc
        if decoded.get("checkpoint_sha256") != sha256_json(body):
            raise IncompatibleCheckpoint("native-memory checkpoint hash mismatch")
        if (
            body.get("schema_version") != NATIVE_MEMORY_SCHEMA_VERSION
            or body.get("format") != CHECKPOINT_FORMAT
        ):
            raise IncompatibleCheckpoint("unsupported native-memory checkpoint format")
        if body.get("config") != expected_config.identity_dict():
            raise IncompatibleCheckpoint("checkpoint model/ontology/session configuration mismatch")
        if int(body.get("session_epoch", -1)) != expected_access.session_epoch:
            raise IncompatibleCheckpoint("checkpoint session epoch mismatch")
        if (
            expected_access.session_id != expected_config.session_id
            or expected_access.task_id != expected_config.task_id
            or expected_access.owner_id != expected_config.owner_id
        ):
            raise IncompatibleCheckpoint("checkpoint access identity mismatch")
        memory = cls(expected_config)
        memory._epoch = int(body["session_epoch"])
        memory._turn = int(body["turn"])
        memory._next_slot = int(body["next_slot"])
        slots = [MemorySlot.from_dict(item) for item in body["slots"]]
        memory._commit_slots({slot.slot_id: slot for slot in slots})
        if memory.payload_state_hash != body.get("payload_state_hash"):
            raise IncompatibleCheckpoint("checkpoint payload state hash mismatch")
        memory._operation_counts = {
            str(key): int(value) for key, value in body["operation_counts"].items()
        }
        memory._flops = int(body["realized_memory_flops"])
        memory._charge("restore", {"checkpoint_sha256": decoded["checkpoint_sha256"]})
        return memory


def _reaches_observation(event: Event, events: Mapping[str, Event]) -> bool:
    pending = [event.event_id]
    seen: set[str] = set()
    while pending:
        event_id = pending.pop()
        if event_id in seen:
            continue
        seen.add(event_id)
        current = events.get(event_id)
        if current is None:
            return False
        if current.speech_act == SpeechAct.OBSERVED:
            return True
        pending.extend(current.parent_ids)
        pending.extend(
            witness.removeprefix("event:")
            for witness in current.provenance_witnesses
            if witness.startswith("event:")
        )
    return False


def checkpoint_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
