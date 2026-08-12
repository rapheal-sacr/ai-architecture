"""Authority-limited metamemory and explicit-view compression boundary."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from .canonical import canonical_json, sha256_json
from .events import SpeechAct
from .metamemory_tasks import MetamemoryTask
from .resolver import resolve
from .store import AppendOnlyEventStore


METAMEMORY_COMPONENT_VERSION = "wamrx-metamemory-boundary-v1"
POLICY_ACTIONS = (
    "ignore",
    "stage",
    "link",
    "retrieve",
    "summarize",
    "structure",
    "request_evidence",
)
POLICY_BASE_FLOPS = {
    "ignore": 8,
    "stage": 64,
    "link": 40,
    "retrieve": 32,
    "summarize": 96,
    "structure": 80,
    "request_evidence": 24,
}
FLOPS_PER_CANONICAL_INPUT_BYTE = 2
FLOPS_PER_SCANNED_ITEM = 8
MAXIMUM_SOURCE_ITEMS = 64
MAXIMUM_SERIALIZED_BYTES = 32768
MINIMUM_SKILL_REPETITIONS = 3


class MetamemoryError(RuntimeError):
    pass


class MetamemoryAuthorityViolation(MetamemoryError):
    pass


class MetamemoryEvidenceViolation(MetamemoryError):
    pass


class MetamemoryCoverageViolation(MetamemoryError):
    pass


class MetamemoryCapacityExceeded(MetamemoryError):
    pass


class StaleCompressionArtifact(MetamemoryError):
    pass


@dataclass(frozen=True)
class MetamemoryComputeRecord:
    operation_counts: dict[str, int]
    realized_policy_flops: int
    serialized_decision_bytes: int
    serialized_artifact_bytes: int
    serialized_skill_candidate_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_counts": dict(sorted(self.operation_counts.items())),
            "realized_policy_flops": self.realized_policy_flops,
            "serialized_decision_bytes": self.serialized_decision_bytes,
            "serialized_artifact_bytes": self.serialized_artifact_bytes,
            "serialized_skill_candidate_bytes": self.serialized_skill_candidate_bytes,
        }


@dataclass(frozen=True)
class MetamemoryDecision:
    decision_id: str
    policy_id: str
    task_id: str
    action: str
    selected_item_ids: tuple[str, ...]
    supporting_event_ids: tuple[str, ...]
    reason: str
    authority: str = "candidate-only"
    direct_ledger_write_authorized: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "policy_id": self.policy_id,
            "task_id": self.task_id,
            "action": self.action,
            "selected_item_ids": list(self.selected_item_ids),
            "supporting_event_ids": list(self.supporting_event_ids),
            "reason": self.reason,
            "authority": self.authority,
            "direct_ledger_write_authorized": self.direct_ledger_write_authorized,
        }


@dataclass(frozen=True)
class CompressionArtifact:
    artifact_id: str
    task_id: str
    source_item_ids: tuple[str, ...]
    source_item_hash: str
    supporting_event_ids: tuple[str, ...]
    region_manifest: dict[str, tuple[str, ...]]
    contradiction_item_ids: tuple[str, ...]
    refutation_item_ids: tuple[str, ...]
    temporal_qualifiers: dict[str, str]
    unresolved_item_ids: tuple[str, ...]
    summary_text: str
    summary_digest: str
    ontology_version: str
    ledger_frontier_sequence: int
    ledger_frontier_hash: str
    authority: str = "candidate-only-advisory-view"
    independent_evidence: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "task_id": self.task_id,
            "source_item_ids": list(self.source_item_ids),
            "source_item_hash": self.source_item_hash,
            "supporting_event_ids": list(self.supporting_event_ids),
            "region_manifest": {
                key: list(value) for key, value in sorted(self.region_manifest.items())
            },
            "contradiction_item_ids": list(self.contradiction_item_ids),
            "refutation_item_ids": list(self.refutation_item_ids),
            "temporal_qualifiers": dict(sorted(self.temporal_qualifiers.items())),
            "unresolved_item_ids": list(self.unresolved_item_ids),
            "summary_text": self.summary_text,
            "summary_digest": self.summary_digest,
            "ontology_version": self.ontology_version,
            "ledger_frontier_sequence": self.ledger_frontier_sequence,
            "ledger_frontier_hash": self.ledger_frontier_hash,
            "authority": self.authority,
            "independent_evidence": self.independent_evidence,
        }


@dataclass(frozen=True)
class SkillCandidate:
    candidate_id: str
    task_id: str
    procedure_key: str
    repetition_item_ids: tuple[str, ...]
    supporting_event_ids: tuple[str, ...]
    authority: str = "candidate-only"
    promotion_authority: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "task_id": self.task_id,
            "procedure_key": self.procedure_key,
            "repetition_item_ids": list(self.repetition_item_ids),
            "supporting_event_ids": list(self.supporting_event_ids),
            "authority": self.authority,
            "promotion_authority": self.promotion_authority,
        }


def _canonical_size(value: Any) -> int:
    return len(canonical_json(value).encode("utf-8"))


def _items(task: MetamemoryTask) -> dict[str, dict[str, Any]]:
    return {str(item["item_id"]): dict(item) for item in task.items}


def _support_for(items: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(
        sorted({str(item["event_id"]) for item in items if bool(item["verified"])})
    )


def _usable_events(
    store: AppendOnlyEventStore,
    *,
    valid_at: str,
    ontology_version: str,
) -> dict[str, Any]:
    snapshot = resolve(store, valid_at=valid_at)
    return {
        record.event_id: record
        for record in snapshot.records
        if record.usable
        and record.ontology_version == ontology_version
        and record.verifier_class != "unverified"
    }


def _validate_ledger_support(
    store: AppendOnlyEventStore,
    support: tuple[str, ...],
    *,
    valid_at: str,
    ontology_version: str,
    allow_empty: bool = False,
) -> None:
    if not support and not allow_empty:
        raise MetamemoryAuthorityViolation("metamemory output requires ledger support")
    if any(
        item.startswith(("memory:", "slot:", "summary:", "policy:"))
        for item in support
    ):
        raise MetamemoryAuthorityViolation(
            "metamemory identifiers cannot become evidence"
        )
    usable = _usable_events(
        store,
        valid_at=valid_at,
        ontology_version=ontology_version,
    )
    if set(support) - set(usable):
        raise MetamemoryEvidenceViolation("metamemory support is absent or unusable")
    events = {event.event_id: event for event in store.events()}
    if support and not all(_reaches_observation(event_id, events) for event_id in support):
        raise MetamemoryAuthorityViolation("metamemory support lacks observed roots")


def _reaches_observation(event_id: str, events: Mapping[str, Any]) -> bool:
    pending = [event_id]
    seen: set[str] = set()
    while pending:
        current_id = pending.pop()
        if current_id in seen:
            continue
        seen.add(current_id)
        event = events.get(current_id)
        if event is None:
            return False
        if event.speech_act == SpeechAct.OBSERVED:
            return True
        pending.extend(event.parent_ids)
        pending.extend(
            witness.removeprefix("event:")
            for witness in event.provenance_witnesses
            if witness.startswith("event:")
        )
    return False


class DeterministicMetamemoryPolicy:
    """Registered heuristic baseline over explicit task views."""

    policy_id = "deterministic-metamemory-heuristic-v1"

    def __init__(self) -> None:
        self._operation_counts: dict[str, int] = {}
        self._flops = 0
        self._decision_bytes = 0
        self._artifact_bytes = 0
        self._skill_bytes = 0

    def decide(self, task: MetamemoryTask) -> MetamemoryDecision:
        task.validate()
        items = list(task.items)
        if task.query.get("target_item_id"):
            action = "retrieve"
        elif all(not item["verified"] and item["kind"] == "noise" for item in items):
            action = "ignore"
        elif any(not item["verified"] for item in items):
            action = "request_evidence"
        elif (
            len(items) >= MINIMUM_SKILL_REPETITIONS
            and all(item["kind"] == "procedure" for item in items)
            and len({item.get("procedure_key") for item in items}) == 1
        ):
            action = "stage"
        elif any(item["kind"] == "constraint" for item in items):
            action = "structure"
        elif len(items) >= 6 or any(
            item["kind"] in {"contradiction", "refutation"} for item in items
        ):
            action = "summarize"
        elif len(items) >= 2 and any(item["kind"] == "analytic" for item in items):
            action = "link"
        else:
            raise MetamemoryEvidenceViolation(
                "deterministic policy has no registered feature rule"
            )
        if action == "retrieve" and task.query.get("target_item_id"):
            selected = [
                item for item in items if item["item_id"] == task.query["target_item_id"]
            ]
        else:
            selected = items
        support = _support_for(selected)
        payload = {
            "policy_id": self.policy_id,
            "task_id": task.task_id,
            "action": action,
            "selected_item_ids": sorted(str(item["item_id"]) for item in selected),
            "supporting_event_ids": list(support),
        }
        decision = MetamemoryDecision(
            decision_id=f"policy:{sha256_json(payload)}",
            policy_id=self.policy_id,
            task_id=task.task_id,
            action=action,
            selected_item_ids=tuple(payload["selected_item_ids"]),
            supporting_event_ids=support,
            reason=f"registered-feature-rule:{action}",
        )
        self._operation_counts[action] = self._operation_counts.get(action, 0) + 1
        self._flops += (
            POLICY_BASE_FLOPS[action]
            + FLOPS_PER_CANONICAL_INPUT_BYTE * _canonical_size(task.to_dict())
            + FLOPS_PER_SCANNED_ITEM * len(task.items)
        )
        self._decision_bytes += _canonical_size(decision.to_dict())
        return decision

    def charge_artifact(self, artifact: CompressionArtifact) -> None:
        self._artifact_bytes += _canonical_size(artifact.to_dict())

    def charge_skill_candidate(self, candidate: SkillCandidate) -> None:
        self._skill_bytes += _canonical_size(candidate.to_dict())

    def compute_record(self) -> MetamemoryComputeRecord:
        return MetamemoryComputeRecord(
            operation_counts=dict(sorted(self._operation_counts.items())),
            realized_policy_flops=self._flops,
            serialized_decision_bytes=self._decision_bytes,
            serialized_artifact_bytes=self._artifact_bytes,
            serialized_skill_candidate_bytes=self._skill_bytes,
        )


def validate_decision(
    store: AppendOnlyEventStore,
    task: MetamemoryTask,
    decision: MetamemoryDecision,
    *,
    valid_at: str,
    ontology_version: str = "v1",
) -> None:
    task.validate()
    if (
        decision.action not in POLICY_ACTIONS
        or decision.action != task.target_action
        or decision.task_id != task.task_id
        or decision.authority != "candidate-only"
        or decision.direct_ledger_write_authorized
    ):
        raise MetamemoryAuthorityViolation("metamemory decision identity or authority drifted")
    items = _items(task)
    if not decision.selected_item_ids or set(decision.selected_item_ids) - set(items):
        raise MetamemoryEvidenceViolation("metamemory decision selects unknown items")
    selected = [items[item_id] for item_id in decision.selected_item_ids]
    expected_support = _support_for(selected)
    if decision.supporting_event_ids != expected_support:
        raise MetamemoryEvidenceViolation("metamemory decision support is incomplete")
    if decision.action == "ignore" and any(item["verified"] for item in selected):
        raise MetamemoryAuthorityViolation("ignore cannot discard verified evidence")
    if decision.action == "ignore":
        _validate_ledger_support(
            store,
            decision.supporting_event_ids,
            valid_at=valid_at,
            ontology_version=ontology_version,
            allow_empty=True,
        )
        return
    if decision.action == "request_evidence":
        if not any(item["unresolved"] or not item["verified"] for item in selected):
            raise MetamemoryEvidenceViolation("evidence request has no registered gap")
        _validate_ledger_support(
            store,
            decision.supporting_event_ids,
            valid_at=valid_at,
            ontology_version=ontology_version,
            allow_empty=True,
        )
        return
    if decision.action == "link" and len(selected) < 2:
        raise MetamemoryEvidenceViolation("link requires at least two explicit items")
    if decision.action == "summarize" and len(selected) < 2:
        raise MetamemoryEvidenceViolation("summarize requires at least two explicit items")
    if decision.action == "structure" and not any(
        item["kind"] == "constraint" or item["unresolved"] for item in selected
    ):
        raise MetamemoryEvidenceViolation("structure requires a constraint or unresolved item")
    if decision.action == "stage" and len(selected) < MINIMUM_SKILL_REPETITIONS:
        raise MetamemoryEvidenceViolation("stage requires repeated verified procedure items")
    if decision.action == "stage" and (
        any(item["kind"] != "procedure" or not item["verified"] for item in selected)
        or len({item.get("procedure_key") for item in selected}) != 1
        or selected[0].get("procedure_key") is None
    ):
        raise MetamemoryEvidenceViolation(
            "stage requires one repeated verified procedure identity"
        )
    _validate_ledger_support(
        store,
        decision.supporting_event_ids,
        valid_at=valid_at,
        ontology_version=ontology_version,
    )


def build_compression_artifact(
    store: AppendOnlyEventStore,
    task: MetamemoryTask,
    *,
    valid_at: str,
    ontology_version: str = "v1",
) -> CompressionArtifact:
    if len(task.items) > MAXIMUM_SOURCE_ITEMS:
        raise MetamemoryCapacityExceeded("compression source-item cap exceeded")
    items = [dict(item) for item in task.items]
    support = _support_for(items)
    _validate_ledger_support(
        store,
        support,
        valid_at=valid_at,
        ontology_version=ontology_version,
    )
    regions: dict[str, list[str]] = {}
    for item in items:
        regions.setdefault(str(item["region"]), []).append(str(item["item_id"]))
    summary_payload = [
        {"kind": item["kind"], "content": item["content"]} for item in items
    ]
    frontier = store.frontier()
    source_hash = sha256_json(items)
    artifact = CompressionArtifact(
        artifact_id=f"summary:{sha256_json({'task_id': task.task_id, 'source_hash': source_hash})}",
        task_id=task.task_id,
        source_item_ids=tuple(sorted(str(item["item_id"]) for item in items)),
        source_item_hash=source_hash,
        supporting_event_ids=support,
        region_manifest={
            key: tuple(sorted(value)) for key, value in sorted(regions.items())
        },
        contradiction_item_ids=tuple(
            sorted(str(item["item_id"]) for item in items if item["kind"] == "contradiction")
        ),
        refutation_item_ids=tuple(
            sorted(str(item["item_id"]) for item in items if item["kind"] == "refutation")
        ),
        temporal_qualifiers={
            str(item["item_id"]): str(item["temporal_qualifier"])
            for item in items
            if item.get("temporal_qualifier") is not None
        },
        unresolved_item_ids=tuple(
            sorted(str(item["item_id"]) for item in items if item["unresolved"])
        ),
        summary_text=f"advisory-explicit-view-summary:{sha256_json(summary_payload)}",
        summary_digest=sha256_json(summary_payload),
        ontology_version=ontology_version,
        ledger_frontier_sequence=frontier.sequence,
        ledger_frontier_hash=frontier.chain_hash,
    )
    if _canonical_size(artifact.to_dict()) > MAXIMUM_SERIALIZED_BYTES:
        raise MetamemoryCapacityExceeded("compression serialized-byte cap exceeded")
    return artifact


def validate_compression_artifact(
    store: AppendOnlyEventStore,
    task: MetamemoryTask,
    artifact: CompressionArtifact,
    *,
    valid_at: str,
    ontology_version: str = "v1",
) -> None:
    if artifact.authority != "candidate-only-advisory-view" or artifact.independent_evidence:
        raise MetamemoryAuthorityViolation("compression artifact gained authority")
    if artifact.ontology_version != ontology_version:
        raise StaleCompressionArtifact("compression ontology version drifted")
    frontier = store.frontier()
    if (
        artifact.ledger_frontier_sequence != frontier.sequence
        or artifact.ledger_frontier_hash != frontier.chain_hash
    ):
        raise StaleCompressionArtifact("compression ledger frontier is stale")
    if len(task.items) > MAXIMUM_SOURCE_ITEMS:
        raise MetamemoryCapacityExceeded("compression source-item cap exceeded")
    if _canonical_size(artifact.to_dict()) > MAXIMUM_SERIALIZED_BYTES:
        raise MetamemoryCapacityExceeded("compression serialized-byte cap exceeded")
    expected = build_compression_artifact(
        store,
        task,
        valid_at=valid_at,
        ontology_version=ontology_version,
    )
    protected_fields = (
        "task_id",
        "source_item_ids",
        "source_item_hash",
        "supporting_event_ids",
        "region_manifest",
        "contradiction_item_ids",
        "refutation_item_ids",
        "temporal_qualifiers",
        "unresolved_item_ids",
        "summary_digest",
    )
    for field in protected_fields:
        if getattr(artifact, field) != getattr(expected, field):
            raise MetamemoryCoverageViolation(
                f"compression protected field drifted: {field}"
            )
    _validate_ledger_support(
        store,
        artifact.supporting_event_ids,
        valid_at=valid_at,
        ontology_version=ontology_version,
    )


def build_skill_candidate(
    store: AppendOnlyEventStore,
    task: MetamemoryTask,
    *,
    valid_at: str,
    ontology_version: str = "v1",
) -> SkillCandidate:
    procedure_items = [
        dict(item)
        for item in task.items
        if item["kind"] == "procedure" and item["verified"]
    ]
    keys = {item.get("procedure_key") for item in procedure_items}
    if len(procedure_items) < MINIMUM_SKILL_REPETITIONS or len(keys) != 1 or None in keys:
        raise MetamemoryEvidenceViolation(
            "skill candidate requires one repeated verified procedure"
        )
    support = _support_for(procedure_items)
    _validate_ledger_support(
        store,
        support,
        valid_at=valid_at,
        ontology_version=ontology_version,
    )
    payload = {
        "task_id": task.task_id,
        "procedure_key": next(iter(keys)),
        "item_ids": sorted(item["item_id"] for item in procedure_items),
        "support": list(support),
    }
    return SkillCandidate(
        candidate_id=f"policy:skill-candidate:{sha256_json(payload)}",
        task_id=task.task_id,
        procedure_key=str(payload["procedure_key"]),
        repetition_item_ids=tuple(payload["item_ids"]),
        supporting_event_ids=support,
    )


def validate_skill_candidate(
    store: AppendOnlyEventStore,
    task: MetamemoryTask,
    candidate: SkillCandidate,
    *,
    valid_at: str,
    ontology_version: str = "v1",
) -> None:
    if candidate.authority != "candidate-only" or candidate.promotion_authority:
        raise MetamemoryAuthorityViolation("skill candidate gained promotion authority")
    expected = build_skill_candidate(
        store,
        task,
        valid_at=valid_at,
        ontology_version=ontology_version,
    )
    if candidate != expected:
        raise MetamemoryEvidenceViolation("skill candidate support or repetitions drifted")


def with_replaced_artifact(
    artifact: CompressionArtifact,
    **changes: Any,
) -> CompressionArtifact:
    """Explicit mutation helper used only by structural negative controls."""

    return replace(artifact, **changes)
