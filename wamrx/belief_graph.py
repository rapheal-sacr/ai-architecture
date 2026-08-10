"""Evidence-addressable belief, contradiction, and constraint graph."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import math
from typing import Any

from .artifacts import (
    ArtifactCompatibilityPolicy,
    ArtifactEnvelope,
    ArtifactStamp,
    SupportManifest,
)
from .canonical import canonical_json
from .resolver import resolve
from .store import AppendOnlyEventStore

GRAPH_COMPILER_VERSION = "wamrx-belief-graph-v1"
ACTIVE_STATUSES = frozenset({"observed", "asserted", "inferred", "verified"})


class BeliefGraphCompileError(ValueError):
    pass


def _claim_scalar(value: Any) -> bool:
    return (
        value is None
        or isinstance(value, (str, bool, int))
        or (isinstance(value, float) and math.isfinite(value))
    )


@dataclass(frozen=True)
class ClaimEdge:
    edge_id: str
    event_id: str
    subject: str
    relation: str
    object: Any
    conditions: tuple[str, ...]
    region: str
    status: str
    support: SupportManifest

    @property
    def active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "event_id": self.event_id,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "conditions": list(self.conditions),
            "region": self.region,
            "status": self.status,
            "support": self.support.to_dict(),
        }


@dataclass(frozen=True)
class ContradictionSet:
    subject: str
    relation: str
    conditions: tuple[str, ...]
    status: str
    edges: tuple[ClaimEdge, ...]


@dataclass(frozen=True)
class ConstraintRequirement:
    relation: str
    operator: str
    expected: Any = None
    next_step: str | None = None

    def __post_init__(self) -> None:
        allowed = {"equals", "not_equals", "lte", "gte", "lt", "gt", "exists"}
        if not self.relation or self.operator not in allowed:
            raise ValueError("constraint requirement has an invalid relation or operator")


@dataclass(frozen=True)
class RequirementResult:
    requirement: ConstraintRequirement
    status: str
    matching_edges: tuple[ClaimEdge, ...]
    conflicting_edges: tuple[ClaimEdge, ...]


@dataclass(frozen=True)
class ConstraintEvaluation:
    subject: str
    status: str
    requirements: tuple[RequirementResult, ...]
    missing_evidence: tuple[str, ...]
    violated_constraints: tuple[str, ...]
    conflicting_constraints: tuple[str, ...]
    next_steps: tuple[str, ...]
    support: SupportManifest


@dataclass(frozen=True)
class ConstraintSelection:
    eligible_subjects: tuple[str, ...]
    unresolved_subjects: tuple[str, ...]
    rejected_subjects: tuple[str, ...]
    evaluations: tuple[ConstraintEvaluation, ...]


def _matches(value: Any, requirement: ConstraintRequirement) -> bool:
    if requirement.operator == "exists":
        return True
    if requirement.operator == "equals":
        return value == requirement.expected
    if requirement.operator == "not_equals":
        return value != requirement.expected
    try:
        if requirement.operator == "lte":
            return value <= requirement.expected
        if requirement.operator == "gte":
            return value >= requirement.expected
        if requirement.operator == "lt":
            return value < requirement.expected
        if requirement.operator == "gt":
            return value > requirement.expected
    except TypeError:
        return False
    raise ValueError(requirement.operator)


class BeliefGraph:
    def __init__(
        self,
        *,
        store: AppendOnlyEventStore,
        edges: tuple[ClaimEdge, ...],
        envelope: ArtifactEnvelope,
        compatibility_policy: ArtifactCompatibilityPolicy,
    ) -> None:
        self.store = store
        self.edges = edges
        self.envelope = envelope
        self.compatibility_policy = compatibility_policy

    @classmethod
    def build(
        cls,
        store: AppendOnlyEventStore,
        *,
        artifact_id: str,
        valid_at: str,
        ontology_version: str = "v1",
        verifier_version: str = "executable-v1",
    ) -> "BeliefGraph":
        snapshot = resolve(store, valid_at=valid_at)
        edges = []
        edge_ids: set[str] = set()
        all_candidates: set[str] = set()
        contradiction_ids: set[str] = set()
        for record in snapshot.records:
            claim = record.payload.get("claim")
            if claim is None:
                continue
            if not isinstance(claim, dict):
                raise BeliefGraphCompileError(
                    f"{record.event_id}: claim payload must be an object"
                )
            subject = claim.get("subject")
            relation = claim.get("relation")
            if (
                not isinstance(subject, str)
                or not subject.strip()
                or not isinstance(relation, str)
                or not relation.strip()
                or "object" not in claim
                or not _claim_scalar(claim.get("object"))
            ):
                raise BeliefGraphCompileError(
                    f"{record.event_id}: claims require string subject/relation "
                    "and a finite JSON-scalar object"
                )
            subject = subject.strip()
            relation = relation.strip()
            raw_conditions = claim.get("conditions", ())
            if not isinstance(raw_conditions, (list, tuple)) or any(
                not isinstance(condition, str) or not condition.strip()
                for condition in raw_conditions
            ):
                raise BeliefGraphCompileError(
                    f"{record.event_id}: claim conditions must be strings"
                )
            conditions = tuple(
                sorted(set(condition.strip() for condition in raw_conditions))
            )
            raw_edge_id = claim.get("edge_id", record.event_id)
            if not isinstance(raw_edge_id, str) or not raw_edge_id.strip():
                raise BeliefGraphCompileError(
                    f"{record.event_id}: edge_id must be a non-empty string"
                )
            edge_id = raw_edge_id.strip()
            if edge_id in edge_ids:
                raise BeliefGraphCompileError(
                    f"{record.event_id}: duplicate graph edge ID {edge_id!r}"
                )
            edge_ids.add(edge_id)
            support = SupportManifest.create(
                supporting_event_ids=[record.event_id],
                candidate_event_ids=[record.event_id, *record.control_event_ids],
            )
            edge = ClaimEdge(
                edge_id=edge_id,
                event_id=record.event_id,
                subject=subject,
                relation=relation,
                object=claim["object"],
                conditions=conditions,
                region=record.region,
                status=record.status,
                support=support,
            )
            edges.append(edge)
            all_candidates.update(support.candidate_event_ids)
            if not edge.active:
                contradiction_ids.update(record.control_event_ids)
        edges.sort(key=lambda edge: (edge.subject, edge.relation, edge.edge_id))
        content = {
            "compiler_version": GRAPH_COMPILER_VERSION,
            "valid_at": valid_at,
            "edges": [edge.to_dict() for edge in edges],
        }
        supporting = [edge.event_id for edge in edges if edge.active]
        manifest = SupportManifest.create(
            supporting_event_ids=supporting,
            contradicting_event_ids=sorted(contradiction_ids),
            candidate_event_ids=sorted(all_candidates),
            require_all_support=False,
            minimum_live_support=0,
        )
        stamp = ArtifactStamp.create(
            artifact_id=artifact_id,
            artifact_type="belief-constraint-graph",
            content=content,
            store=store,
            base_weight_version="none",
            component_versions={
                "graph_compiler": GRAPH_COMPILER_VERSION,
                "resolver": snapshot.resolver_version,
            },
            ontology_version=ontology_version,
            verifier_version=verifier_version,
            build_config={"valid_at": valid_at, "preserve_inactive_claims": True},
        )
        return cls(
            store=store,
            edges=tuple(edges),
            envelope=ArtifactEnvelope(content=content, stamp=stamp, support=manifest),
            compatibility_policy=ArtifactCompatibilityPolicy.exact_for_stamp(stamp),
        )

    @property
    def content_hash(self) -> str:
        return self.envelope.stamp.content_hash

    def current_edges(self, *, valid_at: str) -> tuple[ClaimEdge, ...]:
        self.envelope.validate(
            self.store,
            compatibility_policy=self.compatibility_policy,
            valid_at=valid_at,
            check_support=False,
        )
        snapshot = resolve(self.store, valid_at=valid_at)
        status_by_id = {record.event_id: record.status for record in snapshot.records}
        return tuple(
            dataclasses.replace(edge, status=status_by_id.get(edge.event_id, "missing"))
            for edge in self.edges
        )

    def claims_for(
        self,
        *,
        valid_at: str,
        subject: str | None = None,
        relation: str | None = None,
        active_only: bool = True,
    ) -> tuple[ClaimEdge, ...]:
        return tuple(
            edge
            for edge in self.current_edges(valid_at=valid_at)
            if (subject is None or edge.subject == subject)
            and (relation is None or edge.relation == relation)
            and (not active_only or edge.active)
        )

    def entities(self, *, valid_at: str, active_only: bool = True) -> tuple[str, ...]:
        edges = self.claims_for(valid_at=valid_at, active_only=active_only)
        values = {edge.subject for edge in edges}
        values.update(edge.object for edge in edges if isinstance(edge.object, str))
        return tuple(sorted(values))

    def contradictions(self, *, valid_at: str) -> tuple[ContradictionSet, ...]:
        groups: dict[tuple[str, str, tuple[str, ...]], list[ClaimEdge]] = {}
        for edge in self.current_edges(valid_at=valid_at):
            key = (edge.subject, edge.relation, edge.conditions)
            groups.setdefault(key, []).append(edge)
        results = []
        for (subject, relation, conditions), edges in groups.items():
            objects = {canonical_json(edge.object) for edge in edges}
            if len(objects) < 2:
                continue
            active_objects = {canonical_json(edge.object) for edge in edges if edge.active}
            if len(active_objects) > 1:
                status = "unresolved"
            elif len(active_objects) == 1:
                status = "resolved"
            else:
                status = "retired"
            results.append(
                ContradictionSet(
                    subject=subject,
                    relation=relation,
                    conditions=conditions,
                    status=status,
                    edges=tuple(edges),
                )
            )
        return tuple(sorted(results, key=lambda item: (item.subject, item.relation)))

    def evaluate(
        self,
        subject: str,
        requirements: tuple[ConstraintRequirement, ...],
        *,
        valid_at: str,
    ) -> ConstraintEvaluation:
        subject_edges = self.claims_for(
            valid_at=valid_at, subject=subject, active_only=True
        )
        results = []
        missing = []
        violated = []
        conflicts = []
        next_steps = []
        used_edges: dict[str, ClaimEdge] = {}
        for requirement in requirements:
            candidates = tuple(
                edge for edge in subject_edges if edge.relation == requirement.relation
            )
            matching = tuple(edge for edge in candidates if _matches(edge.object, requirement))
            nonmatching = tuple(edge for edge in candidates if edge not in matching)
            if not candidates:
                status = "missing"
                missing.append(requirement.relation)
                next_steps.append(
                    requirement.next_step
                    or f"obtain evidence for {subject}.{requirement.relation}"
                )
            elif matching and nonmatching:
                status = "conflict"
                conflicts.append(requirement.relation)
                used_edges.update((edge.event_id, edge) for edge in candidates)
                next_steps.append(
                    requirement.next_step
                    or f"resolve conflicting evidence for {subject}.{requirement.relation}"
                )
            elif matching:
                status = "satisfied"
                used_edges.update((edge.event_id, edge) for edge in matching)
            else:
                status = "violated"
                violated.append(requirement.relation)
                used_edges.update((edge.event_id, edge) for edge in nonmatching)
            results.append(
                RequirementResult(
                    requirement=requirement,
                    status=status,
                    matching_edges=matching,
                    conflicting_edges=nonmatching,
                )
            )
        if violated:
            overall = "rejected"
        elif missing or conflicts:
            overall = "unresolved"
        else:
            overall = "satisfied"
        all_candidates = {
            event_id
            for edge in self.edges
            for event_id in edge.support.candidate_event_ids
        }
        return ConstraintEvaluation(
            subject=subject,
            status=overall,
            requirements=tuple(results),
            missing_evidence=tuple(sorted(missing)),
            violated_constraints=tuple(sorted(violated)),
            conflicting_constraints=tuple(sorted(conflicts)),
            next_steps=tuple(next_steps),
            support=SupportManifest.create(
                supporting_event_ids=sorted(used_edges),
                candidate_event_ids=sorted(all_candidates),
                require_all_support=True,
                minimum_live_support=len(used_edges),
            ),
        )

    def select_subjects(
        self,
        subjects: tuple[str, ...],
        requirements: tuple[ConstraintRequirement, ...],
        *,
        valid_at: str,
    ) -> ConstraintSelection:
        evaluations = tuple(
            self.evaluate(subject, requirements, valid_at=valid_at)
            for subject in subjects
        )
        return ConstraintSelection(
            eligible_subjects=tuple(
                item.subject for item in evaluations if item.status == "satisfied"
            ),
            unresolved_subjects=tuple(
                item.subject for item in evaluations if item.status == "unresolved"
            ),
            rejected_subjects=tuple(
                item.subject for item in evaluations if item.status == "rejected"
            ),
            evaluations=evaluations,
        )
