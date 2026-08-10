"""Provenance-linked typed temporal analytics compiled from ledger events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any

from .artifacts import (
    ArtifactCompatibilityPolicy,
    ArtifactEnvelope,
    ArtifactStamp,
    SupportManifest,
)
from .canonical import sha256_json
from .resolver import resolve
from .store import AppendOnlyEventStore

ANALYTIC_COMPILER_VERSION = "wamrx-analytic-v1"


class AnalyticCompileError(ValueError):
    pass


def _time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AnalyticCompileError(f"invalid analytic effective_at: {value!r}") from exc
    if parsed.tzinfo is None:
        raise AnalyticCompileError("analytic effective_at must include a timezone")
    return parsed


def _canonical_time(value: str) -> str:
    return _time(value).astimezone(timezone.utc).isoformat()


def _scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


@dataclass(frozen=True)
class CandidateField:
    value: Any
    confidence: float
    witness_event_ids: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any], event_id: str) -> "CandidateField":
        raw_confidence = value.get("confidence")
        if (
            isinstance(raw_confidence, bool)
            or not isinstance(raw_confidence, (int, float))
        ):
            raise AnalyticCompileError(
                "candidate-field confidence must be numeric"
            )
        confidence = float(raw_confidence)
        if not 0.0 <= confidence <= 1.0:
            raise AnalyticCompileError("candidate-field confidence must be in [0, 1]")
        candidate_value = value.get("value")
        if not _scalar(candidate_value):
            raise AnalyticCompileError("candidate-field values must be JSON scalars")
        raw_witnesses = value.get("witness_event_ids", (event_id,))
        if not isinstance(raw_witnesses, (list, tuple)) or any(
            not isinstance(witness, str) or not witness
            for witness in raw_witnesses
        ):
            raise AnalyticCompileError(
                "candidate-field witnesses must be non-empty event-ID strings"
            )
        witnesses = tuple(sorted(set(raw_witnesses)))
        if not witnesses:
            raise AnalyticCompileError("candidate fields require a witness event")
        return cls(
            value=candidate_value,
            confidence=confidence,
            witness_event_ids=witnesses,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "confidence": self.confidence,
            "witness_event_ids": list(self.witness_event_ids),
        }


@dataclass(frozen=True)
class AnalyticRow:
    row_id: str
    event_id: str
    record_type: str
    entity: str
    effective_at: str
    region: str
    dimensions: dict[str, str]
    measures: dict[str, float]
    fields: dict[str, Any]
    candidate_fields: dict[str, CandidateField]
    support: SupportManifest

    @classmethod
    def from_payload(
        cls,
        *,
        event_id: str,
        region: str,
        analytic: dict[str, Any],
        control_event_ids: tuple[str, ...],
    ) -> "AnalyticRow":
        record_type = str(analytic.get("record_type", "")).strip()
        entity = str(analytic.get("entity", "")).strip()
        effective_at = str(analytic.get("effective_at", "")).strip()
        if not record_type or not entity or not effective_at:
            raise AnalyticCompileError(
                f"{event_id}: analytic rows require record_type, entity, and effective_at"
            )
        effective_at = _canonical_time(effective_at)
        raw_dimensions = analytic.get("dimensions", {})
        if not isinstance(raw_dimensions, dict) or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            for key, value in raw_dimensions.items()
        ):
            raise AnalyticCompileError(
                f"{event_id}: dimensions must map non-empty strings to strings"
            )
        dimensions = dict(raw_dimensions)
        raw_measures = analytic.get("measures", {})
        if not isinstance(raw_measures, dict) or not raw_measures:
            raise AnalyticCompileError(f"{event_id}: analytic rows require measures")
        if any(
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for key, value in raw_measures.items()
        ):
            raise AnalyticCompileError(
                f"{event_id}: measures must map non-empty strings to finite numbers"
            )
        measures = {str(key): float(value) for key, value in raw_measures.items()}
        raw_fields = analytic.get("fields", {})
        if not isinstance(raw_fields, dict) or any(
            not isinstance(key, str)
            or not key
            or not _scalar(value)
            for key, value in raw_fields.items()
        ):
            raise AnalyticCompileError(
                f"{event_id}: fields must map non-empty strings to JSON scalars"
            )
        fields = dict(raw_fields)
        raw_candidates = analytic.get("candidate_fields", {})
        if not isinstance(raw_candidates, dict) or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, dict)
            for key, value in raw_candidates.items()
        ):
            raise AnalyticCompileError(
                f"{event_id}: candidate fields must be named objects"
            )
        candidates = {
            key: CandidateField.from_dict(value, event_id)
            for key, value in raw_candidates.items()
        }
        candidate_witnesses = {
            witness
            for candidate in candidates.values()
            for witness in candidate.witness_event_ids
        }
        row_id = str(analytic.get("row_id") or f"{record_type}:{entity}:{effective_at}:{event_id}")
        return cls(
            row_id=row_id,
            event_id=event_id,
            record_type=record_type,
            entity=entity,
            effective_at=effective_at,
            region=region,
            dimensions=dict(sorted(dimensions.items())),
            measures=dict(sorted(measures.items())),
            fields=dict(sorted(fields.items())),
            candidate_fields=dict(sorted(candidates.items())),
            support=SupportManifest.create(
                supporting_event_ids=[event_id],
                candidate_event_ids=[
                    event_id,
                    *control_event_ids,
                    *candidate_witnesses,
                ],
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "event_id": self.event_id,
            "record_type": self.record_type,
            "entity": self.entity,
            "effective_at": self.effective_at,
            "region": self.region,
            "dimensions": self.dimensions,
            "measures": self.measures,
            "fields": self.fields,
            "candidate_fields": {
                key: value.to_dict() for key, value in self.candidate_fields.items()
            },
            "support": self.support.to_dict(),
        }


@dataclass(frozen=True)
class AnalyticQueryResult:
    query_id: str
    operation: str
    value: Any
    rows: tuple[AnalyticRow, ...]
    support: SupportManifest
    journal_hash: str


class AnalyticMemory:
    def __init__(
        self,
        *,
        store: AppendOnlyEventStore,
        rows: tuple[AnalyticRow, ...],
        disabled_rows: dict[str, str],
        envelope: ArtifactEnvelope,
        compatibility_policy: ArtifactCompatibilityPolicy,
    ) -> None:
        self.store = store
        self.rows = rows
        self.disabled_rows = disabled_rows
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
    ) -> "AnalyticMemory":
        snapshot = resolve(store, valid_at=valid_at)
        rows = []
        disabled: dict[str, str] = {}
        all_candidates: set[str] = set()
        for record in snapshot.records:
            analytic = record.payload.get("analytic")
            if analytic is None:
                continue
            if not isinstance(analytic, dict):
                raise AnalyticCompileError(f"{record.event_id}: analytic payload must be an object")
            all_candidates.add(record.event_id)
            all_candidates.update(record.control_event_ids)
            if not record.usable:
                disabled[record.event_id] = record.status
                continue
            row = AnalyticRow.from_payload(
                event_id=record.event_id,
                region=record.region,
                analytic=analytic,
                control_event_ids=record.control_event_ids,
            )
            rows.append(row)
            all_candidates.update(row.support.candidate_event_ids)
        rows.sort(key=lambda row: (row.effective_at, row.row_id))
        content = {
            "compiler_version": ANALYTIC_COMPILER_VERSION,
            "valid_at": valid_at,
            "rows": [row.to_dict() for row in rows],
            "disabled_rows": dict(sorted(disabled.items())),
        }
        support_ids = [row.event_id for row in rows]
        contradiction_ids = sorted(
            all_candidates - set(support_ids) - set(disabled)
        )
        manifest = SupportManifest.create(
            supporting_event_ids=support_ids,
            contradicting_event_ids=contradiction_ids,
            candidate_event_ids=sorted(all_candidates),
            require_all_support=False,
            minimum_live_support=0,
        )
        stamp = ArtifactStamp.create(
            artifact_id=artifact_id,
            artifact_type="analytic-temporal-view",
            content=content,
            store=store,
            base_weight_version="none",
            component_versions={
                "analytic_compiler": ANALYTIC_COMPILER_VERSION,
                "resolver": snapshot.resolver_version,
            },
            ontology_version=ontology_version,
            verifier_version=verifier_version,
            build_config={"valid_at": valid_at, "strict_schema": True},
        )
        return cls(
            store=store,
            rows=tuple(rows),
            disabled_rows=disabled,
            envelope=ArtifactEnvelope(content=content, stamp=stamp, support=manifest),
            compatibility_policy=ArtifactCompatibilityPolicy.exact_for_stamp(stamp),
        )

    @property
    def content_hash(self) -> str:
        return self.envelope.stamp.content_hash

    def _live_rows(self, *, valid_at: str) -> tuple[AnalyticRow, ...]:
        self.envelope.validate(
            self.store,
            compatibility_policy=self.compatibility_policy,
            valid_at=valid_at,
            check_support=False,
        )
        usable = resolve(self.store, valid_at=valid_at).usable_event_ids()
        return tuple(row for row in self.rows if row.event_id in usable)

    def _select(
        self,
        *,
        valid_at: str,
        record_type: str | None = None,
        entity: str | None = None,
        dimensions: dict[str, str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> tuple[AnalyticRow, ...]:
        dimensions = dimensions or {}
        start_time = _time(start) if start else None
        end_time = _time(end) if end else None
        selected = []
        for row in self._live_rows(valid_at=valid_at):
            point = _time(row.effective_at)
            if record_type is not None and row.record_type != record_type:
                continue
            if entity is not None and row.entity != entity:
                continue
            if any(row.dimensions.get(key) != str(value) for key, value in dimensions.items()):
                continue
            if start_time is not None and point < start_time:
                continue
            if end_time is not None and point >= end_time:
                continue
            selected.append(row)
        return tuple(sorted(selected, key=lambda row: (row.effective_at, row.row_id)))

    def _result(
        self,
        *,
        query_id: str,
        operation: str,
        value: Any,
        rows: tuple[AnalyticRow, ...],
        valid_at: str,
        parameters: dict[str, Any],
    ) -> AnalyticQueryResult:
        snapshot = resolve(self.store, valid_at=valid_at)
        resolved = {record.event_id: record for record in snapshot.records}
        selected_ids = {row.row_id for row in rows}
        candidates = []
        for row in self.rows:
            record = resolved.get(row.event_id)
            candidates.append(
                {
                    "row_id": row.row_id,
                    "event_id": row.event_id,
                    "resolved_status": record.status if record else "missing",
                    "control_event_ids": (
                        list(record.control_event_ids) if record else []
                    ),
                    "selected": row.row_id in selected_ids,
                }
            )
        compiled_ids = {row.event_id for row in self.rows}
        for event_id, build_status in sorted(self.disabled_rows.items()):
            if event_id in compiled_ids:
                continue
            record = resolved.get(event_id)
            candidates.append(
                {
                    "row_id": f"disabled:{event_id}",
                    "event_id": event_id,
                    "resolved_status": record.status if record else build_status,
                    "control_event_ids": (
                        list(record.control_event_ids) if record else []
                    ),
                    "selected": False,
                }
            )
        manifest = SupportManifest.create(
            supporting_event_ids=[row.event_id for row in rows],
            candidate_event_ids=self.envelope.support.candidate_event_ids,
            require_all_support=True,
            minimum_live_support=len(rows),
        )
        record = {
            "query_id": query_id,
            "operation": operation,
            "valid_at": valid_at,
            "parameters": parameters,
            "artifact_id": self.envelope.stamp.artifact_id,
            "artifact_content_hash": self.content_hash,
            "artifact_frontier": {
                "sequence": self.envelope.stamp.ledger_frontier_sequence,
                "chain_hash": self.envelope.stamp.ledger_frontier_hash,
            },
            "decision_frontier": snapshot.frontier.to_dict(),
            "component_versions": self.envelope.stamp.component_versions,
            "ontology_version": self.envelope.stamp.ontology_version,
            "candidates": sorted(candidates, key=lambda item: item["row_id"]),
            "selected_row_ids": [row.row_id for row in rows],
            "support": manifest.to_dict(),
            "result": value,
            "result_hash": sha256_json(value),
        }
        journal_hash = self.store.append_analytic_query_record(record)
        return AnalyticQueryResult(
            query_id=query_id,
            operation=operation,
            value=value,
            rows=rows,
            support=manifest,
            journal_hash=journal_hash,
        )

    def filter(
        self,
        *,
        query_id: str,
        valid_at: str,
        record_type: str | None = None,
        entity: str | None = None,
        dimensions: dict[str, str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> AnalyticQueryResult:
        parameters = {
            "record_type": record_type,
            "entity": entity,
            "dimensions": dimensions or {},
            "start": start,
            "end": end,
        }
        rows = self._select(valid_at=valid_at, **parameters)
        return self._result(
            query_id=query_id,
            operation="filter",
            value=[row.row_id for row in rows],
            rows=rows,
            valid_at=valid_at,
            parameters=parameters,
        )

    @staticmethod
    def _aggregate_value(values: list[float], operation: str) -> float | int:
        operations = {
            "sum": lambda: sum(values),
            "count": lambda: len(values),
            "mean": lambda: sum(values) / len(values),
            "min": lambda: min(values),
            "max": lambda: max(values),
        }
        if operation not in operations:
            raise ValueError(f"unsupported aggregation {operation!r}")
        return operations[operation]()

    def aggregate(
        self,
        measure: str,
        *,
        query_id: str,
        operation: str,
        valid_at: str,
        **filters: Any,
    ) -> AnalyticQueryResult:
        rows = tuple(
            row
            for row in self._select(valid_at=valid_at, **filters)
            if measure in row.measures
        )
        values = [row.measures[measure] for row in rows]
        if not values:
            raise KeyError(f"no live rows contain measure {measure!r}")
        parameters = {"measure": measure, **filters}
        return self._result(
            query_id=query_id,
            operation=operation,
            value=self._aggregate_value(values, operation),
            rows=rows,
            valid_at=valid_at,
            parameters=parameters,
        )

    def group_aggregate(
        self,
        measure: str,
        *,
        query_id: str,
        group_by: str,
        operation: str,
        valid_at: str,
        **filters: Any,
    ) -> AnalyticQueryResult:
        rows = tuple(
            row
            for row in self._select(valid_at=valid_at, **filters)
            if measure in row.measures and group_by in row.dimensions
        )
        groups: dict[str, list[float]] = {}
        for row in rows:
            groups.setdefault(row.dimensions[group_by], []).append(row.measures[measure])
        if not groups:
            raise KeyError(f"no rows contain measure {measure!r} and dimension {group_by!r}")
        if operation == "sum":
            value = {key: sum(items) for key, items in groups.items()}
        elif operation == "mean":
            value = {key: sum(items) / len(items) for key, items in groups.items()}
        elif operation == "count":
            value = {key: len(items) for key, items in groups.items()}
        else:
            raise ValueError(f"unsupported grouped aggregation {operation!r}")
        return self._result(
            query_id=query_id,
            operation=f"group_{operation}:{group_by}",
            value=dict(sorted(value.items())),
            rows=rows,
            valid_at=valid_at,
            parameters={"measure": measure, "group_by": group_by, **filters},
        )

    def compare_windows(
        self,
        measure: str,
        *,
        query_id: str,
        valid_at: str,
        left: tuple[str, str],
        right: tuple[str, str],
        operation: str = "sum",
        **filters: Any,
    ) -> AnalyticQueryResult:
        left_rows = tuple(
            row
            for row in self._select(
                valid_at=valid_at,
                start=left[0],
                end=left[1],
                **filters,
            )
            if measure in row.measures
        )
        right_rows = tuple(
            row
            for row in self._select(
                valid_at=valid_at,
                start=right[0],
                end=right[1],
                **filters,
            )
            if measure in row.measures
        )
        if not left_rows or not right_rows:
            raise KeyError(f"both windows must contain measure {measure!r}")
        left_value = self._aggregate_value(
            [row.measures[measure] for row in left_rows], operation
        )
        right_value = self._aggregate_value(
            [row.measures[measure] for row in right_rows], operation
        )
        rows = tuple((*left_rows, *right_rows))
        value = {
            "left": left_value,
            "right": right_value,
            "delta": right_value - left_value,
        }
        return self._result(
            query_id=query_id,
            operation=f"compare_{operation}",
            value=value,
            rows=rows,
            valid_at=valid_at,
            parameters={
                "measure": measure,
                "left": left,
                "right": right,
                **filters,
            },
        )

    def trend(
        self,
        measure: str,
        *,
        query_id: str,
        valid_at: str,
        **filters: Any,
    ) -> AnalyticQueryResult:
        rows = tuple(
            row
            for row in self._select(valid_at=valid_at, **filters)
            if measure in row.measures
        )
        if len(rows) < 2:
            raise ValueError("trend requires at least two live rows")
        first, last = rows[0].measures[measure], rows[-1].measures[measure]
        delta = last - first
        value = {
            "first": first,
            "last": last,
            "delta": delta,
            "direction": "up" if delta > 0 else "down" if delta < 0 else "flat",
        }
        return self._result(
            query_id=query_id,
            operation="trend",
            value=value,
            rows=rows,
            valid_at=valid_at,
            parameters={"measure": measure, **filters},
        )

    def rank_groups(
        self,
        measure: str,
        *,
        query_id: str,
        group_by: str,
        valid_at: str,
        descending: bool = True,
        **filters: Any,
    ) -> AnalyticQueryResult:
        rows = tuple(
            row
            for row in self._select(valid_at=valid_at, **filters)
            if measure in row.measures and group_by in row.dimensions
        )
        groups: dict[str, float] = {}
        for row in rows:
            group = row.dimensions[group_by]
            groups[group] = groups.get(group, 0.0) + row.measures[measure]
        ranking = sorted(
            groups.items(),
            key=lambda item: ((-item[1]) if descending else item[1], item[0]),
        )
        return self._result(
            query_id=query_id,
            operation=f"rank:{group_by}",
            value=ranking,
            rows=rows,
            valid_at=valid_at,
            parameters={
                "measure": measure,
                "group_by": group_by,
                "descending": descending,
                **filters,
            },
        )
