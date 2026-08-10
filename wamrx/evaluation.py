"""Regional adequacy and two-sided recompilation measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .resolver import resolve
from .retrieval import HybridRetrievalIndex
from .store import AppendOnlyEventStore


@dataclass(frozen=True)
class QueryProbe:
    probe_id: str
    region: str
    query: str
    expected_event_ids: tuple[str, ...]
    top_k: int = 1


@dataclass(frozen=True)
class CompileAdequacyReport:
    pooled_coverage: float
    regional_coverage: dict[str, float]
    pooled_query_distortion: float
    regional_query_distortion: dict[str, float]
    contradiction_preservation: float
    worst_region_coverage: float
    worst_region_query_distortion: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "pooled_coverage": self.pooled_coverage,
            "regional_coverage": self.regional_coverage,
            "pooled_query_distortion": self.pooled_query_distortion,
            "regional_query_distortion": self.regional_query_distortion,
            "contradiction_preservation": self.contradiction_preservation,
            "worst_region_coverage": self.worst_region_coverage,
            "worst_region_query_distortion": self.worst_region_query_distortion,
            "passed": self.passed,
        }


def measure_compile_adequacy(
    store: AppendOnlyEventStore,
    index: HybridRetrievalIndex,
    *,
    valid_at: str,
    protected_regions: tuple[str, ...],
    probes: tuple[QueryProbe, ...],
    minimum_regional_coverage: float,
    maximum_regional_query_distortion: float,
    require_contradiction_preservation: float = 1.0,
) -> CompileAdequacyReport:
    snapshot = resolve(store, valid_at=valid_at)
    live_by_region: dict[str, set[str]] = {region: set() for region in protected_regions}
    for record in snapshot.records:
        if record.usable and record.region in live_by_region and record.payload.get("text"):
            live_by_region[record.region].add(record.event_id)
    indexed = index.document_ids()
    regional_coverage = {
        region: (
            len(ids & indexed) / len(ids)
            if ids
            else 1.0
        )
        for region, ids in live_by_region.items()
    }
    all_live = set().union(*live_by_region.values()) if live_by_region else set()
    pooled_coverage = len(all_live & indexed) / len(all_live) if all_live else 1.0

    misses: dict[str, list[float]] = {region: [] for region in protected_regions}
    for probe in probes:
        decision_frontier = store.frontier().sequence
        result = index.search(
            probe.query,
            query_id=(
                f"adequacy:{index.envelope.stamp.artifact_id}:"
                f"{decision_frontier}:{probe.probe_id}"
            ),
            valid_at=valid_at,
            top_k=probe.top_k,
        )
        found = {hit.event_id for hit in result.hits}
        expected = set(probe.expected_event_ids)
        recall = len(found & expected) / len(expected) if expected else 1.0
        misses.setdefault(probe.region, []).append(1.0 - recall)
    regional_distortion = {
        region: sum(values) / len(values) if values else 0.0
        for region, values in misses.items()
    }
    all_misses = [value for values in misses.values() for value in values]
    pooled_distortion = sum(all_misses) / len(all_misses) if all_misses else 0.0

    expected_disabled = {
        record.event_id
        for record in snapshot.records
        if record.status in {"refuted", "retracted", "tombstone"}
    }
    preserved = set(index.disabled_records)
    contradiction_preservation = (
        len(expected_disabled & preserved) / len(expected_disabled)
        if expected_disabled
        else 1.0
    )
    worst_coverage = min(regional_coverage.values(), default=1.0)
    worst_distortion = max(regional_distortion.values(), default=0.0)
    passed = (
        worst_coverage >= minimum_regional_coverage
        and worst_distortion <= maximum_regional_query_distortion
        and contradiction_preservation >= require_contradiction_preservation
    )
    return CompileAdequacyReport(
        pooled_coverage=round(pooled_coverage, 12),
        regional_coverage={key: round(value, 12) for key, value in regional_coverage.items()},
        pooled_query_distortion=round(pooled_distortion, 12),
        regional_query_distortion={
            key: round(value, 12) for key, value in regional_distortion.items()
        },
        contradiction_preservation=round(contradiction_preservation, 12),
        worst_region_coverage=round(worst_coverage, 12),
        worst_region_query_distortion=round(worst_distortion, 12),
        passed=passed,
    )
