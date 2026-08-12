"""Deterministic task registry for explicit-view metamemory policy work."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

from .canonical import sha256_json


TASK_GENERATOR_VERSION = "wamrx-metamemory-tasks-v1"
SPLIT_REGISTRY_SCHEMA_VERSION = 1
FAMILIES = (
    "distractor_admission",
    "evidence_gap",
    "cross_view_linkage",
    "targeted_retrieval",
    "redundant_history_compression",
    "constraint_structuring",
    "procedure_repetition",
)
FAMILY_ACTION = {
    "distractor_admission": "ignore",
    "evidence_gap": "request_evidence",
    "cross_view_linkage": "link",
    "targeted_retrieval": "retrieve",
    "redundant_history_compression": "summarize",
    "constraint_structuring": "structure",
    "procedure_repetition": "stage",
}


class MetamemoryTaskError(ValueError):
    pass


@dataclass(frozen=True)
class MetamemoryTask:
    task_id: str
    family: str
    split: str
    generalization_axis: str
    items: tuple[dict[str, Any], ...]
    query: dict[str, Any]
    target_action: str
    protected_regions: tuple[str, ...]
    generator_version: str = TASK_GENERATOR_VERSION

    def validate(self) -> None:
        if self.family not in FAMILIES or self.target_action != FAMILY_ACTION[self.family]:
            raise MetamemoryTaskError("task family/action identity drifted")
        if self.split not in {"train", "id", "ood"}:
            raise MetamemoryTaskError("metamemory task split is invalid")
        if not self.task_id or not self.items or not self.query or not self.protected_regions:
            raise MetamemoryTaskError("metamemory task is incomplete")
        required = {"item_id", "kind", "content", "region", "verified", "event_id"}
        if any(set(item) < required for item in self.items):
            raise MetamemoryTaskError("metamemory item is incomplete")
        if len({item["item_id"] for item in self.items}) != len(self.items):
            raise MetamemoryTaskError("metamemory item IDs collide")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "task_id": self.task_id,
            "family": self.family,
            "split": self.split,
            "generalization_axis": self.generalization_axis,
            "items": list(self.items),
            "query": self.query,
            "target_action": self.target_action,
            "protected_regions": list(self.protected_regions),
            "generator_version": self.generator_version,
        }


def _item(
    task_id: str,
    index: int,
    *,
    kind: str,
    content: str,
    region: str,
    verified: bool = True,
    temporal_qualifier: str | None = None,
    unresolved: bool = False,
    procedure_key: str | None = None,
) -> dict[str, Any]:
    return {
        "item_id": f"{task_id}:item:{index:03d}",
        "kind": kind,
        "content": content,
        "region": region,
        "verified": verified,
        "event_id": f"{task_id}:event:{index:03d}",
        "temporal_qualifier": temporal_qualifier,
        "unresolved": unresolved,
        "procedure_key": procedure_key,
    }


def _task(
    family: str,
    split: str,
    index: int,
    rng: random.Random,
    axis: str,
) -> MetamemoryTask:
    task_id = f"metamemory/{family}/{split}/{index:04d}"
    region = rng.choice(("operations", "finance", "rare-protected"))
    count = int(axis.split(":", 1)[1])
    items: list[dict[str, Any]] = []
    query = {"entity": f"entity-{rng.randrange(100, 999)}", "intent": family}

    if family == "distractor_admission":
        items = [
            _item(
                task_id,
                item_index,
                kind="noise",
                content=f"unverified-noise-{rng.randrange(1000, 9999)}",
                region=region,
                verified=False,
            )
            for item_index in range(count)
        ]
    elif family == "evidence_gap":
        items = [
            _item(
                task_id,
                0,
                kind="claim",
                content=f"unsupported-claim-{rng.randrange(1000, 9999)}",
                region=region,
                verified=False,
                unresolved=True,
            )
        ]
    elif family == "cross_view_linkage":
        items = [
            _item(
                task_id,
                item_index,
                kind="observation" if item_index == 0 else "analytic",
                content=f"linked-{query['entity']}-{item_index}",
                region=region,
            )
            for item_index in range(count)
        ]
    elif family == "targeted_retrieval":
        items = [
            _item(
                task_id,
                item_index,
                kind="observation",
                content=(
                    f"target-{query['entity']}"
                    if item_index == count - 1
                    else f"other-{rng.randrange(1000, 9999)}"
                ),
                region=region if item_index == count - 1 else "operations",
            )
            for item_index in range(count)
        ]
        query["target_item_id"] = items[-1]["item_id"]
    elif family == "redundant_history_compression":
        base = max(count, 6)
        items = [
            _item(
                task_id,
                item_index,
                kind=(
                    "contradiction"
                    if item_index == 1
                    else "refutation"
                    if item_index == 2
                    else "unresolved"
                    if item_index == 3
                    else "observation"
                ),
                content=f"history-{rng.randrange(1000, 9999)}",
                region=("rare-protected" if item_index == 0 else region),
                temporal_qualifier=f"valid-window-{item_index}" if item_index in {0, 4} else None,
                unresolved=item_index == 3,
            )
            for item_index in range(base)
        ]
    elif family == "constraint_structuring":
        items = [
            _item(
                task_id,
                item_index,
                kind="constraint" if item_index == count - 1 else "observation",
                content=f"constraint-node-{item_index}",
                region=region,
                unresolved=item_index == count - 1,
            )
            for item_index in range(count)
        ]
    elif family == "procedure_repetition":
        procedure_key = f"procedure-{rng.randrange(100, 999)}"
        items = [
            _item(
                task_id,
                item_index,
                kind="procedure",
                content=f"{procedure_key}:step:{item_index}",
                region=region,
                procedure_key=procedure_key,
            )
            for item_index in range(count)
        ]
    else:  # pragma: no cover - family table and generator are locked together.
        raise MetamemoryTaskError(f"unsupported family {family!r}")

    task = MetamemoryTask(
        task_id=task_id,
        family=family,
        split=split,
        generalization_axis=axis,
        items=tuple(items),
        query=query,
        target_action=FAMILY_ACTION[family],
        protected_regions=tuple(sorted({item["region"] for item in items})),
    )
    task.validate()
    return task


def load_split_registry(path: str | Path) -> dict[str, Any]:
    registry = json.loads(Path(path).read_text())
    if registry.get("split_registry_schema_version") != SPLIT_REGISTRY_SCHEMA_VERSION:
        raise MetamemoryTaskError("unsupported metamemory split schema")
    if registry.get("generator_version") != TASK_GENERATOR_VERSION:
        raise MetamemoryTaskError("metamemory task generator version drifted")
    if set(registry.get("splits", {})) != {"train", "id", "ood"}:
        raise MetamemoryTaskError("metamemory registry requires train, id, and ood")
    return registry


def generate_family(
    split: str,
    family: str,
    split_spec: dict[str, Any],
) -> tuple[MetamemoryTask, ...]:
    if family not in FAMILIES:
        raise MetamemoryTaskError(f"unknown metamemory family {family!r}")
    rng = random.Random(int(split_spec["family_seeds"][family]))
    axes = tuple(split_spec["family_axes"][family])
    count = int(split_spec["count_per_family"])
    tasks = tuple(
        _task(family, split, index, rng, str(rng.choice(axes)))
        for index in range(count)
    )
    if len({task.task_id for task in tasks}) != len(tasks):
        raise MetamemoryTaskError("generated metamemory task IDs collide")
    return tasks


def generated_hashes(registry: dict[str, Any]) -> dict[str, dict[str, str]]:
    report = {}
    for split, spec in registry["splits"].items():
        hashes = {
            family: sha256_json(
                [task.to_dict() for task in generate_family(split, family, spec)]
            )
            for family in FAMILIES
        }
        report[split] = {**hashes, "aggregate": sha256_json(hashes)}
    return report


def verify_frozen_splits(registry: dict[str, Any]) -> dict[str, Any]:
    actual = generated_hashes(registry)
    mismatches = {}
    axes = {family: {} for family in FAMILIES}
    for split, spec in registry["splits"].items():
        for key, actual_hash in actual[split].items():
            expected = spec["expected_hashes"].get(key)
            if expected != actual_hash:
                mismatches[f"{split}:{key}"] = {
                    "expected": expected,
                    "actual": actual_hash,
                }
        for family in FAMILIES:
            axes[family][split] = sorted(
                {
                    task.generalization_axis
                    for task in generate_family(split, family, spec)
                }
            )
    disjoint = {
        family: set(axes[family]["train"]).isdisjoint(axes[family]["ood"])
        for family in FAMILIES
    }
    return {
        "registry_id": registry["registry_id"],
        "generator_version": registry["generator_version"],
        "hashes": actual,
        "hash_mismatches": mismatches,
        "axes": axes,
        "ood_axis_disjoint": disjoint,
        "passed": not mismatches and all(disjoint.values()),
    }
