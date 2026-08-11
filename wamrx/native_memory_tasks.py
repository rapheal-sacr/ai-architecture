"""Deterministic task registry for the future native-memory comparison."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

from .canonical import sha256_json


TASK_GENERATOR_VERSION = "wamrx-native-memory-tasks-v1"
SPLIT_REGISTRY_SCHEMA_VERSION = 1
FAMILIES = (
    "delayed_recall",
    "correction",
    "temporal_update",
    "distractor_session",
    "similar_fact_interference",
    "context_overflow",
    "reset_task_switch",
    "rare_region_retention",
    "poison_contradiction",
)


class NativeMemoryTaskError(ValueError):
    pass


@dataclass(frozen=True)
class NativeMemoryTask:
    task_id: str
    family: str
    split: str
    generalization_axis: str
    turns: tuple[dict[str, Any], ...]
    query: dict[str, Any]
    expected: Any
    protected_regions: tuple[str, ...]
    generator_version: str = TASK_GENERATOR_VERSION

    def validate(self) -> None:
        if self.family not in FAMILIES:
            raise NativeMemoryTaskError(f"unknown native-memory family {self.family!r}")
        if self.split not in {"train", "id", "ood"}:
            raise NativeMemoryTaskError(f"unknown split {self.split!r}")
        if not self.task_id or not self.generalization_axis or not self.turns:
            raise NativeMemoryTaskError("task identity, axis, and turns are required")
        if not self.query or not self.protected_regions:
            raise NativeMemoryTaskError("query and protected regions are required")
        if any("evidence_id" not in turn or "operation" not in turn for turn in self.turns):
            raise NativeMemoryTaskError("every turn requires an operation and evidence ID")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "task_id": self.task_id,
            "family": self.family,
            "split": self.split,
            "generalization_axis": self.generalization_axis,
            "turns": list(self.turns),
            "query": self.query,
            "expected": self.expected,
            "protected_regions": list(self.protected_regions),
            "generator_version": self.generator_version,
        }


def _value(rng: random.Random, prefix: str) -> str:
    return f"{prefix}-{rng.randrange(1000, 9999)}"


def _task(
    family: str,
    split: str,
    index: int,
    rng: random.Random,
    axis: str,
) -> NativeMemoryTask:
    task_id = f"native-memory/{family}/{split}/{index:04d}"
    key = f"entity-{rng.randrange(100, 999)}"
    first = _value(rng, "value")
    turns: list[dict[str, Any]] = [
        {"operation": "remember", "key": key, "value": first, "evidence_id": f"{task_id}:e0"}
    ]
    expected: Any = first
    regions = ("operations",)

    if family == "delayed_recall":
        delay = int(axis.split(":", 1)[1])
        turns.extend(
            {
                "operation": "distractor",
                "key": f"delay-{step}",
                "value": _value(rng, "noise"),
                "evidence_id": f"{task_id}:d{step}",
            }
            for step in range(delay)
        )
    elif family == "correction":
        corrected = _value(rng, "corrected")
        turns.append(
            {"operation": "update", "key": key, "value": corrected, "evidence_id": f"{task_id}:e1"}
        )
        expected = corrected
    elif family == "temporal_update":
        updates = int(axis.split(":", 1)[1])
        for step in range(updates):
            expected = _value(rng, "temporal")
            turns.append(
                {"operation": "update", "key": key, "value": expected, "evidence_id": f"{task_id}:u{step}"}
            )
    elif family == "distractor_session":
        count = int(axis.split(":", 1)[1])
        turns.extend(
            {
                "operation": "distractor",
                "key": f"other-{step}",
                "value": _value(rng, "noise"),
                "evidence_id": f"{task_id}:d{step}",
            }
            for step in range(count)
        )
    elif family == "similar_fact_interference":
        count = int(axis.split(":", 1)[1])
        turns.extend(
            {
                "operation": "remember",
                "key": f"{key}-variant-{step}",
                "value": _value(rng, "similar"),
                "evidence_id": f"{task_id}:s{step}",
            }
            for step in range(count)
        )
    elif family == "context_overflow":
        length = int(axis.split(":", 1)[1])
        turns.extend(
            {
                "operation": "distractor",
                "key": f"context-{step}",
                "value": _value(rng, "token"),
                "evidence_id": f"{task_id}:c{step}",
            }
            for step in range(length)
        )
    elif family == "reset_task_switch":
        turns.extend(
            [
                {"operation": "reset", "key": key, "value": None, "evidence_id": f"{task_id}:reset"},
                {"operation": "query-old-task", "key": key, "value": None, "evidence_id": f"{task_id}:q"},
            ]
        )
        expected = None
    elif family == "rare_region_retention":
        regions = ("rare-protected",)
        turns.extend(
            {
                "operation": "decay-pressure",
                "key": f"common-{step}",
                "value": _value(rng, "common"),
                "evidence_id": f"{task_id}:r{step}",
            }
            for step in range(int(axis.split(":", 1)[1]))
        )
    elif family == "poison_contradiction":
        turns.extend(
            [
                {"operation": "poison", "key": key, "value": _value(rng, "poison"), "evidence_id": f"{task_id}:poison"},
                {"operation": "contradict", "key": key, "value": first, "evidence_id": f"{task_id}:verified"},
            ]
        )
        regions = ("finance", "operations")
    else:  # pragma: no cover - FAMILIES and generator table are locked together.
        raise NativeMemoryTaskError(f"unsupported family {family!r}")

    return NativeMemoryTask(
        task_id=task_id,
        family=family,
        split=split,
        generalization_axis=axis,
        turns=tuple(turns),
        query={"operation": "recall", "key": key},
        expected=expected,
        protected_regions=regions,
    )


def load_split_registry(path: str | Path) -> dict[str, Any]:
    registry = json.loads(Path(path).read_text())
    if registry.get("split_registry_schema_version") != SPLIT_REGISTRY_SCHEMA_VERSION:
        raise NativeMemoryTaskError("unsupported native-memory split schema")
    if registry.get("generator_version") != TASK_GENERATOR_VERSION:
        raise NativeMemoryTaskError("split registry names a different generator")
    if set(registry.get("splits", {})) != {"train", "id", "ood"}:
        raise NativeMemoryTaskError("native-memory registry requires train, id, and ood")
    return registry


def generate_family(
    split: str,
    family: str,
    split_spec: dict[str, Any],
) -> tuple[NativeMemoryTask, ...]:
    if family not in FAMILIES:
        raise NativeMemoryTaskError(f"unknown family {family!r}")
    rng = random.Random(int(split_spec["family_seeds"][family]))
    axes = tuple(split_spec["family_axes"][family])
    count = int(split_spec["count_per_family"])
    tasks = tuple(_task(family, split, index, rng, str(rng.choice(axes))) for index in range(count))
    if len({task.task_id for task in tasks}) != len(tasks):
        raise NativeMemoryTaskError("generated native-memory task IDs collide")
    for task in tasks:
        task.validate()
    return tasks


def generated_hashes(registry: dict[str, Any]) -> dict[str, dict[str, str]]:
    report: dict[str, dict[str, str]] = {}
    for split, spec in registry["splits"].items():
        family_hashes = {
            family: sha256_json([task.to_dict() for task in generate_family(split, family, spec)])
            for family in FAMILIES
        }
        report[split] = {**family_hashes, "aggregate": sha256_json(family_hashes)}
    return report


def verify_frozen_splits(registry: dict[str, Any]) -> dict[str, Any]:
    actual = generated_hashes(registry)
    mismatches: dict[str, dict[str, str | None]] = {}
    axes: dict[str, dict[str, list[str]]] = {family: {} for family in FAMILIES}
    for split, spec in registry["splits"].items():
        expected = spec["expected_hashes"]
        for key, actual_hash in actual[split].items():
            if expected.get(key) != actual_hash:
                mismatches[f"{split}:{key}"] = {
                    "expected": expected.get(key),
                    "actual": actual_hash,
                }
        for family in FAMILIES:
            axes[family][split] = sorted(
                {task.generalization_axis for task in generate_family(split, family, spec)}
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
