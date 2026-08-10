"""Deterministic frozen task generators for the recurrent-reasoner assay."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

from .canonical import sha256_json

TASK_GENERATOR_VERSION = "wamrx-recurrent-tasks-v1"
SPLIT_REGISTRY_SCHEMA_VERSION = 1
FAMILIES = ("algorithmic", "structured", "multiview")


class RecurrentTaskError(ValueError):
    pass


@dataclass(frozen=True)
class RecurrentTask:
    task_id: str
    family: str
    split: str
    generalization_axis: str
    required_reasoning_steps: int
    problem: dict[str, Any]
    expected: Any
    protected_regions: tuple[str, ...]
    generator_version: str = TASK_GENERATOR_VERSION

    def validate(self) -> None:
        if self.family not in FAMILIES:
            raise RecurrentTaskError(f"unknown recurrent task family {self.family!r}")
        if self.split not in {"train", "id", "ood"}:
            raise RecurrentTaskError(f"unknown split {self.split!r}")
        if not self.task_id or not self.generalization_axis:
            raise RecurrentTaskError("task ID and generalization axis are required")
        if self.required_reasoning_steps < 1:
            raise RecurrentTaskError("required reasoning steps must be positive")
        if not self.problem or not self.protected_regions:
            raise RecurrentTaskError("problem and protected regions are required")
        bundle = self.problem.get("evidence_bundle")
        if not isinstance(bundle, dict) or not bundle.get("views"):
            raise RecurrentTaskError("every task requires a stamped evidence bundle")
        for view in bundle["views"]:
            required = {
                "view_type",
                "artifact_id",
                "content_hash",
                "ontology_version",
                "regions",
            }
            if not isinstance(view, dict) or not required <= set(view):
                raise RecurrentTaskError("task evidence views are incomplete")

    @property
    def problem_hash(self) -> str:
        return sha256_json(self.problem)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "task_id": self.task_id,
            "family": self.family,
            "split": self.split,
            "generalization_axis": self.generalization_axis,
            "required_reasoning_steps": self.required_reasoning_steps,
            "problem": self.problem,
            "problem_hash": self.problem_hash,
            "expected": self.expected,
            "protected_regions": list(self.protected_regions),
            "generator_version": self.generator_version,
        }


def _view(
    task_id: str,
    view_type: str,
    content: Any,
    regions: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "view_type": view_type,
        "artifact_id": f"{task_id}:{view_type}:v1",
        "content_hash": sha256_json(content),
        "ontology_version": "v1",
        "component_versions": {f"{view_type}_compiler": "v1"},
        "ledger_frontier_sequence": 1,
        "ledger_frontier_hash": sha256_json({"task_id": task_id, "view": view_type}),
        "support_event_ids": [f"evidence:{task_id}:{view_type}"],
        "regions": list(regions),
    }


def _algorithmic_task(
    split: str,
    index: int,
    rng: random.Random,
    parameters: dict[str, Any],
) -> RecurrentTask:
    length = int(rng.choice(parameters["lengths"]))
    values = [rng.randrange(10) for _ in range(length)]
    state = 0
    intermediate = []
    for value in values:
        state = (state * 3 + value) % 17
        intermediate.append(state)
    task_id = f"algorithmic/{split}/{index:04d}"
    evidence = {"sequence": values, "modulus": 17, "multiplier": 3}
    problem = {
        "instruction": "Apply state=(state*3+value) mod 17 from state zero.",
        "sequence": values,
        "evidence_bundle": {
            "views": [_view(task_id, "retrieval", evidence, ("algorithmic",))]
        },
        "intermediate_targets": intermediate,
    }
    return RecurrentTask(
        task_id=task_id,
        family="algorithmic",
        split=split,
        generalization_axis=f"length:{length}",
        required_reasoning_steps=max(1, length // 2),
        problem=problem,
        expected=state,
        protected_regions=("algorithmic",),
    )


def _graph_edges(
    structure: str,
    node_count: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    if structure == "chain":
        edges.update((node, node + 1) for node in range(node_count - 1))
        for _ in range(max(1, node_count // 3)):
            left = rng.randrange(node_count - 2)
            right = rng.randrange(left + 2, node_count)
            edges.add((left, right))
    elif structure == "binary_tree":
        edges.update(((node - 1) // 2, node) for node in range(1, node_count))
    elif structure == "ladder":
        width = node_count // 2
        edges.update((node, node + 1) for node in range(width - 1))
        edges.update(
            (width + node, width + node + 1) for node in range(width - 1)
        )
        edges.update((node, width + node) for node in range(width))
    elif structure == "two_cluster_bridge":
        split = node_count // 2
        edges.update((node, node + 1) for node in range(split - 1))
        edges.update((node, node + 1) for node in range(split, node_count - 1))
        edges.add((split - 1, split))
        for start in (0, split):
            stop = split if start == 0 else node_count
            for _ in range(max(1, (stop - start) // 4)):
                left = rng.randrange(start, stop - 1)
                right = rng.randrange(left + 1, stop)
                edges.add((left, right))
    else:
        raise RecurrentTaskError(f"unsupported graph structure {structure!r}")
    return sorted(edges)


def _shortest_path(
    node_count: int,
    edges: list[tuple[int, int]],
    source: int,
    target: int,
) -> tuple[int, list[int]]:
    adjacency: dict[int, list[int]] = {node: [] for node in range(node_count)}
    for left, right in edges:
        adjacency[left].append(right)
    pending = deque([(source, [source])])
    seen = {source}
    while pending:
        node, path = pending.popleft()
        if node == target:
            return len(path) - 1, path
        for neighbour in sorted(adjacency[node]):
            if neighbour not in seen:
                seen.add(neighbour)
                pending.append((neighbour, [*path, neighbour]))
    return -1, []


def _structured_task(
    split: str,
    index: int,
    rng: random.Random,
    parameters: dict[str, Any],
) -> RecurrentTask:
    node_count = int(rng.choice(parameters["node_counts"]))
    structure = str(rng.choice(parameters["structures"]))
    edges = _graph_edges(structure, node_count, rng)
    distance, path = _shortest_path(node_count, edges, 0, node_count - 1)
    task_id = f"structured/{split}/{index:04d}"
    evidence = {
        "node_count": node_count,
        "directed_edges": [list(edge) for edge in edges],
    }
    problem = {
        "instruction": "Return the shortest directed path length from 0 to the final node.",
        **evidence,
        "evidence_bundle": {
            "views": [_view(task_id, "graph", evidence, ("structured",))]
        },
        "intermediate_target_path": path,
    }
    return RecurrentTask(
        task_id=task_id,
        family="structured",
        split=split,
        generalization_axis=f"structure:{structure}",
        required_reasoning_steps=max(1, (distance + 1) // 2),
        problem=problem,
        expected=distance,
        protected_regions=("structured",),
    )


def _aggregate_payload(rng: random.Random) -> tuple[dict[str, Any], float]:
    records = []
    for index in range(rng.randint(4, 7)):
        records.append(
            {
                "record_id": f"expense-{index}",
                "amount": float(rng.randrange(20, 200)),
                "status": "active" if index != 1 else "retracted",
            }
        )
    total = sum(record["amount"] for record in records if record["status"] == "active")
    return {"records": records, "operation": "sum_active"}, total


def _contradiction_payload(rng: random.Random) -> tuple[dict[str, Any], str]:
    old, current = rng.sample(
        ["Montreal", "Toronto", "Vancouver", "Halifax"],
        2,
    )
    claims = [
        {"value": old, "status": "refuted"},
        {"value": current, "status": "verified"},
    ]
    return {"claims": claims, "operation": "current_verified_value"}, current


def _constraint_payload(rng: random.Random) -> tuple[dict[str, Any], list[str]]:
    eligible_cost = rng.randrange(60, 95)
    candidates = [
        {
            "candidate": "candidate-a",
            "security": True,
            "latency": rng.randrange(60, 95),
            "cost": 120,
        },
        {
            "candidate": "candidate-b",
            "security": False,
            "latency": rng.randrange(40, 90),
            "cost": 80,
        },
        {
            "candidate": "candidate-c",
            "security": True,
            "latency": rng.randrange(60, 99),
            "cost": eligible_cost,
        },
    ]
    requirements = {"security": True, "maximum_latency": 100, "maximum_cost": 100}
    return {
        "candidates": candidates,
        "requirements": requirements,
        "operation": "select_satisfying_all_constraints",
    }, ["candidate-c"]


def _aggregate_then_constraint_payload(
    rng: random.Random,
) -> tuple[dict[str, Any], list[str]]:
    candidates = []
    for index, name in enumerate(("candidate-a", "candidate-b", "candidate-c")):
        monthly = [rng.randrange(20, 55) for _ in range(3)]
        candidates.append(
            {
                "candidate": name,
                "monthly_costs": monthly,
                "security": index != 1,
            }
        )
    limit = sum(candidates[2]["monthly_costs"])
    candidates[0]["monthly_costs"] = [limit, limit, limit]
    eligible = [
        item["candidate"]
        for item in candidates
        if item["security"] and sum(item["monthly_costs"]) <= limit
    ]
    return {
        "candidates": candidates,
        "maximum_total_cost": limit,
        "operation": "aggregate_each_candidate_then_apply_security_and_cost_constraints",
    }, eligible


def _contradiction_then_aggregate_payload(
    rng: random.Random,
) -> tuple[dict[str, Any], float]:
    old, current = rng.sample(["compute", "travel", "storage"], 2)
    claims = [
        {"category": old, "status": "refuted"},
        {"category": current, "status": "verified"},
    ]
    records = [
        {
            "category": category,
            "amount": float(rng.randrange(20, 100)),
            "status": "active",
        }
        for category in (old, current, current, "other")
    ]
    total = sum(
        row["amount"]
        for row in records
        if row["category"] == current and row["status"] == "active"
    )
    return {
        "category_claims": claims,
        "records": records,
        "operation": "resolve_current_category_then_sum_matching_active_records",
    }, total


def _multiview_task(
    split: str,
    index: int,
    rng: random.Random,
    parameters: dict[str, Any],
) -> RecurrentTask:
    composition = str(rng.choice(parameters["compositions"]))
    builders = {
        "aggregate": _aggregate_payload,
        "contradiction": _contradiction_payload,
        "constraint": _constraint_payload,
        "aggregate_then_constraint": _aggregate_then_constraint_payload,
        "contradiction_then_aggregate": _contradiction_then_aggregate_payload,
    }
    payload, expected = builders[composition](rng)
    task_id = f"multiview/{split}/{index:04d}"
    analytic_content = {
        key: value for key, value in payload.items() if key not in {"claims", "category_claims"}
    }
    graph_content = {
        key: value for key, value in payload.items() if key in {"claims", "category_claims", "candidates", "requirements"}
    }
    views = [
        _view(task_id, "analytic", analytic_content, ("finance",)),
        _view(task_id, "graph", graph_content, ("operations",)),
    ]
    problem = {
        "instruction": payload["operation"],
        "evidence": payload,
        "evidence_bundle": {"views": views},
    }
    return RecurrentTask(
        task_id=task_id,
        family="multiview",
        split=split,
        generalization_axis=f"composition:{composition}",
        required_reasoning_steps=4 if "then" in composition else 2,
        problem=problem,
        expected=expected,
        protected_regions=("finance", "operations"),
    )


GENERATORS = {
    "algorithmic": _algorithmic_task,
    "structured": _structured_task,
    "multiview": _multiview_task,
}


def load_split_registry(path: str | Path) -> dict[str, Any]:
    registry = json.loads(Path(path).read_text())
    if registry.get("split_registry_schema_version") != SPLIT_REGISTRY_SCHEMA_VERSION:
        raise RecurrentTaskError("unsupported recurrent split registry schema")
    if registry.get("generator_version") != TASK_GENERATOR_VERSION:
        raise RecurrentTaskError("split registry names a different generator version")
    splits = registry.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "id", "ood"}:
        raise RecurrentTaskError("split registry must define train, id, and ood")
    return registry


def generate_family(
    split: str,
    family: str,
    split_spec: dict[str, Any],
) -> tuple[RecurrentTask, ...]:
    if family not in GENERATORS:
        raise RecurrentTaskError(f"unknown family {family!r}")
    count = int(split_spec["count_per_family"])
    rng = random.Random(int(split_spec["family_seeds"][family]))
    parameters = dict(split_spec["family_parameters"][family])
    tasks = tuple(
        GENERATORS[family](split, index, rng, parameters)
        for index in range(count)
    )
    if len({task.task_id for task in tasks}) != len(tasks):
        raise RecurrentTaskError("generated task IDs are not unique")
    for task in tasks:
        task.validate()
    return tasks


def generated_hashes(registry: dict[str, Any]) -> dict[str, dict[str, str]]:
    reports: dict[str, dict[str, str]] = {}
    for split, split_spec in registry["splits"].items():
        family_hashes = {
            family: sha256_json(
                [task.to_dict() for task in generate_family(split, family, split_spec)]
            )
            for family in FAMILIES
        }
        reports[split] = {
            **family_hashes,
            "aggregate": sha256_json(family_hashes),
        }
    return reports


def verify_frozen_splits(registry: dict[str, Any]) -> dict[str, Any]:
    actual = generated_hashes(registry)
    mismatches = {}
    axes: dict[str, dict[str, list[str]]] = {family: {} for family in FAMILIES}
    max_steps: dict[str, dict[str, int]] = {family: {} for family in FAMILIES}
    for split, split_spec in registry["splits"].items():
        expected = split_spec["expected_hashes"]
        for key, actual_hash in actual[split].items():
            if expected.get(key) != actual_hash:
                mismatches[f"{split}:{key}"] = {
                    "expected": expected.get(key),
                    "actual": actual_hash,
                }
        for family in FAMILIES:
            tasks = generate_family(split, family, split_spec)
            axes[family][split] = sorted({task.generalization_axis for task in tasks})
            max_steps[family][split] = max(
                task.required_reasoning_steps for task in tasks
            )
    ood_axis_disjoint = {
        family: set(axes[family]["train"]).isdisjoint(axes[family]["ood"])
        for family in FAMILIES
    }
    return {
        "registry_id": registry["registry_id"],
        "generator_version": registry["generator_version"],
        "hashes": actual,
        "hash_mismatches": mismatches,
        "axes": axes,
        "ood_axis_disjoint": ood_axis_disjoint,
        "maximum_required_reasoning_steps": max_steps,
        "passed": not mismatches and all(ood_axis_disjoint.values()),
    }
