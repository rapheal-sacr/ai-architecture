"""Frozen standard-library helpers for the E0.16 native-memory run.

This module derives gate-supervision examples and schedules without importing
MLX or reading any accuracy metric.  It is safe to use in the standard suite
and binds every later neural operation to the E0.15 task registry.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable

from .canonical import sha256_json
from .native_memory_tasks import FAMILIES, NativeMemoryTask, generate_family


RUN_CONTRACT_SCHEMA_VERSION = 1
RUN_CONTRACT_ID = "wamrx-native-memory-run-v1"
RUNNER_VERSION = "wamrx-native-memory-comparison-v1"
ARM_IDS = (
    "explicit-multiview-v1",
    "deterministic-session-cache-v1",
    "minimal-gated-native-memory-v1",
)
OPERATION_ORDER = ("remember", "update", "merge", "forget")


class NativeMemoryRunError(ValueError):
    pass


@dataclass(frozen=True)
class GateExample:
    example_id: str
    task_id: str
    family: str
    split: str
    turn_index: int
    input: dict[str, Any]
    target_operation: str

    def to_dict(self) -> dict[str, Any]:
        if self.target_operation not in OPERATION_ORDER:
            raise NativeMemoryRunError("gate example has an unregistered target")
        return {
            "example_id": self.example_id,
            "task_id": self.task_id,
            "family": self.family,
            "split": self.split,
            "turn_index": self.turn_index,
            "input": self.input,
            "target_operation": self.target_operation,
        }


@dataclass(frozen=True)
class GateBatch:
    update: int
    example_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"update": self.update, "example_ids": list(self.example_ids)}


def _target_for_turn(
    task: NativeMemoryTask,
    turn: dict[str, Any],
    *,
    similar_variant_index: int,
) -> str | None:
    operation = turn["operation"]
    if operation == "remember":
        if (
            task.family == "similar_fact_interference"
            and "-variant-" in str(turn.get("key", ""))
        ):
            return "remember" if similar_variant_index == 0 else "merge"
        return "remember"
    if operation in {"update", "contradict"}:
        return "update"
    if operation == "reset":
        return "forget"
    return None


def gate_examples(tasks: Iterable[NativeMemoryTask]) -> tuple[GateExample, ...]:
    examples: list[GateExample] = []
    for task in sorted(tasks, key=lambda item: item.task_id):
        similar_variant_index = 0
        for turn_index, turn in enumerate(task.turns):
            target = _target_for_turn(
                task,
                turn,
                similar_variant_index=similar_variant_index,
            )
            if (
                task.family == "similar_fact_interference"
                and turn["operation"] == "remember"
                and "-variant-" in str(turn.get("key", ""))
            ):
                similar_variant_index += 1
            if target is None:
                continue
            examples.append(
                GateExample(
                    example_id=f"{task.task_id}:turn:{turn_index:03d}",
                    task_id=task.task_id,
                    family=task.family,
                    split=task.split,
                    turn_index=turn_index,
                    input={
                        "family": task.family,
                        "turn": dict(turn),
                        "query": dict(task.query),
                    },
                    target_operation=target,
                )
            )
    if len({example.example_id for example in examples}) != len(examples):
        raise NativeMemoryRunError("gate example identities collide")
    return tuple(examples)


def split_tasks(registry: dict[str, Any], split: str) -> tuple[NativeMemoryTask, ...]:
    if split not in {"train", "id", "ood"}:
        raise NativeMemoryRunError(f"unsupported split {split!r}")
    tasks = tuple(
        task
        for family in FAMILIES
        for task in generate_family(split, family, registry["splits"][split])
    )
    return tuple(sorted(tasks, key=lambda item: item.task_id))


def split_gate_examples(
    registry: dict[str, Any], split: str
) -> tuple[GateExample, ...]:
    return gate_examples(split_tasks(registry, split))


def build_gate_schedule(
    examples: tuple[GateExample, ...],
    *,
    seed: int,
    updates: int,
    examples_per_operation: int,
    schedule_seed_offset: int,
) -> tuple[GateBatch, ...]:
    if updates < 1 or examples_per_operation < 1:
        raise NativeMemoryRunError("gate schedule counts must be positive")
    by_operation: dict[str, list[str]] = defaultdict(list)
    for example in examples:
        by_operation[example.target_operation].append(example.example_id)
    if set(by_operation) != set(OPERATION_ORDER):
        raise NativeMemoryRunError("every registered gate operation needs examples")
    rng = random.Random(seed + schedule_seed_offset)
    cursor: dict[str, int] = {}
    for operation in OPERATION_ORDER:
        by_operation[operation].sort()
        rng.shuffle(by_operation[operation])
        cursor[operation] = 0
    batches = []
    for update in range(updates):
        ids: list[str] = []
        for operation in OPERATION_ORDER:
            for _ in range(examples_per_operation):
                if cursor[operation] >= len(by_operation[operation]):
                    rng.shuffle(by_operation[operation])
                    cursor[operation] = 0
                ids.append(by_operation[operation][cursor[operation]])
                cursor[operation] += 1
        rng.shuffle(ids)
        batches.append(GateBatch(update=update, example_ids=tuple(ids)))
    return tuple(batches)


def gate_schedule_hash(schedule: tuple[GateBatch, ...]) -> str:
    return sha256_json([batch.to_dict() for batch in schedule])


def validate_run_registration(root: Path) -> dict[str, Any]:
    contract_path = root / "contracts" / "wamrx_native_memory_run_v1.json"
    split_path = root / "contracts" / "wamrx_native_memory_splits_v1.json"
    contract = json.loads(contract_path.read_text())
    registry = json.loads(split_path.read_text())
    if contract.get("run_contract_schema_version") != RUN_CONTRACT_SCHEMA_VERSION:
        raise NativeMemoryRunError("unsupported native-memory run contract")
    if contract.get("contract_id") != RUN_CONTRACT_ID:
        raise NativeMemoryRunError("unexpected native-memory run contract identity")
    if tuple(contract.get("comparison_arms", ())) != ARM_IDS:
        raise NativeMemoryRunError("run contract comparison arms drifted")
    dependency_hashes: dict[str, str] = {}
    for dependency, registration in contract.get("depends_on", {}).items():
        if dependency == "selected_core":
            continue
        path = root / registration["path"]
        if not path.is_file():
            raise NativeMemoryRunError(f"registered dependency is missing: {dependency}")
        actual = sha256_json(json.loads(path.read_text()))
        if actual != registration["content_hash"]:
            raise NativeMemoryRunError(
                f"registered dependency hash drifted: {dependency}"
            )
        dependency_hashes[dependency] = actual
    selection = json.loads(
        (root / contract["depends_on"]["reasoner_selection"]["path"]).read_text()
    )
    selected = contract["depends_on"]["selected_core"]
    if (
        selection.get("selected_core", {}).get("arm_id") != selected["arm_id"]
        or int(selection.get("selected_core", {}).get("macro_depth", -1))
        != int(selected["macro_depth"])
        or selection.get("selected_from", {}).get("terminal_status")
        != "COMPLETE_RETAIN_FIXED"
    ):
        raise NativeMemoryRunError("selected fixed-depth core registration drifted")
    e0_15 = json.loads(
        (root / contract["depends_on"]["e0_15_structural_result"]["path"]).read_text()
    )
    if (
        e0_15.get("status") != "PASS"
        or e0_15.get("neural_comparison_status") != "NOT_RUN"
        or e0_15.get("accuracy_metrics_read") != 0
        or not all(e0_15.get("manipulations", {}).values())
    ):
        raise NativeMemoryRunError("E0.15 structural authorization drifted")
    gate = contract["learned_gate"]
    if tuple(gate["operation_order"]) != OPERATION_ORDER:
        raise NativeMemoryRunError("gate operation order drifted")
    if int(gate["parameter_count"]) != (472 * len(OPERATION_ORDER) + len(OPERATION_ORDER)):
        raise NativeMemoryRunError("gate parameter count is not the registered affine map")

    example_report: dict[str, Any] = {}
    for split in ("train", "id", "ood"):
        examples = split_gate_examples(registry, split)
        rows = [example.to_dict() for example in examples]
        expected = gate["frozen_example_registry"][split]
        counts = dict(Counter(example.target_operation for example in examples))
        if len(examples) != int(expected["count"]):
            raise NativeMemoryRunError(f"{split} gate example count drifted")
        if counts != expected["class_counts"]:
            raise NativeMemoryRunError(f"{split} gate class counts drifted")
        content_hash = sha256_json(rows)
        if content_hash != expected["content_hash"]:
            raise NativeMemoryRunError(f"{split} gate example hash drifted")
        example_report[split] = {
            "count": len(examples),
            "class_counts": counts,
            "content_hash": content_hash,
        }

    training = contract["gate_training"]
    schedules: dict[str, str] = {}
    train_examples = split_gate_examples(registry, "train")
    for seed in contract["paired_seeds"]:
        schedule = build_gate_schedule(
            train_examples,
            seed=int(seed),
            updates=int(training["optimizer_updates"]),
            examples_per_operation=int(training["examples_per_operation_per_batch"]),
            schedule_seed_offset=int(training["schedule_seed_offset"]),
        )
        actual = gate_schedule_hash(schedule)
        if actual != training["schedule_hashes"][str(seed)]:
            raise NativeMemoryRunError(f"gate schedule hash drifted for seed {seed}")
        if any(len(batch.example_ids) != int(training["batch_size"]) for batch in schedule):
            raise NativeMemoryRunError("gate schedule batch size drifted")
        schedules[str(seed)] = actual
    return {
        "contract_id": RUN_CONTRACT_ID,
        "runner_version": RUNNER_VERSION,
        "comparison_arms": list(ARM_IDS),
        "operation_order": list(OPERATION_ORDER),
        "gate_parameter_count": gate["parameter_count"],
        "examples": example_report,
        "schedule_hashes": schedules,
        "dependency_hashes": dependency_hashes,
        "accuracy_metrics_read": 0,
        "status": "PASS",
    }
