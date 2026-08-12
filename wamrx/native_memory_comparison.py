"""Complete inert-by-default E0.16 native-memory comparison implementation.

Only :func:`run_registered_comparison` consumes the frozen budget.  Importing
this module never loads checkpoints, trains the gate, or reads accuracy.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any, Mapping

import mlx.core as mx

from .canonical import canonical_json, sha256_json
from .events import Event, SpeechAct
from .native_memory import (
    CapacityExceeded,
    MemoryEvidenceBundle,
    NativeMemoryConfig,
    NativeMemoryError,
    SessionMemory,
)
from .native_memory_model import (
    GATE_FORWARD_FLOPS_PER_EXAMPLE,
    GATE_PARAMETER_COUNT,
    GateCheckpointMetadata,
    MinimalOperationGate,
    NativeMemoryModelError,
    create_gate_optimizer,
    feature_maps,
    gate_state_hash,
    load_frozen_core,
    load_gate_checkpoint,
    model_parameter_hash,
    save_gate_checkpoint,
    train_gate_segment,
)
from .native_memory_run import (
    ARM_IDS,
    OPERATION_ORDER,
    RUNNER_VERSION,
    GateExample,
    build_gate_schedule,
    gate_examples,
    gate_schedule_hash,
    split_gate_examples,
    split_tasks,
    validate_run_registration,
)
from .native_memory_tasks import NativeMemoryTask
from .recurrent_checkpoint import file_sha256
from .recurrent_model import ByteCodec, RecurrentReasoner, decode_batch
from .recurrent_run import holm_adjusted_tests, paired_student_t
from .store import AppendOnlyEventStore


RUN_ID = "e0.16-registered-native-memory-comparison-v1"
VALID_AT = "2026-08-12T12:00:00+00:00"
NOW = "2026-08-12T12:05:00+00:00"
EXPIRES = "2026-08-12T13:00:00+00:00"


class NativeMemoryComparisonError(ValueError):
    pass


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".json",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_hash(path: Path) -> str:
    return sha256_json(json.loads(path.read_text()))


def _implementation_hash(root: Path) -> str:
    paths = (
        "wamrx/canonical.py",
        "wamrx/events.py",
        "wamrx/grounding.py",
        "wamrx/artifacts.py",
        "wamrx/store.py",
        "wamrx/resolver.py",
        "wamrx/native_memory.py",
        "wamrx/native_memory_tasks.py",
        "wamrx/native_memory_run.py",
        "wamrx/native_memory_model.py",
        "wamrx/native_memory_comparison.py",
        "wamrx/recurrent_checkpoint.py",
        "wamrx/recurrent.py",
        "wamrx/recurrent_model.py",
        "wamrx/recurrent_model_constants.py",
        "wamrx/recurrent_run.py",
        "wamrx/recurrent_tasks.py",
        "wamrx/recurrent_training.py",
        "rig_a/experiments/e0_16_native_memory_comparison.py",
    )
    return sha256_json({path: file_sha256(root / path) for path in paths})


def build_run_manifest(root: Path) -> dict[str, Any]:
    return {
        "runner_version": RUNNER_VERSION,
        "implementation_hash": _implementation_hash(root),
        "e0_15_result_hash": _json_hash(
            root / "results" / "e0_15_native_memory_boundary.json"
        ),
        "reasoner_selection_hash": _json_hash(
            root / "contracts" / "wamrx_reasoner_selection_v1.json"
        ),
        "run_contract_hash": _json_hash(
            root / "contracts" / "wamrx_native_memory_run_v1.json"
        ),
        "native_memory_contract_hash": _json_hash(
            root / "contracts" / "wamrx_native_memory_v1.json"
        ),
        "split_registry_hash": _json_hash(
            root / "contracts" / "wamrx_native_memory_splits_v1.json"
        ),
        "core_checkpoint_manifest_hash": _json_hash(
            root / "contracts" / "wamrx_native_memory_core_checkpoints_v1.json"
        ),
    }


def _prelude_flops(model: RecurrentReasoner) -> int:
    config = model.config
    dimensions = config.hidden_dimensions
    return int(
        config.maximum_input_tokens * dimensions * 2
        + config.vocabulary_size * dimensions * 2
        + 2 * dimensions * dimensions
    )


def _gate_metadata(
    manifest: dict[str, Any],
    *,
    seed: int,
    update: int,
    batch_size: int,
    schedule_hash: str,
    core_file_sha256: str,
) -> GateCheckpointMetadata:
    return GateCheckpointMetadata(
        seed=seed,
        completed_updates=update,
        examples_seen=update * batch_size,
        schedule_hash=schedule_hash,
        run_contract_hash=manifest["run_contract_hash"],
        core_file_sha256=core_file_sha256,
    )


def train_gate_seed(
    root: Path,
    run_directory: Path,
    *,
    seed: int,
    manifest: dict[str, Any],
    resume: bool,
) -> tuple[RecurrentReasoner, MinimalOperationGate, dict[str, Any]]:
    contract = json.loads(
        (root / "contracts" / "wamrx_native_memory_run_v1.json").read_text()
    )
    registry = json.loads(
        (root / "contracts" / "wamrx_native_memory_splits_v1.json").read_text()
    )
    training = contract["gate_training"]
    examples = split_gate_examples(registry, "train")
    schedule = build_gate_schedule(
        examples,
        seed=seed,
        updates=int(training["optimizer_updates"]),
        examples_per_operation=int(training["examples_per_operation_per_batch"]),
        schedule_seed_offset=int(training["schedule_seed_offset"]),
    )
    frozen_schedule_hash = gate_schedule_hash(schedule)
    if frozen_schedule_hash != training["schedule_hashes"][str(seed)]:
        raise NativeMemoryComparisonError("gate schedule differs from registration")

    core, core_identity = load_frozen_core(root, seed)
    core_hash_before = model_parameter_hash(core)
    codec = ByteCodec(
        maximum_input_tokens=core.config.maximum_input_tokens,
        maximum_output_tokens=core.config.maximum_output_tokens,
    )
    feature_by_id, target_by_id = feature_maps(core, codec, examples)
    mx.random.seed(seed + int(training["gate_initialization_seed_offset"]))
    gate = MinimalOperationGate(core.config.hidden_dimensions)
    optimizer = create_gate_optimizer(contract)

    checkpoint_directory = run_directory / "gate-checkpoints" / f"seed-{seed}"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    training_state_path = checkpoint_directory / "training-state.json"
    candidates = sorted(checkpoint_directory.glob("update-*.safetensors"))
    if candidates and not resume:
        raise NativeMemoryComparisonError(
            "gate checkpoints already exist; use --resume or a new run directory"
        )
    start = 0
    segments: list[dict[str, Any]] = []
    restored_report = None
    if resume and candidates:
        latest = candidates[-1]
        try:
            start = int(latest.stem.removeprefix("update-"))
        except ValueError as error:
            raise NativeMemoryComparisonError("invalid gate checkpoint filename") from error
        expected = _gate_metadata(
            manifest,
            seed=seed,
            update=start,
            batch_size=int(training["batch_size"]),
            schedule_hash=frozen_schedule_hash,
            core_file_sha256=core_identity.file_sha256,
        )
        restored_report = load_gate_checkpoint(
            latest, gate, optimizer, expected=expected
        )
        if not training_state_path.is_file():
            raise NativeMemoryComparisonError("resume gate checkpoint lacks training state")
        prior = json.loads(training_state_path.read_text())
        if (
            prior.get("manifest") != manifest
            or prior.get("seed") != seed
            or prior.get("completed_updates") != start
        ):
            raise NativeMemoryComparisonError("gate training state identity mismatch")
        if prior.get("checkpoint") != restored_report:
            raise NativeMemoryComparisonError("gate checkpoint file or state hash drifted")
        segments = list(prior.get("segments", ()))

    frequency = int(training["checkpoint_frequency_updates"])
    final_checkpoint = restored_report
    for stop in range(start + frequency, len(schedule) + frequency, frequency):
        stop = min(stop, len(schedule))
        if stop <= start:
            continue
        segment = train_gate_segment(
            gate,
            optimizer,
            feature_by_id=feature_by_id,
            target_by_id=target_by_id,
            schedule=schedule,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            start_update=start,
            stop_update=stop,
        )
        segments.append(segment)
        metadata = _gate_metadata(
            manifest,
            seed=seed,
            update=stop,
            batch_size=int(training["batch_size"]),
            schedule_hash=frozen_schedule_hash,
            core_file_sha256=core_identity.file_sha256,
        )
        final_checkpoint = save_gate_checkpoint(
            checkpoint_directory / f"update-{stop:04d}.safetensors",
            gate,
            optimizer,
            metadata,
        )
        _atomic_json(
            training_state_path,
            {
                "manifest": manifest,
                "seed": seed,
                "completed_updates": stop,
                "segments": segments,
                "checkpoint": final_checkpoint,
            },
        )
        start = stop
        if stop == len(schedule):
            break
    if start != len(schedule) or final_checkpoint is None:
        raise NativeMemoryComparisonError("gate training did not reach its final update")
    core_hash_after = model_parameter_hash(core)
    if core_hash_before != core_hash_after or core_hash_before != core_identity.model_parameter_hash:
        raise NativeMemoryComparisonError("gate training mutated the frozen core")
    return core, gate, {
        "seed": seed,
        "schedule_hash": frozen_schedule_hash,
        "optimizer_updates": len(schedule),
        "examples_seen": len(schedule) * int(training["batch_size"]),
        "feature_examples": len(examples),
        "feature_core_flops": len(examples) * _prelude_flops(core),
        "gate_training_flops": sum(
            int(segment["realized_gate_training_flops"]) for segment in segments
        ),
        "segments": segments,
        "checkpoint": final_checkpoint,
        "gate_state_hash": gate_state_hash(gate, optimizer),
        "core_identity": core_identity.to_dict(),
        "core_hash_before": core_hash_before,
        "core_hash_after": core_hash_after,
    }


def _observed_turn_event(turn: dict[str, Any], index: int) -> Event:
    transaction = datetime(2026, 8, 12, 8, 0, tzinfo=timezone.utc) + timedelta(
        seconds=index
    )
    verified = turn["operation"] != "poison"
    return Event.create(
        event_id=str(turn["evidence_id"]),
        transaction_time=transaction.isoformat(),
        valid_from="2026-08-12T00:00:00+00:00",
        actor="native-memory-task-world",
        source_id=f"source:{turn['evidence_id']}",
        verifier_id="verifier:native-memory-task" if verified else None,
        modality="structured-text",
        speech_act=SpeechAct.OBSERVED,
        payload={
            "operation": turn["operation"],
            "key": turn.get("key"),
            "value": turn.get("value"),
        },
        confidence=1.0 if verified else None,
        verifier_class="executable" if verified else "unverified",
        provenance_witnesses=(f"external:task:{turn['evidence_id']}",) if verified else (),
    )


def _build_task_store(
    path: Path,
    tasks: tuple[NativeMemoryTask, ...],
) -> tuple[AppendOnlyEventStore, dict[str, tuple[str, ...]]]:
    store = AppendOnlyEventStore(path)
    if store.count() != 0:
        raise NativeMemoryComparisonError("evaluation ledger must start empty")
    events = []
    support: dict[str, tuple[str, ...]] = {}
    index = 0
    for task in tasks:
        ids = []
        for turn in task.turns:
            events.append(_observed_turn_event(turn, index))
            index += 1
            if turn["operation"] != "poison":
                ids.append(str(turn["evidence_id"]))
        support[task.task_id] = tuple(ids)
    store.append_batch(events)
    return store, support


@dataclass(frozen=True)
class PreparedAnswer:
    value: Any
    support_event_ids: tuple[str, ...]
    memory_flops: int
    storage_bytes: int
    policy_errors: int
    operation_counts: dict[str, int]
    state_trace_hash: str
    unsupported_durable_writes: int = 0


def _explicit_answer(task: NativeMemoryTask) -> PreparedAnswer:
    values: dict[str, tuple[Any, tuple[str, ...]]] = {}
    scanned = 0
    for turn in task.turns:
        scanned += 1
        operation = turn["operation"]
        key = str(turn.get("key", ""))
        event_id = str(turn["evidence_id"])
        if operation == "remember":
            values[key] = (turn.get("value"), (event_id,))
        elif operation in {"update", "contradict"}:
            values[key] = (turn.get("value"), (event_id,))
        elif operation == "reset":
            values.clear()
        elif operation == "poison":
            continue
    selected = values.get(str(task.query["key"]))
    payload = {
        "task_id": task.task_id,
        "turns": list(task.turns),
        "resolved": values,
    }
    return PreparedAnswer(
        value=selected[0] if selected else None,
        support_event_ids=selected[1] if selected else (),
        memory_flops=2 * len(canonical_json(payload).encode("utf-8")) + 8 * scanned,
        storage_bytes=0,
        policy_errors=0,
        operation_counts={"explicit_view_scan": scanned},
        state_trace_hash=sha256_json(payload),
    )


def _eligible_examples(task: NativeMemoryTask) -> dict[int, GateExample]:
    return {example.turn_index: example for example in gate_examples((task,))}


def _find_slot(memory: SessionMemory, content_key: str):
    matches = [
        slot for slot in memory.slots if slot.active and slot.content_key == content_key
    ]
    return matches[-1] if matches else None


def _apply_memory_decision(
    memory: SessionMemory,
    store: AppendOnlyEventStore,
    bundle: MemoryEvidenceBundle,
    task: NativeMemoryTask,
    turn: dict[str, Any],
    decision: str,
) -> None:
    key = str(turn.get("key", ""))
    support = (str(turn["evidence_id"]),)
    if decision in {"remember", "merge"} and len(memory.slots) >= memory.config.maximum_slots:
        raise CapacityExceeded("learned operation would exceed fixed slot capacity")
    if decision == "remember":
        memory.remember(
            store,
            access=memory.access,
            now=NOW,
            bundle=bundle,
            content_key=key,
            value=turn.get("value"),
            support_event_ids=support,
            protected_regions=task.protected_regions,
        )
    elif decision == "update":
        slot = _find_slot(memory, key)
        if slot is None:
            raise NativeMemoryError("predicted update has no active exact-key slot")
        memory.update(
            store,
            access=memory.access,
            now=NOW,
            bundle=bundle,
            slot_id=slot.slot_id,
            value=turn.get("value"),
            support_event_ids=support,
        )
    elif decision == "merge":
        candidates = [
            slot
            for slot in memory.slots
            if slot.active
            and (
                "-variant-" in slot.content_key
                or slot.content_key.startswith("variants:")
            )
        ]
        if not candidates:
            raise NativeMemoryError("predicted merge has no prior variant slot")
        staged = memory.remember(
            store,
            access=memory.access,
            now=NOW,
            bundle=bundle,
            content_key=key,
            value=turn.get("value"),
            support_event_ids=support,
            protected_regions=task.protected_regions,
        )
        prior = sorted(candidates, key=lambda slot: (slot.updated_turn, slot.slot_id))[-1]
        original_key = str(task.query["key"])
        memory.merge(
            store,
            access=memory.access,
            now=NOW,
            bundle=bundle,
            slot_ids=(prior.slot_id, staged.slot_id),
            content_key=f"variants:{original_key}",
            value=[prior.value, staged.value],
        )
    elif decision == "forget":
        active = [slot for slot in memory.slots if slot.active]
        if not active:
            raise NativeMemoryError("predicted forget has no active slot")
        for slot in active:
            memory.forget(
                access=memory.access,
                now=NOW,
                slot_id=slot.slot_id,
                reason="explicit-user",
            )
    else:
        raise NativeMemoryError(f"unregistered learned decision {decision!r}")


def _memory_answer(
    task: NativeMemoryTask,
    store: AppendOnlyEventStore,
    support_ids: tuple[str, ...],
    decisions: Mapping[str, str],
    *,
    prefill_occupancy: int = 0,
) -> PreparedAnswer:
    bundle = MemoryEvidenceBundle.create(
        store,
        event_ids=support_ids,
        valid_at=VALID_AT,
        ontology_version="v1",
    )
    memory = SessionMemory(
        NativeMemoryConfig(
            session_id=f"session:{task.task_id}",
            task_id=task.task_id,
            owner_id="owner:e0.16",
            maximum_slots=16,
            maximum_serialized_bytes=65536,
            expires_at=EXPIRES,
            base_weight_version="fixed-depth-v1:e0.14:update-2000",
            ontology_version="v1",
        )
    )
    examples = _eligible_examples(task)
    policy_errors = 0
    trace = []
    if prefill_occupancy < 0 or prefill_occupancy > 16:
        raise NativeMemoryComparisonError("capacity probe occupancy is outside 0..16")
    filler_support = (support_ids[0],)
    for index in range(prefill_occupancy):
        memory.remember(
            store,
            access=memory.access,
            now=NOW,
            bundle=bundle,
            content_key=f"capacity:{task.task_id}:{index:02d}",
            value=f"filler-{index:02d}",
            support_event_ids=filler_support,
            protected_regions=task.protected_regions,
        )
    for turn_index, turn in enumerate(task.turns):
        example = examples.get(turn_index)
        if example is not None:
            decision = decisions.get(example.example_id)
            if decision is None:
                raise NativeMemoryComparisonError("memory policy omitted an eligible turn")
            try:
                _apply_memory_decision(memory, store, bundle, task, turn, decision)
            except NativeMemoryError:
                policy_errors += 1
            trace.append(
                {
                    "turn_index": turn_index,
                    "decision": decision,
                    "state_hash": memory.state_hash,
                }
            )
        if turn["operation"] == "reset":
            memory.reset(access=memory.access, now=NOW)
            trace.append(
                {
                    "turn_index": turn_index,
                    "decision": "mandatory-reset",
                    "state_hash": memory.state_hash,
                }
            )
    read = memory.prepare_model_call(
        store,
        access=memory.access,
        now=NOW,
        bundle=bundle,
    )
    selected = [
        slot
        for slot in read.active_slots
        if slot.content_key == str(task.query["key"])
    ]
    slot = selected[-1] if selected else None
    compute = memory.compute_record()
    return PreparedAnswer(
        value=slot.value if slot else None,
        support_event_ids=slot.support_event_ids if slot else (),
        memory_flops=compute.realized_memory_flops,
        storage_bytes=compute.serialized_slot_bytes,
        policy_errors=policy_errors,
        operation_counts=compute.operation_counts,
        state_trace_hash=sha256_json(trace),
    )


def _tombstone_event(event_id: str, target: str) -> Event:
    return Event.create(
        event_id=event_id,
        transaction_time="2026-08-12T12:01:00+00:00",
        valid_from="2026-08-12T00:00:00+00:00",
        actor="native-memory-e0.16-verifier",
        source_id="verifier:native-memory-e0.16",
        verifier_id="verifier:native-memory-e0.16",
        modality="structured-control",
        speech_act=SpeechAct.TOMBSTONE,
        payload={"reason": "registered post-invalidation probe"},
        parent_ids=(target,),
        target_event_ids=(target,),
        verifier_class="executable",
        provenance_witnesses=("external:e0.16:invalidation",),
    )


def run_secondary_probes(
    root: Path,
    *,
    seed: int,
    core: RecurrentReasoner,
    gate: MinimalOperationGate,
) -> dict[str, Any]:
    registry = json.loads(
        (root / "contracts" / "wamrx_native_memory_splits_v1.json").read_text()
    )
    task = next(
        task
        for task in split_tasks(registry, "ood")
        if task.family == "delayed_recall"
    )
    codec = ByteCodec(
        maximum_input_tokens=core.config.maximum_input_tokens,
        maximum_output_tokens=core.config.maximum_output_tokens,
    )
    learned, _ = _gate_decisions(
        core, gate, codec, tuple(_eligible_examples(task).values())
    )
    curves: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARM_IDS}
    for occupancy in (0, 4, 8, 12, 16):
        with tempfile.TemporaryDirectory() as directory:
            store, support = _build_task_store(Path(directory) / "ledger.sqlite", (task,))
            for arm_id in ARM_IDS:
                if arm_id == "explicit-multiview-v1":
                    answer = _explicit_answer(task)
                else:
                    decisions = (
                        _target_decisions(task)
                        if arm_id == "deterministic-session-cache-v1"
                        else learned
                    )
                    answer = _memory_answer(
                        task,
                        store,
                        support[task.task_id],
                        decisions,
                        prefill_occupancy=occupancy,
                    )
                curves[arm_id].append(
                    {
                        "occupancy": occupancy,
                        "selected_value_correct": answer.value == task.expected,
                        "policy_errors": answer.policy_errors,
                        "storage_bytes": answer.storage_bytes,
                        "memory_or_view_flops": answer.memory_flops,
                    }
                )

    invalidation: dict[str, dict[str, Any]] = {}
    for arm_id in ARM_IDS:
        if arm_id == "explicit-multiview-v1":
            invalidation[arm_id] = {
                "had_slot": False,
                "disabled_after_tombstone": True,
                "stale_value_emitted": False,
            }
            continue
        with tempfile.TemporaryDirectory() as directory:
            store, support = _build_task_store(Path(directory) / "ledger.sqlite", (task,))
            bundle = MemoryEvidenceBundle.create(
                store,
                event_ids=support[task.task_id],
                valid_at=VALID_AT,
                ontology_version="v1",
            )
            memory = SessionMemory(
                NativeMemoryConfig(
                    session_id=f"invalidation:{seed}:{arm_id}",
                    task_id=task.task_id,
                    owner_id="owner:e0.16",
                    maximum_slots=16,
                    maximum_serialized_bytes=65536,
                    expires_at=EXPIRES,
                    base_weight_version="fixed-depth-v1:e0.14:update-2000",
                    ontology_version="v1",
                )
            )
            first = task.turns[0]
            decision = (
                "remember"
                if arm_id == "deterministic-session-cache-v1"
                else learned[f"{task.task_id}:turn:000"]
            )
            try:
                _apply_memory_decision(memory, store, bundle, task, first, decision)
            except NativeMemoryError:
                pass
            prior = _find_slot(memory, str(task.query["key"]))
            target = str(first["evidence_id"])
            store.append(_tombstone_event(f"invalidate:{seed}:{arm_id}", target))
            remaining = tuple(item for item in support[task.task_id] if item != target)
            fresh = MemoryEvidenceBundle.create(
                store,
                event_ids=remaining,
                valid_at=VALID_AT,
                ontology_version="v1",
            )
            read = memory.prepare_model_call(
                store,
                access=memory.access,
                now=NOW,
                bundle=fresh,
            )
            active_query = any(
                slot.content_key == str(task.query["key"]) for slot in read.active_slots
            )
            invalidation[arm_id] = {
                "had_slot": prior is not None,
                "disabled_after_tombstone": not active_query,
                "stale_value_emitted": active_query,
            }
    counterfactual_task = next(
        candidate
        for candidate in split_tasks(registry, "ood")
        if candidate.family == "correction"
    )
    counterfactual_decisions, _ = _gate_decisions(
        core,
        gate,
        codec,
        tuple(_eligible_examples(counterfactual_task).values()),
    )
    with tempfile.TemporaryDirectory() as directory:
        store, support = _build_task_store(
            Path(directory) / "ledger.sqlite", (counterfactual_task,)
        )
        learned_answer = _memory_answer(
            counterfactual_task,
            store,
            support[counterfactual_task.task_id],
            counterfactual_decisions,
        )
    with tempfile.TemporaryDirectory() as directory:
        store, support = _build_task_store(
            Path(directory) / "ledger.sqlite", (counterfactual_task,)
        )
        fixed_remember = {
            example.example_id: "remember"
            for example in _eligible_examples(counterfactual_task).values()
        }
        counterfactual_answer = _memory_answer(
            counterfactual_task,
            store,
            support[counterfactual_task.task_id],
            fixed_remember,
        )
    return {
        "seed": seed,
        "task_id": task.task_id,
        "capacity_saturation_curve": curves,
        "post_invalidation_performance": invalidation,
        "decision_counterfactual": {
            "task_id": counterfactual_task.task_id,
            "learned_trace_hash": learned_answer.state_trace_hash,
            "fixed_remember_trace_hash": counterfactual_answer.state_trace_hash,
            "changes_trace": (
                learned_answer.state_trace_hash
                != counterfactual_answer.state_trace_hash
            ),
        },
    }


def _target_decisions(task: NativeMemoryTask) -> dict[str, str]:
    examples = _eligible_examples(task)
    return {
        example.example_id: example.target_operation for example in examples.values()
    }


def _gate_decisions(
    core: RecurrentReasoner,
    gate: MinimalOperationGate,
    codec: ByteCodec,
    examples: tuple[GateExample, ...],
) -> tuple[dict[str, str], dict[str, Any]]:
    if not examples:
        return {}, {"examples": 0, "correct": 0, "accuracy": 0.0}
    feature_by_id, target_by_id = feature_maps(core, codec, examples)
    ordered = [example.example_id for example in examples]
    features = mx.stack([feature_by_id[item] for item in ordered])
    decisions = gate.decide(features)
    predicted = dict(zip(ordered, decisions))
    correct = sum(
        OPERATION_ORDER[target_by_id[item]] == predicted[item] for item in ordered
    )
    return predicted, {
        "examples": len(ordered),
        "correct": correct,
        "accuracy": correct / len(ordered),
    }


def _readout_prompts(
    codec: ByteCodec,
    prepared: list[PreparedAnswer],
    tasks: list[NativeMemoryTask],
) -> mx.array:
    tokens = []
    for answer, task in zip(prepared, tasks):
        prompt = {
            "instruction": "Return candidate_value exactly; return null when absent.",
            "query": task.query,
            "candidate_value": answer.value,
            "support_event_ids": list(answer.support_event_ids),
        }
        encoded, _ = codec.encode_problem(prompt)
        tokens.append(encoded)
    return mx.array(tokens, dtype=mx.int32)


def _journal_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["manifest_hash"],
        row["seed"],
        row["arm_id"],
        row["split"],
        row["task_id"],
    )


def _load_journal(path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise NativeMemoryComparisonError(
                    f"invalid evaluation journal line {line_number}"
                ) from error
            key = _journal_key(row)
            if key in rows and rows[key] != row:
                raise NativeMemoryComparisonError("evaluation journal key conflict")
            rows[key] = row
    return rows


def _append_journal(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with path.open("a") as handle:
        handle.write(payload + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def evaluate_seed(
    root: Path,
    run_directory: Path,
    *,
    seed: int,
    core: RecurrentReasoner,
    gate: MinimalOperationGate,
    manifest: dict[str, Any],
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registry = json.loads(
        (root / "contracts" / "wamrx_native_memory_splits_v1.json").read_text()
    )
    codec = ByteCodec(
        maximum_input_tokens=core.config.maximum_input_tokens,
        maximum_output_tokens=core.config.maximum_output_tokens,
    )
    journal_path = run_directory / "evaluation" / "journal.jsonl"
    journal = _load_journal(journal_path)
    manifest_hash = sha256_json(manifest)
    rows: list[dict[str, Any]] = []
    auxiliary: dict[str, Any] = {}
    for split in ("id", "ood"):
        tasks = split_tasks(registry, split)
        examples = split_gate_examples(registry, split)
        learned_decisions, auxiliary[split] = _gate_decisions(
            core, gate, codec, examples
        )
        with tempfile.TemporaryDirectory() as directory:
            store, support = _build_task_store(Path(directory) / "ledger.sqlite", tasks)
            for arm_id in ARM_IDS:
                expected_keys = {
                    (manifest_hash, seed, arm_id, split, task.task_id) for task in tasks
                }
                existing = {
                    key: row for key, row in journal.items() if key in expected_keys
                }
                missing_tasks = [
                    task
                    for task in tasks
                    if (manifest_hash, seed, arm_id, split, task.task_id) not in existing
                ]
                prepared: list[PreparedAnswer] = []
                for task in missing_tasks:
                    if arm_id == "explicit-multiview-v1":
                        answer = _explicit_answer(task)
                    elif arm_id == "deterministic-session-cache-v1":
                        answer = _memory_answer(
                            task, store, support[task.task_id], _target_decisions(task)
                        )
                    else:
                        decisions = {
                            key: value
                            for key, value in learned_decisions.items()
                            if key.startswith(f"{task.task_id}:turn:")
                        }
                        answer = _memory_answer(
                            task, store, support[task.task_id], decisions
                        )
                    prepared.append(answer)
                if missing_tasks:
                    tokens = _readout_prompts(codec, prepared, missing_tasks)
                    outputs = core(tokens, macro_steps=4, micro_steps=1)
                    mx.eval(outputs["logits"])
                    predictions = decode_batch(codec, outputs["logits"])
                    core_flops = core.estimated_inference_flops(
                        input_tokens=core.config.maximum_input_tokens,
                        macro_steps=4,
                        micro_steps=1,
                    )
                    for task, answer, prediction in zip(
                        missing_tasks, prepared, predictions
                    ):
                        gate_calls = (
                            len(_eligible_examples(task))
                            if arm_id == "minimal-gated-native-memory-v1"
                            else 0
                        )
                        gate_feature_flops = gate_calls * _prelude_flops(core)
                        gate_flops = gate_calls * GATE_FORWARD_FLOPS_PER_EXAMPLE
                        total_flops = (
                            core_flops
                            + answer.memory_flops
                            + gate_feature_flops
                            + gate_flops
                        )
                        row = {
                            "manifest_hash": manifest_hash,
                            "seed": seed,
                            "arm_id": arm_id,
                            "split": split,
                            "task_id": task.task_id,
                            "family": task.family,
                            "generalization_axis": task.generalization_axis,
                            "protected_regions": list(task.protected_regions),
                            "expected": task.expected,
                            "pre_reset_values": [
                                turn.get("value")
                                for turn in task.turns
                                if turn["operation"] == "remember"
                            ]
                            if task.family == "reset_task_switch"
                            else [],
                            "selected_value": answer.value,
                            "prediction": prediction,
                            "correct": prediction == task.expected,
                            "support_event_ids": list(answer.support_event_ids),
                            "policy_errors": answer.policy_errors,
                            "operation_counts": answer.operation_counts,
                            "state_trace_hash": answer.state_trace_hash,
                            "core_readout_flops": core_flops,
                            "gate_feature_calls": gate_calls,
                            "gate_feature_flops": gate_feature_flops,
                            "gate_flops": gate_flops,
                            "memory_or_view_flops": answer.memory_flops,
                            "total_inference_flops": total_flops,
                            "storage_bytes": answer.storage_bytes,
                            "unsupported_durable_writes": answer.unsupported_durable_writes,
                        }
                        key = _journal_key(row)
                        if key in journal and journal[key] != row:
                            raise NativeMemoryComparisonError("evaluation resume row drifted")
                        if key not in journal:
                            _append_journal(journal_path, row)
                            journal[key] = row
                rows.extend(
                    journal[(manifest_hash, seed, arm_id, split, task.task_id)]
                    for task in tasks
                )
    return rows, auxiliary


def aggregate_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for grouping, labels in (
            ("overall", ("all",)),
            ("family", (row["family"],)),
            ("region", tuple(row["protected_regions"])),
        ):
            for label in labels:
                grouped[
                    (row["arm_id"], row["seed"], row["split"], grouping, label)
                ].append(row)
    metrics = []
    for (arm_id, seed, split, grouping, label), values in sorted(grouped.items()):
        correct = sum(bool(row["correct"]) for row in values)
        total_flops = sum(int(row["total_inference_flops"]) for row in values)
        metrics.append(
            {
                "arm_id": arm_id,
                "seed": seed,
                "split": split,
                "grouping": grouping,
                "label": label,
                "examples": len(values),
                "correct": correct,
                "accuracy": correct / len(values),
                "total_inference_flops": total_flops,
                "compute_normalized_score": correct * 100000000 / total_flops,
                "maximum_storage_bytes": max(int(row["storage_bytes"]) for row in values),
                "policy_errors": sum(int(row["policy_errors"]) for row in values),
            }
        )
    return metrics


def _metric_index(metrics: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {
        (
            row["arm_id"],
            row["seed"],
            row["split"],
            row["grouping"],
            row["label"],
        ): row
        for row in metrics
    }


def secondary_reports(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correction: dict[str, dict[str, Any]] = {}
    reset: dict[str, dict[str, Any]] = {}
    for arm_id in ARM_IDS:
        correction_rows = [
            row
            for row in rows
            if row["arm_id"] == arm_id
            and row["family"] in {"correction", "temporal_update"}
        ]
        successful = [row for row in correction_rows if row["correct"]]
        per_seed = {}
        for seed in sorted({int(row["seed"]) for row in correction_rows}):
            seed_rows = [row for row in correction_rows if int(row["seed"]) == seed]
            seed_successful = [row for row in seed_rows if row["correct"]]
            per_seed[str(seed)] = {
                "attempts": len(seed_rows),
                "successful": len(seed_successful),
                "median_turns_after_final_correction": (
                    0 if seed_successful else None
                ),
                "median_realized_flops": (
                    statistics.median(
                        row["total_inference_flops"] for row in seed_successful
                    )
                    if seed_successful
                    else None
                ),
            }
        correction[arm_id] = {
            "attempts": len(correction_rows),
            "successful": len(successful),
            "median_turns_after_final_correction": 0 if successful else None,
            "median_realized_flops": (
                statistics.median(row["total_inference_flops"] for row in successful)
                if successful
                else None
            ),
            "per_seed": per_seed,
        }
        reset_rows = [
            row
            for row in rows
            if row["arm_id"] == arm_id and row["family"] == "reset_task_switch"
        ]
        leaks = sum(
            row["prediction"] in row["pre_reset_values"] for row in reset_rows
        )
        reset[arm_id] = {
            "examples": len(reset_rows),
            "leaks": leaks,
            "leakage_rate": leaks / len(reset_rows) if reset_rows else 0.0,
        }
    return {
        "correction_latency": correction,
        "reset_leakage": reset,
        "unsupported_durable_writes": sum(
            int(row["unsupported_durable_writes"]) for row in rows
        ),
    }


def promotion_audit(
    metrics: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    seeds: tuple[int, ...],
    manipulations: dict[str, bool],
    secondary_probes: list[dict[str, Any]],
) -> dict[str, Any]:
    index = _metric_index(metrics)
    learned = "minimal-gated-native-memory-v1"
    explicit = "explicit-multiview-v1"
    cache = "deterministic-session-cache-v1"

    def deltas(split: str, grouping: str, label: str, left: str, right: str, field: str):
        return [
            float(index[(left, seed, split, grouping, label)][field])
            - float(index[(right, seed, split, grouping, label)][field])
            for seed in seeds
        ]

    ood = holm_adjusted_tests(
        {
            "learned_vs_explicit": (
                deltas("ood", "overall", "all", learned, explicit, "accuracy"),
                0.0,
            ),
            "learned_vs_cache": (
                deltas("ood", "overall", "all", learned, cache, "accuracy"),
                0.0,
            ),
        }
    )
    id_tests = {
        baseline: paired_student_t(
            deltas("id", "overall", "all", learned, baseline, "accuracy"),
            null_margin=-0.02,
            alpha=0.05,
        )
        for baseline in (explicit, cache)
    }
    regions = sorted(
        {row["label"] for row in metrics if row["grouping"] == "region"}
    )
    protected = {
        f"learned_vs_{baseline}:{region}": paired_student_t(
            deltas("ood", "region", region, learned, baseline, "accuracy"),
            null_margin=0.0,
            alpha=0.05,
        )
        for baseline in (explicit, cache)
        for region in regions
    }
    compute = {
        baseline: paired_student_t(
            deltas(
                "ood",
                "overall",
                "all",
                learned,
                baseline,
                "compute_normalized_score",
            ),
            null_margin=0.0,
            alpha=0.05,
        )
        for baseline in (explicit, cache)
    }
    cache_test = paired_student_t(
        deltas("ood", "overall", "all", cache, explicit, "accuracy"),
        null_margin=0.0,
        alpha=0.05,
    )
    cache_id_tests = paired_student_t(
        deltas("id", "overall", "all", cache, explicit, "accuracy"),
        null_margin=-0.02,
        alpha=0.05,
    )
    cache_protected = {
        region: paired_student_t(
            deltas("ood", "region", region, cache, explicit, "accuracy"),
            null_margin=0.0,
            alpha=0.05,
        )
        for region in regions
    }
    family_tests = {
        family: paired_student_t(
            deltas("ood", "family", family, learned, cache, "accuracy"),
            null_margin=0.0,
            alpha=0.05,
        )
        for family in sorted(
            {row["label"] for row in metrics if row["grouping"] == "family"}
        )
    }
    primary_pass = all(
        result["holm_rejects_null"]
        and result["one_sided_lower_bound"] > 0.0
        for result in ood["hypotheses"].values()
    )
    id_pass = all(result["one_sided_lower_bound"] >= -0.02 for result in id_tests.values())
    protected_pass = all(
        result["one_sided_lower_bound"] >= 0.0 for result in protected.values()
    )
    compute_pass = all(
        result["one_sided_lower_bound"] > 0.0 for result in compute.values()
    )
    secondary = secondary_reports(rows)
    secondary["registered_probes"] = secondary_probes
    invalidation_safe = all(
        not report["stale_value_emitted"]
        and (not report["had_slot"] or report["disabled_after_tombstone"])
        for probe in secondary_probes
        for report in probe["post_invalidation_performance"].values()
    )
    capacity_complete = all(
        [row["occupancy"] for row in curve] == [0, 4, 8, 12, 16]
        and all(row["storage_bytes"] <= 65536 for row in curve)
        for probe in secondary_probes
        for curve in probe["capacity_saturation_curve"].values()
    )
    structural_pass = all(manipulations.values())
    boundary_safety_pass = (
        secondary["unsupported_durable_writes"] == 0
        and invalidation_safe
        and capacity_complete
    )
    learned_reset_pass = secondary["reset_leakage"][learned]["leakage_rate"] == 0.0
    cache_reset_pass = secondary["reset_leakage"][cache]["leakage_rate"] == 0.0
    learned_correction = secondary["correction_latency"][learned]
    learned_correction_pass = (
        learned_correction["attempts"] > 0
        and learned_correction["successful"] == learned_correction["attempts"]
        and all(
            seed_report["median_turns_after_final_correction"] is not None
            and seed_report["median_turns_after_final_correction"]
            <= max(
                secondary["correction_latency"][baseline]["per_seed"][seed][
                    "median_turns_after_final_correction"
                ]
                if secondary["correction_latency"][baseline]["per_seed"][seed][
                    "median_turns_after_final_correction"
                ]
                is not None
                else 0
                for baseline in (explicit, cache)
            )
            for seed, seed_report in learned_correction["per_seed"].items()
        )
    )
    learned_safety_pass = (
        structural_pass
        and boundary_safety_pass
        and learned_reset_pass
        and learned_correction_pass
    )
    cache_safety_pass = structural_pass and boundary_safety_pass and cache_reset_pass
    cache_id_pass = cache_id_tests["one_sided_lower_bound"] >= -0.02
    cache_protected_pass = all(
        result["one_sided_lower_bound"] >= 0.0
        for result in cache_protected.values()
    )
    family_positive = [
        family
        for family, result in family_tests.items()
        if result["one_sided_lower_bound"] > 0.0
    ]
    if not structural_pass:
        decision = "INVALID"
    elif primary_pass and id_pass and protected_pass and learned_safety_pass and compute_pass:
        decision = "COMPLETE_ADOPT_LEARNED_MEMORY"
    elif primary_pass and id_pass and protected_pass and learned_safety_pass:
        decision = "COMPLETE_COMPUTE_ONLY_ACCELERATOR"
    elif len(family_positive) == 1 and id_pass and protected_pass and learned_safety_pass:
        decision = "COMPLETE_SPECIALIST_ONLY"
    elif (
        cache_test["one_sided_lower_bound"] > 0.0
        and cache_id_pass
        and cache_protected_pass
        and cache_safety_pass
    ):
        decision = "COMPLETE_RETAIN_DETERMINISTIC_CACHE"
    else:
        decision = "COMPLETE_RETAIN_EXPLICIT_MULTIVIEW"
    return {
        "statistical_procedure": "one-sided paired Student-t; Holm step-down within registered learned OOD comparisons",
        "learned_ood_superiority": ood,
        "id_noninferiority": id_tests,
        "protected_region_noninferiority": protected,
        "compute_normalized_superiority": compute,
        "cache_vs_explicit": cache_test,
        "cache_id_noninferiority": cache_id_tests,
        "cache_protected_region_noninferiority": cache_protected,
        "learned_vs_cache_family_tests": family_tests,
        "learned_positive_families": family_positive,
        "secondary_reports": secondary,
        "manipulations": manipulations,
        "registered_gates": {
            "primary_pass": primary_pass,
            "id_pass": id_pass,
            "protected_pass": protected_pass,
            "compute_pass": compute_pass,
            "structural_pass": structural_pass,
            "boundary_safety_pass": boundary_safety_pass,
            "learned_reset_pass": learned_reset_pass,
            "learned_correction_pass": learned_correction_pass,
            "learned_safety_pass": learned_safety_pass,
            "cache_id_pass": cache_id_pass,
            "cache_protected_pass": cache_protected_pass,
            "cache_reset_pass": cache_reset_pass,
            "cache_safety_pass": cache_safety_pass,
        },
        "decision": decision,
    }


def run_post_training_manipulations(
    root: Path,
    *,
    training_reports: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    secondary_probes: list[dict[str, Any]],
) -> dict[str, bool]:
    e0_15 = json.loads(
        (root / "results" / "e0_15_native_memory_boundary.json").read_text()
    )
    structural = all(e0_15.get("manipulations", {}).values())
    return {
        "M1_frozen_core_state_exact": all(
            report["core_hash_before"] == report["core_hash_after"]
            for report in training_reports
        ),
        "M2_learned_decisions_change_state_trace": any(
            probe["decision_counterfactual"]["changes_trace"]
            for probe in secondary_probes
        ),
        "M3_omitted_support_fails": structural,
        "M4_tombstone_disables_before_readout": structural
        and all(
            not report["stale_value_emitted"]
            and (not report["had_slot"] or report["disabled_after_tombstone"])
            for probe in secondary_probes
            for report in probe["post_invalidation_performance"].values()
        ),
        "M5_reset_and_cross_owner_leave_no_state": structural,
        "M6_capacity_fails_closed": structural
        and all(
            [row["occupancy"] for row in curve] == [0, 4, 8, 12, 16]
            and all(row["storage_bytes"] <= 65536 for row in curve)
            for probe in secondary_probes
            for curve in probe["capacity_saturation_curve"].values()
        ),
        "M7_stale_checkpoint_identity_fails": structural,
        "M8_unverified_poison_rejected": structural,
        "M9_protected_decay_rejected": structural,
        "M10_memory_identifiers_not_evidence": structural,
        "M11_complete_compute_and_storage_accounting": all(
            row["total_inference_flops"] > 0
            and row["storage_bytes"] >= 0
            for row in rows
        ),
    }


def run_registered_comparison(
    root: Path,
    run_directory: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    registration = validate_run_registration(root)
    contract = json.loads(
        (root / "contracts" / "wamrx_native_memory_run_v1.json").read_text()
    )
    manifest = build_run_manifest(root)
    run_directory.mkdir(parents=True, exist_ok=True)
    state_path = run_directory / "run-state.json"
    journal_path = run_directory / "evaluation" / "journal.jsonl"
    state = {
        "experiment": "E0.16",
        "run_id": RUN_ID,
        "status": "INCOMPLETE",
        "manifest": manifest,
        "registration": registration,
        "completed_seeds": [],
        "invalid_reasons": [],
        "incomplete_reasons": [],
    }
    if state_path.exists() and resume:
        prior = json.loads(state_path.read_text())
        if prior.get("manifest") != manifest or prior.get("run_id") != RUN_ID:
            raise NativeMemoryComparisonError("resume run identity mismatch")
        if prior.get("status") == "INVALID":
            raise NativeMemoryComparisonError(
                "an INVALID result requires a new versioned protocol"
            )
        state["completed_seeds"] = list(prior.get("completed_seeds", ()))
        state["incomplete_reasons"] = list(prior.get("incomplete_reasons", ()))
    elif (state_path.exists() or journal_path.exists()) and not resume:
        raise NativeMemoryComparisonError(
            "run state already exists; use --resume or a new run directory"
        )
    _atomic_json(state_path, state)

    all_rows: list[dict[str, Any]] = []
    training_reports: list[dict[str, Any]] = []
    gate_auxiliary: list[dict[str, Any]] = []
    secondary_probes: list[dict[str, Any]] = []
    try:
        for seed in map(int, contract["paired_seeds"]):
            core, gate, training = train_gate_seed(
                root,
                run_directory,
                seed=seed,
                manifest=manifest,
                resume=resume,
            )
            training_reports.append(training)
            rows, auxiliary = evaluate_seed(
                root,
                run_directory,
                seed=seed,
                core=core,
                gate=gate,
                manifest=manifest,
                resume=resume,
            )
            all_rows.extend(rows)
            gate_auxiliary.append({"seed": seed, "splits": auxiliary})
            secondary_probes.append(
                run_secondary_probes(root, seed=seed, core=core, gate=gate)
            )
            if seed not in state["completed_seeds"]:
                state["completed_seeds"].append(seed)
            _atomic_json(state_path, state)
        expected_rows = len(contract["paired_seeds"]) * len(ARM_IDS) * 2 * 36
        if len(all_rows) != expected_rows:
            state["incomplete_reasons"].append(
                f"expected {expected_rows} evaluation rows, found {len(all_rows)}"
            )
            state["status"] = "INCOMPLETE"
            _atomic_json(state_path, state)
            return state
        metrics = aggregate_metrics(all_rows)
        manipulations = run_post_training_manipulations(
            root,
            training_reports=training_reports,
            rows=all_rows,
            secondary_probes=secondary_probes,
        )
        audit = promotion_audit(
            metrics,
            all_rows,
            seeds=tuple(map(int, contract["paired_seeds"])),
            manipulations=manipulations,
            secondary_probes=secondary_probes,
        )
        result = {
            **state,
            "status": audit["decision"],
            "training_reports": training_reports,
            "gate_auxiliary_metrics": gate_auxiliary,
            "secondary_probes": secondary_probes,
            "evaluation_rows": all_rows,
            "metric_rows": metrics,
            "promotion_audit": audit,
        }
        if audit["decision"] == "INVALID":
            result["invalid_reasons"] = [
                "one or more registered post-training manipulations failed"
            ]
        _atomic_json(state_path, result)
        return result
    except (KeyboardInterrupt, MemoryError) as error:
        state["incomplete_reasons"].append(type(error).__name__)
        state["status"] = "INCOMPLETE"
        _atomic_json(state_path, state)
        raise
    except (
        NativeMemoryComparisonError,
        NativeMemoryModelError,
        NativeMemoryError,
        ValueError,
        FloatingPointError,
    ) as error:
        state["invalid_reasons"].append(f"{type(error).__name__}: {error}")
        state["status"] = "INVALID"
        _atomic_json(state_path, state)
        raise
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        lowered = str(error).lower()
        if isinstance(error, RuntimeError) and any(
            marker in lowered
            for marker in ("out of memory", "allocation", "memory exhausted")
        ):
            state["incomplete_reasons"].append(message)
            state["status"] = "INCOMPLETE"
        else:
            state["invalid_reasons"].append(message)
            state["status"] = "INVALID"
        _atomic_json(state_path, state)
        raise
