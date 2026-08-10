"""Complete registered Milestone 3 comparison runner.

Importing this module performs no training. ``run_registered_comparison`` is
the only entry point that can consume the frozen five-seed budget.
"""

from __future__ import annotations

from collections import defaultdict
import dataclasses
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any, Iterable

import mlx.core as mx

from .canonical import canonical_json, sha256_json
from .recurrent import (
    ComputeBudget,
    EvidenceBundle,
    RecurrentContractError,
    UnresolvedConstraintState,
    decide_halt,
)
from .recurrent_checkpoint import (
    CheckpointMetadata,
    RecurrentCheckpointError,
    file_sha256,
    load_checkpoint,
    save_checkpoint,
)
from .recurrent_executor import execute_task, task_evidence_bundle
from .recurrent_model import (
    ARM_IDS,
    ByteCodec,
    RecurrentModelConfig,
    RecurrentReasoner,
    parameter_count,
)
from .recurrent_run import (
    RunComputeRecord,
    holm_adjusted_tests,
    paired_student_t,
    resolve_terminal_status,
)
from .recurrent_tasks import FAMILIES, RecurrentTask, generate_family, load_split_registry
from .recurrent_training import (
    OptimizerConfig,
    build_schedule,
    create_optimizer,
    estimated_schedule_training_flops,
    schedule_hash,
    train_schedule_segment,
    training_tasks,
)

COMPARISON_RUNNER_VERSION = "wamrx-recurrent-comparison-v1"


class RecurrentComparisonError(ValueError):
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
    relative_paths = (
        "wamrx/recurrent_model.py",
        "wamrx/recurrent_training.py",
        "wamrx/recurrent_executor.py",
        "wamrx/recurrent_checkpoint.py",
        "wamrx/recurrent_comparison.py",
    )
    return sha256_json(
        {relative: file_sha256(root / relative) for relative in relative_paths}
    )


def build_run_manifest(root: Path) -> dict[str, Any]:
    run_contract = root / "contracts" / "wamrx_recurrent_run_v1.json"
    training_config = root / "contracts" / "wamrx_recurrent_training_v1.json"
    split_registry = root / "contracts" / "wamrx_recurrent_splits.json"
    reasoner_contract = root / "contracts" / "wamrx_recurrent_reasoner.json"
    return {
        "runner_version": COMPARISON_RUNNER_VERSION,
        "model_implementation_hash": _implementation_hash(root),
        "run_contract_hash": _json_hash(run_contract),
        "training_config_hash": _json_hash(training_config),
        "split_registry_hash": _json_hash(split_registry),
        "reasoner_contract_hash": _json_hash(reasoner_contract),
    }


def _checkpoint_metadata(
    manifest: dict[str, Any],
    *,
    run_id: str,
    arm_id: str,
    seed: int,
    completed_updates: int,
    frozen_schedule_hash: str,
    realized_training_flops: int,
    training_examples_seen: int,
) -> CheckpointMetadata:
    return CheckpointMetadata(
        run_id=run_id,
        arm_id=arm_id,
        seed=seed,
        completed_updates=completed_updates,
        schedule_hash=frozen_schedule_hash,
        model_implementation_hash=manifest["model_implementation_hash"],
        training_config_hash=manifest["training_config_hash"],
        split_registry_hash=manifest["split_registry_hash"],
        realized_training_flops=realized_training_flops,
        training_examples_seen=training_examples_seen,
    )


def _prefix_compute(model: RecurrentReasoner, schedule, stop: int) -> tuple[int, int]:
    prefix = schedule[:stop]
    return (
        estimated_schedule_training_flops(model, prefix) if prefix else 0,
        sum(len(batch.task_ids) for batch in prefix),
    )


def train_arm_seed(
    root: Path,
    run_directory: Path,
    *,
    run_id: str,
    arm_id: str,
    seed: int,
    manifest: dict[str, Any],
    resume: bool,
) -> tuple[RecurrentReasoner, dict[str, Any], RunComputeRecord]:
    training_path = root / "contracts" / "wamrx_recurrent_training_v1.json"
    split_path = root / "contracts" / "wamrx_recurrent_splits.json"
    model_config = RecurrentModelConfig.load(training_path)
    optimizer_config = OptimizerConfig.load(training_path)
    registry = load_split_registry(split_path)
    tasks = training_tasks(registry["splits"]["train"])
    schedule = build_schedule(tasks, optimizer_config, seed=seed)
    frozen_schedule_hash = schedule_hash(schedule)
    estimated_flops = None
    codec = ByteCodec(
        maximum_input_tokens=model_config.maximum_input_tokens,
        maximum_output_tokens=model_config.maximum_output_tokens,
    )
    mx.random.seed(seed)
    model = RecurrentReasoner(arm_id, model_config)
    optimizer = create_optimizer(optimizer_config, total_updates=len(schedule))
    estimated_flops = estimated_schedule_training_flops(model, schedule)

    checkpoint_directory = run_directory / "checkpoints" / f"seed-{seed}" / arm_id
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    start = 0
    latest_report = None
    candidates = sorted(checkpoint_directory.glob("update-*.safetensors"))
    if candidates and not resume:
        raise RecurrentComparisonError(
            "checkpoints already exist; use --resume or a new run directory"
        )
    if resume:
        if candidates:
            latest = candidates[-1]
            try:
                start = int(latest.stem.removeprefix("update-"))
            except ValueError as error:
                raise RecurrentCheckpointError(
                    f"invalid checkpoint filename {latest.name!r}"
                ) from error
            prefix_flops, prefix_examples = _prefix_compute(model, schedule, start)
            expected = _checkpoint_metadata(
                manifest,
                run_id=run_id,
                arm_id=arm_id,
                seed=seed,
                completed_updates=start,
                frozen_schedule_hash=frozen_schedule_hash,
                realized_training_flops=prefix_flops,
                training_examples_seen=prefix_examples,
            )
            latest_report = load_checkpoint(
                latest, model, optimizer, expected=expected
            )

    checkpoint_frequency = 50
    reports = []
    for stop in range(start + checkpoint_frequency, len(schedule) + checkpoint_frequency, checkpoint_frequency):
        stop = min(stop, len(schedule))
        if stop <= start:
            continue
        report = train_schedule_segment(
            model,
            codec,
            tasks,
            schedule,
            optimizer_config,
            optimizer=optimizer,
            start_update=start,
            stop_update=stop,
        )
        reports.append(report)
        cumulative_flops, cumulative_examples = _prefix_compute(model, schedule, stop)
        metadata = _checkpoint_metadata(
            manifest,
            run_id=run_id,
            arm_id=arm_id,
            seed=seed,
            completed_updates=stop,
            frozen_schedule_hash=frozen_schedule_hash,
            realized_training_flops=cumulative_flops,
            training_examples_seen=cumulative_examples,
        )
        latest_report = save_checkpoint(
            checkpoint_directory / f"update-{stop:04d}.safetensors",
            model,
            optimizer,
            metadata,
        )
        start = stop
        if stop == len(schedule):
            break
    if start != len(schedule) or latest_report is None:
        raise RecurrentComparisonError("training did not reach the final checkpoint")
    realized_flops, examples_seen = _prefix_compute(model, schedule, len(schedule))
    compute = RunComputeRecord(
        run_id=run_id,
        arm_id=arm_id,
        seed=seed,
        status="INCOMPLETE",
        parameter_count=parameter_count(model),
        schedule_hash=frozen_schedule_hash,
        estimated_training_flops=estimated_flops,
        realized_training_flops=realized_flops,
        estimated_inference_flops=0,
        realized_inference_flops=0,
        training_examples_planned=len(schedule) * optimizer_config.batch_size,
        training_examples_seen=examples_seen,
        optimizer_updates_planned=len(schedule),
        optimizer_updates_completed=len(schedule),
        evaluation_examples_planned=0,
        evaluation_examples_completed=0,
    )
    compute.validate()
    return model, {
        "checkpoint": latest_report,
        "segments": reports,
        "schedule_hash": frozen_schedule_hash,
        "estimated_training_flops": estimated_flops,
        "realized_training_flops": realized_flops,
        "examples_seen": examples_seen,
        "optimizer_updates": len(schedule),
    }, compute


def maximum_feasible_macro_depth(
    model: RecurrentReasoner,
    registered_depths: Iterable[int],
    *,
    micro_steps: int,
    readout_every_position: bool,
) -> int:
    feasible = [
        depth
        for depth in registered_depths
        if (
            model.estimated_training_forward_flops(
                input_tokens=model.config.maximum_input_tokens,
                macro_steps=depth,
                micro_steps=micro_steps,
            )
            if readout_every_position
            else model.estimated_inference_flops(
                input_tokens=model.config.maximum_input_tokens,
                macro_steps=depth,
                micro_steps=micro_steps,
            )
        )
        <= model.config.maximum_inference_flops
    ]
    if not feasible:
        raise RecurrentComparisonError("no registered depth fits the common FLOP cap")
    if model.arm_id == "fixed-depth-v1":
        return model.config.fixed_reasoning_blocks
    return max(feasible)


def _journal_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["arm_id"],
        record["seed"],
        record["task_id"],
        record["mode"],
        record["requested_macro_depth"],
        record["micro_steps"],
    )


def _load_journal(path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    if not path.exists():
        return {}
    records = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line:
            continue
        record = json.loads(line)
        record_hash = record.pop("record_hash", None)
        if record_hash != sha256_json(record):
            raise RecurrentComparisonError(
                f"evaluation journal hash mismatch at line {line_number}"
            )
        key = _journal_key(record)
        if key in records:
            raise RecurrentComparisonError("duplicate evaluation journal key")
        records[key] = record
    return records


def _append_journal(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    durable = {**record, "record_hash": sha256_json(record)}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(durable) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _evaluation_tasks(registry: dict[str, Any]) -> tuple[RecurrentTask, ...]:
    return tuple(
        task
        for split in ("id", "ood")
        for family in FAMILIES
        for task in generate_family(split, family, registry["splits"][split])
    )


def evaluate_arm_seed(
    root: Path,
    run_directory: Path,
    model: RecurrentReasoner,
    *,
    seed: int,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    training_path = root / "contracts" / "wamrx_recurrent_training_v1.json"
    split_path = root / "contracts" / "wamrx_recurrent_splits.json"
    reasoner_path = root / "contracts" / "wamrx_recurrent_reasoner.json"
    registry = load_split_registry(split_path)
    reasoner_contract = json.loads(reasoner_path.read_text())
    optimizer_config = OptimizerConfig.load(training_path)
    codec = ByteCodec(
        maximum_input_tokens=model.config.maximum_input_tokens,
        maximum_output_tokens=model.config.maximum_output_tokens,
    )
    tasks = _evaluation_tasks(registry)
    depths = tuple(reasoner_contract["depth_protocol"]["evaluation_macro_depths"])
    primary_depth = maximum_feasible_macro_depth(
        model, depths, micro_steps=1, readout_every_position=False
    )
    adaptive_depth = maximum_feasible_macro_depth(
        model, depths, micro_steps=1, readout_every_position=True
    )
    budget = ComputeBudget(
        maximum_inference_flops=model.config.maximum_inference_flops,
        maximum_macro_steps=reasoner_contract["depth_protocol"]["maximum_macro_steps"],
        maximum_micro_steps_per_macro=reasoner_contract["depth_protocol"]["maximum_micro_steps_per_macro"],
        maximum_total_micro_steps=reasoner_contract["depth_protocol"]["maximum_total_micro_steps"],
        maximum_retrieval_calls=4,
        maximum_tool_calls=4,
    )
    journal_path = (
        run_directory / "evaluation" / f"seed-{seed}" / f"{model.arm_id}.jsonl"
    )
    existing = _load_journal(journal_path) if resume else {}
    if journal_path.exists() and not resume:
        raise RecurrentComparisonError(
            f"evaluation journal already exists without --resume: {journal_path}"
        )
    plans = [("final_depth", depth) for depth in depths]
    plans.extend(
        [("maximum_depth", adaptive_depth), ("adaptive", adaptive_depth)]
    )
    records = dict(existing)
    for mode, requested_depth in plans:
        for task in tasks:
            key = (
                model.arm_id,
                seed,
                task.task_id,
                mode,
                requested_depth,
                1,
            )
            if key in records:
                continue
            output, trace, compute = execute_task(
                model,
                codec,
                task,
                budget=budget,
                maximum_answer_instability=reasoner_contract["promotion_thresholds"]["maximum_answer_instability_for_resolved_halt"],
                maximum_macro_steps=requested_depth,
                micro_steps=1,
                execution_policy=mode,
            )
            trace.validate()
            output.validate(trace)
            compute.validate(budget)
            journal_valid = all(
                operation.journal_id
                for step in trace.steps
                for operation in step.operations
            )
            if not journal_valid:
                raise RecurrentComparisonError("unjournaled evidence/tool operation")
            record = {
                "arm_id": model.arm_id,
                "seed": seed,
                "task_id": task.task_id,
                "family": task.family,
                "split": task.split,
                "protected_regions": list(task.protected_regions),
                "mode": mode,
                "requested_macro_depth": requested_depth,
                "micro_steps": 1,
                "exact": output.answer == task.expected,
                "inference_flops": compute.inference_flops,
                "executed_macro_steps": compute.macro_steps,
                "executed_micro_steps": compute.micro_steps,
                "trace_hash": trace.trace_hash,
                "trace_valid": True,
                "journal_valid": journal_valid,
                "halt_reason": output.halt_reason,
                "evidence_bundle_hash": output.evidence_bundle_hash,
                "supporting_event_ids": list(output.support.supporting_event_ids),
            }
            _append_journal(journal_path, record)
            records[key] = record
    values = list(records.values())
    expected_count = len(tasks) * len(plans)
    if len(values) != expected_count:
        raise RecurrentComparisonError(
            f"evaluation journal is incomplete: {len(values)} of {expected_count}"
        )
    return values, {
        "journal_path": str(journal_path),
        "journal_sha256": file_sha256(journal_path),
        "records": len(values),
        "primary_macro_depth": primary_depth,
        "adaptive_macro_depth": adaptive_depth,
        "estimated_inference_flops": sum(
            (
                model.estimated_inference_flops(
                    input_tokens=model.config.maximum_input_tokens,
                    macro_steps=record["requested_macro_depth"],
                    micro_steps=record["micro_steps"],
                )
                if record["mode"] == "final_depth"
                else model.estimated_training_forward_flops(
                    input_tokens=model.config.maximum_input_tokens,
                    macro_steps=record["requested_macro_depth"],
                    micro_steps=record["micro_steps"],
                )
            )
            for record in values
        ),
        "realized_inference_flops": sum(record["inference_flops"] for record in values),
    }


def run_post_training_manipulations(
    model: RecurrentReasoner,
    codec: ByteCodec,
    task: RecurrentTask,
    *,
    schedule,
) -> dict[str, bool]:
    problem_tokens, _ = codec.encode_problem(task.problem)
    tokens = mx.array([problem_tokens], dtype=mx.int32)
    context = model.encode(tokens)
    zero = mx.zeros_like(context)
    normal = model.flat_step(context, zero)
    no_reinjection = model.flat_step(mx.zeros_like(context), zero)
    mx.eval(normal, no_reinjection)
    m1 = sha256_json(normal.tolist()) != sha256_json(no_reinjection.tolist())

    fixed_schedule = tuple(
        dataclasses.replace(batch, macro_steps=4, micro_steps=1) for batch in schedule
    )
    m2 = schedule_hash(fixed_schedule) != schedule_hash(schedule) and len(
        {batch.macro_steps for batch in fixed_schedule}
    ) == 1

    corrupted = model.flat_step(context, mx.ones_like(context))
    mx.eval(corrupted)
    m3 = sha256_json(corrupted.tolist()) != sha256_json(normal.tolist())

    residual = UnresolvedConstraintState(
        unanswered_constraints=("forced-disagreement",), answer_instability=1.0
    )
    m4 = (
        decide_halt(
            "stop", residual, budget_exhausted=False, maximum_answer_instability=0.02
        )
        == "continue"
    )
    m5 = (
        decide_halt(
            "stop",
            UnresolvedConstraintState(
                conflicting_evidence=("contradictory-row",), answer_instability=0.0
            ),
            budget_exhausted=False,
            maximum_answer_instability=0.02,
        )
        == "continue"
    )

    depth_four = model(tokens, macro_steps=4, micro_steps=1)["final_state"]
    depth_six = model(tokens, macro_steps=6, micro_steps=1)["final_state"]
    mx.eval(depth_four, depth_six)
    m6 = sha256_json(depth_four.tolist()) != sha256_json(depth_six.tolist())

    bundle = task_evidence_bundle(task)
    stale = dataclasses.replace(
        bundle,
        views=(
            dataclasses.replace(bundle.views[0], compatible_with_runtime=False),
            *bundle.views[1:],
        ),
    )
    try:
        stale.validate(required_regions=task.protected_regions)
        m7 = False
    except RecurrentContractError:
        m7 = True
    removed = EvidenceBundle(
        bundle_id=bundle.bundle_id,
        problem_hash=bundle.problem_hash,
        views=bundle.views[:-1],
        active_ontology_version=bundle.active_ontology_version,
        active_runtime_id=bundle.active_runtime_id,
    )
    try:
        removed.validate(required_regions=task.protected_regions)
        m8 = len(bundle.views) == 1
    except RecurrentContractError:
        m8 = True
    return {
        "M1_disable_reinjection_changes_state": m1,
        "M2_fixed_loop_counterfactual_changes_schedule": m2,
        "M3_corrupt_state_changes_recurrence": m3,
        "M4_external_halt_blocks_premature_stop": m4,
        "M5_contradiction_keeps_residual_active": m5,
        "M6_unseen_depth_changes_state": m6,
        "M7_stale_artifact_fails_closed": m7,
        "M8_missing_protected_region_fails_closed": m8,
    }


def aggregate_evaluation(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    family_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    region_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        common = (
            record["arm_id"],
            record["seed"],
            record["split"],
            record["mode"],
            record["requested_macro_depth"],
        )
        family_groups[(*common, "family", record["family"])].append(record)
        for region in record["protected_regions"]:
            region_groups[(*common, "region", region)].append(record)
    rows = []
    for key, values in sorted({**family_groups, **region_groups}.items()):
        arm, seed, split, mode, depth, grouping, label = key
        correct = sum(bool(value["exact"]) for value in values)
        rows.append(
            {
                "arm_id": arm,
                "seed": seed,
                "split": split,
                "mode": mode,
                "requested_macro_depth": depth,
                "grouping": grouping,
                "label": label,
                "correct": correct,
                "examples": len(values),
                "accuracy": correct / len(values),
                "total_inference_flops": sum(value["inference_flops"] for value in values),
                "median_inference_flops": statistics.median(
                    value["inference_flops"] for value in values
                ),
            }
        )
    return rows


def training_compute_normalized_control(
    root: Path,
    run_directory: Path,
    *,
    run_id: str,
    manifest: dict[str, Any],
    seeds: tuple[int, ...],
    primary_depths: dict[str, int],
    final_compute_records: list[dict[str, Any]],
    resume: bool,
) -> dict[str, Any]:
    training_path = root / "contracts" / "wamrx_recurrent_training_v1.json"
    split_path = root / "contracts" / "wamrx_recurrent_splits.json"
    reasoner_contract = json.loads(
        (root / "contracts" / "wamrx_recurrent_reasoner.json").read_text()
    )
    model_config = RecurrentModelConfig.load(training_path)
    optimizer_config = OptimizerConfig.load(training_path)
    registry = load_split_registry(split_path)
    train_tasks = training_tasks(registry["splits"]["train"])
    ood_tasks = tuple(
        task
        for family in FAMILIES
        for task in generate_family("ood", family, registry["splits"]["ood"])
    )
    codec = ByteCodec(
        maximum_input_tokens=model_config.maximum_input_tokens,
        maximum_output_tokens=model_config.maximum_output_tokens,
    )
    budget = ComputeBudget(
        maximum_inference_flops=model_config.maximum_inference_flops,
        maximum_macro_steps=12,
        maximum_micro_steps_per_macro=4,
        maximum_total_micro_steps=48,
        maximum_retrieval_calls=4,
        maximum_tool_calls=4,
    )
    final_by_seed_arm = {
        (record["seed"], record["arm_id"]): record
        for record in final_compute_records
    }
    selections = []
    family_accuracy: dict[tuple[int, str, str], float] = {}
    for seed in seeds:
        target = min(
            final_by_seed_arm[(seed, arm_id)]["realized_training_flops"]
            for arm_id in ARM_IDS
        )
        schedule = build_schedule(train_tasks, optimizer_config, seed=seed)
        frozen_schedule_hash = schedule_hash(schedule)
        for arm_id in ARM_IDS:
            mx.random.seed(seed)
            model = RecurrentReasoner(arm_id, model_config)
            candidates = []
            for update in range(50, len(schedule) + 1, 50):
                flops, examples = _prefix_compute(model, schedule, update)
                if flops <= target:
                    candidates.append((update, flops, examples))
            if not candidates:
                raise RecurrentComparisonError(
                    "no 50-update checkpoint fits the training-compute control"
                )
            selected_update, selected_flops, selected_examples = max(candidates)
            optimizer = create_optimizer(
                optimizer_config, total_updates=len(schedule)
            )
            metadata = _checkpoint_metadata(
                manifest,
                run_id=run_id,
                arm_id=arm_id,
                seed=seed,
                completed_updates=selected_update,
                frozen_schedule_hash=frozen_schedule_hash,
                realized_training_flops=selected_flops,
                training_examples_seen=selected_examples,
            )
            checkpoint_path = (
                run_directory
                / "checkpoints"
                / f"seed-{seed}"
                / arm_id
                / f"update-{selected_update:04d}.safetensors"
            )
            checkpoint_report = load_checkpoint(
                checkpoint_path, model, optimizer, expected=metadata
            )
            journal_path = (
                run_directory
                / "compute-normalized-training"
                / f"seed-{seed}"
                / f"{arm_id}-update-{selected_update:04d}.jsonl"
            )
            existing = _load_journal(journal_path) if resume else {}
            if journal_path.exists() and not resume:
                raise RecurrentComparisonError(
                    "training-compute control journal exists without --resume"
                )
            records = dict(existing)
            for task in ood_tasks:
                key = (
                    arm_id,
                    seed,
                    task.task_id,
                    "training_compute_normalized",
                    primary_depths[arm_id],
                    1,
                )
                if key in records:
                    continue
                output, trace, compute = execute_task(
                    model,
                    codec,
                    task,
                    budget=budget,
                    maximum_answer_instability=reasoner_contract["promotion_thresholds"]["maximum_answer_instability_for_resolved_halt"],
                    maximum_macro_steps=primary_depths[arm_id],
                    micro_steps=1,
                    execution_policy="final_depth",
                )
                trace.validate()
                output.validate(trace)
                compute.validate(budget)
                record = {
                    "arm_id": arm_id,
                    "seed": seed,
                    "task_id": task.task_id,
                    "family": task.family,
                    "split": task.split,
                    "protected_regions": list(task.protected_regions),
                    "mode": "training_compute_normalized",
                    "requested_macro_depth": primary_depths[arm_id],
                    "micro_steps": 1,
                    "exact": output.answer == task.expected,
                    "inference_flops": compute.inference_flops,
                    "executed_macro_steps": compute.macro_steps,
                    "executed_micro_steps": compute.micro_steps,
                    "trace_hash": trace.trace_hash,
                    "trace_valid": True,
                    "journal_valid": all(
                        operation.journal_id
                        for step in trace.steps
                        for operation in step.operations
                    ),
                    "halt_reason": output.halt_reason,
                    "evidence_bundle_hash": output.evidence_bundle_hash,
                    "supporting_event_ids": list(
                        output.support.supporting_event_ids
                    ),
                    "checkpoint_update": selected_update,
                    "checkpoint_sha256": checkpoint_report["file_sha256"],
                }
                if not record["journal_valid"]:
                    raise RecurrentComparisonError(
                        "unjournaled training-compute control operation"
                    )
                _append_journal(journal_path, record)
                records[key] = record
            if len(records) != len(ood_tasks):
                raise RecurrentComparisonError(
                    "training-compute control evaluation is incomplete"
                )
            for family in FAMILIES:
                values = [
                    record
                    for record in records.values()
                    if record["family"] == family
                ]
                family_accuracy[(seed, arm_id, family)] = sum(
                    record["exact"] for record in values
                ) / len(values)
            selections.append(
                {
                    "seed": seed,
                    "arm_id": arm_id,
                    "target_training_flops": target,
                    "selected_update": selected_update,
                    "selected_training_flops": selected_flops,
                    "unused_training_flops": target - selected_flops,
                    "training_examples_seen": selected_examples,
                    "checkpoint_sha256": checkpoint_report["file_sha256"],
                    "evaluation_journal_sha256": file_sha256(journal_path),
                }
            )
    tests = {}
    for arm_id in ("flat-recurrent-v1", "hierarchical-recurrent-v1"):
        deltas = []
        for seed in seeds:
            candidate = sum(
                family_accuracy[(seed, arm_id, family)] for family in FAMILIES
            ) / len(FAMILIES)
            fixed = sum(
                family_accuracy[(seed, "fixed-depth-v1", family)]
                for family in FAMILIES
            ) / len(FAMILIES)
            deltas.append(candidate - fixed)
        tests[arm_id] = paired_student_t(
            deltas, null_margin=0.0, alpha=0.05
        )
    return {
        "rule": "latest 50-update checkpoint not exceeding the smallest final-arm realized training FLOPs within each seed",
        "selections": selections,
        "ood_tests": tests,
    }


def _row_index(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {
        (
            row["arm_id"],
            row["seed"],
            row["split"],
            row["mode"],
            row["requested_macro_depth"],
            row["grouping"],
            row["label"],
        ): row
        for row in rows
    }


def promotion_audit(
    rows: list[dict[str, Any]],
    *,
    seeds: tuple[int, ...],
    primary_depths: dict[str, int],
    adaptive_depths: dict[str, int],
    manipulations: dict[str, bool],
    training_compute_control: dict[str, Any],
) -> dict[str, Any]:
    index = _row_index(rows)

    def accuracy(arm: str, seed: int, split: str, family: str, mode: str = "final_depth", depth: int | None = None) -> float:
        selected_depth = primary_depths[arm] if depth is None else depth
        key = (arm, seed, split, mode, selected_depth, "family", family)
        if key not in index:
            raise RecurrentComparisonError(f"missing primary metric {key}")
        return float(index[key]["accuracy"])

    def pooled(arm: str, seed: int, split: str, mode: str = "final_depth", depth: int | None = None) -> float:
        return sum(accuracy(arm, seed, split, family, mode, depth) for family in FAMILIES) / len(FAMILIES)

    recurrent = ("flat-recurrent-v1", "hierarchical-recurrent-v1")
    family_superiority = {}
    for arm in recurrent:
        for family in FAMILIES:
            family_superiority[f"{arm}:{family}"] = (
                [
                    accuracy(arm, seed, "ood", family)
                    - accuracy("fixed-depth-v1", seed, "ood", family)
                    for seed in seeds
                ],
                0.0,
            )
    family_tests = holm_adjusted_tests(family_superiority)
    overall_ood = {
        arm: paired_student_t(
            [
                pooled(arm, seed, "ood")
                - pooled("fixed-depth-v1", seed, "ood")
                for seed in seeds
            ],
            null_margin=0.0,
            alpha=0.05,
        )
        for arm in recurrent
    }
    id_tests = holm_adjusted_tests(
        {
            arm: (
                [
                    pooled(arm, seed, "id")
                    - pooled("fixed-depth-v1", seed, "id")
                    for seed in seeds
                ],
                -0.02,
            )
            for arm in recurrent
        }
    )
    region_hypotheses = {}
    for arm in recurrent:
        for region in ("algorithmic", "structured", "finance", "operations"):
            deltas = []
            for seed in seeds:
                accuracies = {}
                for selected_arm in (arm, "fixed-depth-v1"):
                    selected_rows = []
                    for split in ("id", "ood"):
                        key = (
                            selected_arm,
                            seed,
                            split,
                            "final_depth",
                            primary_depths[selected_arm],
                            "region",
                            region,
                        )
                        if key not in index:
                            raise RecurrentComparisonError(
                                "protected-region metric is missing"
                            )
                        selected_rows.append(index[key])
                    correct = sum(row["correct"] for row in selected_rows)
                    examples = sum(row["examples"] for row in selected_rows)
                    accuracies[selected_arm] = correct / examples
                deltas.append(accuracies[arm] - accuracies["fixed-depth-v1"])
            region_hypotheses[f"{arm}:{region}"] = (deltas, -0.02)
    region_tests = holm_adjusted_tests(region_hypotheses)

    hierarchy_family = {
        family: (
            [
                accuracy("hierarchical-recurrent-v1", seed, "ood", family)
                - accuracy("flat-recurrent-v1", seed, "ood", family)
                for seed in seeds
            ],
            0.0,
        )
        for family in FAMILIES
    }
    hierarchy_tests = holm_adjusted_tests(hierarchy_family)
    hierarchy_noninferiority = holm_adjusted_tests(
        {
            family: (values, -0.02)
            for family, (values, _) in hierarchy_family.items()
        }
    )

    depth_declines = {}
    available_depths = sorted(
        {
            row["requested_macro_depth"]
            for row in rows
            if row["mode"] == "final_depth"
        }
    )
    for arm in recurrent:
        comparisons = []
        for depth in available_depths:
            if depth <= 4 or depth > primary_depths[arm]:
                continue
            for family in FAMILIES:
                comparisons.extend(
                    accuracy(arm, seed, "ood", family, depth=depth)
                    - accuracy(arm, seed, "ood", family, depth=4)
                    for seed in seeds
                )
        depth_declines[arm] = min(comparisons) if comparisons else 0.0

    adaptive = {}
    for arm in recurrent:
        accuracy_losses = []
        compute_savings = []
        for seed in seeds:
            for split in ("id", "ood"):
                for family in FAMILIES:
                    maximum_key = (
                        arm,
                        seed,
                        split,
                        "maximum_depth",
                        adaptive_depths[arm],
                        "family",
                        family,
                    )
                    adaptive_key = (
                        arm,
                        seed,
                        split,
                        "adaptive",
                        adaptive_depths[arm],
                        "family",
                        family,
                    )
                    maximum = index[maximum_key]
                    adapted = index[adaptive_key]
                    accuracy_losses.append(maximum["accuracy"] - adapted["accuracy"])
                    compute_savings.append(
                        1.0
                        - adapted["median_inference_flops"]
                        / maximum["median_inference_flops"]
                    )
        adaptive[arm] = {
            "maximum_accuracy_loss": max(accuracy_losses),
            "median_compute_saving": statistics.median(compute_savings),
        }

    compute_normalized = {}
    for arm in recurrent:
        deltas = []
        selected_depths = []
        for seed in seeds:
            candidate_family = []
            fixed_family = []
            for family in FAMILIES:
                fixed_key = (
                    "fixed-depth-v1",
                    seed,
                    "ood",
                    "final_depth",
                    primary_depths["fixed-depth-v1"],
                    "family",
                    family,
                )
                fixed_row = index[fixed_key]
                candidates = [
                    row
                    for row in rows
                    if row["arm_id"] == arm
                    and row["seed"] == seed
                    and row["split"] == "ood"
                    and row["mode"] == "final_depth"
                    and row["grouping"] == "family"
                    and row["label"] == family
                    and row["median_inference_flops"]
                    <= fixed_row["median_inference_flops"]
                ]
                if not candidates:
                    raise RecurrentComparisonError("no compute-normalized depth exists")
                selected = max(
                    candidates,
                    key=lambda row: (
                        row["median_inference_flops"],
                        -row["requested_macro_depth"],
                    ),
                )
                selected_depths.append(selected["requested_macro_depth"])
                candidate_family.append(selected["accuracy"])
                fixed_family.append(fixed_row["accuracy"])
            deltas.append(
                sum(candidate_family) / len(candidate_family)
                - sum(fixed_family) / len(fixed_family)
            )
        compute_normalized[arm] = {
            "test": paired_student_t(deltas, null_margin=0.0, alpha=0.05),
            "selected_depths": selected_depths,
        }

    family_positive_counts = {
        arm: sum(
            family_tests["hypotheses"][f"{arm}:{family}"]["holm_rejects_null"]
            for family in FAMILIES
        )
        for arm in recurrent
    }
    recurrent_eligible = {}
    for arm in recurrent:
        recurrent_eligible[arm] = (
            overall_ood[arm]["one_sided_lower_bound"] > 0.0
            and id_tests["hypotheses"][arm]["one_sided_lower_bound"] >= -0.02
            and all(
                region_tests["hypotheses"][f"{arm}:{region}"]["one_sided_lower_bound"]
                >= -0.02
                for region in ("algorithmic", "structured", "finance", "operations")
            )
            and depth_declines[arm] >= -0.01
            and adaptive[arm]["median_compute_saving"] >= 0.20
            and adaptive[arm]["maximum_accuracy_loss"] <= 0.01
            and family_positive_counts[arm] >= 2
            and all(manipulations.values())
        )
    hierarchy_wins = sum(
        hierarchy_tests["hypotheses"][family]["holm_rejects_null"]
        for family in FAMILIES
    )
    hierarchy_noninferior = all(
        hierarchy_noninferiority["hypotheses"][family]["one_sided_lower_bound"]
        >= -0.02
        for family in FAMILIES
    )
    primary_candidate = None
    if recurrent_eligible["hierarchical-recurrent-v1"] and hierarchy_wins >= 1 and hierarchy_noninferior:
        primary_candidate = "hierarchical-recurrent-v1"
    elif recurrent_eligible["flat-recurrent-v1"]:
        primary_candidate = "flat-recurrent-v1"

    scaling_reasons = []
    if primary_candidate is not None and not training_compute_control["ood_tests"][primary_candidate]["passes"]:
        scaling_reasons.append("training_compute")
    if primary_candidate is not None and not compute_normalized[primary_candidate]["test"]["passes"]:
        scaling_reasons.append("test_time_compute")
    if primary_candidate is not None and scaling_reasons:
        decision = "COMPLETE_COMPUTE_SCALING_ONLY"
    elif primary_candidate == "hierarchical-recurrent-v1":
        decision = "COMPLETE_ADOPT_HIERARCHY"
    elif primary_candidate == "flat-recurrent-v1":
        decision = "COMPLETE_ADOPT_FLAT"
    elif any(family_positive_counts[arm] == 1 for arm in recurrent):
        decision = "COMPLETE_SPECIALIST_ONLY"
    else:
        decision = "COMPLETE_RETAIN_FIXED"

    return {
        "statistical_procedure": "one-sided paired Student-t; Holm step-down within preregistered families",
        "overall_ood": overall_ood,
        "family_superiority": family_tests,
        "id_noninferiority": id_tests,
        "protected_region_noninferiority": region_tests,
        "hierarchy_vs_flat_superiority": hierarchy_tests,
        "hierarchy_vs_flat_noninferiority": hierarchy_noninferiority,
        "depth_declines": depth_declines,
        "adaptive_halting": adaptive,
        "compute_normalized_secondary": {
            "training": training_compute_control,
            "inference": compute_normalized,
            "scaling_reasons": scaling_reasons,
        },
        "family_positive_counts": family_positive_counts,
        "recurrent_eligible": recurrent_eligible,
        "manipulations": manipulations,
        "decision": decision,
    }


def run_registered_comparison(
    root: Path,
    run_directory: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    run_contract = json.loads(
        (root / "contracts" / "wamrx_recurrent_run_v1.json").read_text()
    )
    training_config = RecurrentModelConfig.load(
        root / "contracts" / "wamrx_recurrent_training_v1.json"
    )
    optimizer_config = OptimizerConfig.load(
        root / "contracts" / "wamrx_recurrent_training_v1.json"
    )
    split_registry = load_split_registry(
        root / "contracts" / "wamrx_recurrent_splits.json"
    )
    run_id = "e0.14-registered-recurrent-comparison-v1"
    manifest = build_run_manifest(root)
    run_directory.mkdir(parents=True, exist_ok=True)
    state_path = run_directory / "run-state.json"
    state = {
        "experiment": "E0.14",
        "run_id": run_id,
        "status": "INCOMPLETE",
        "comparison_label": run_contract["comparison_label"],
        "manifest": manifest,
        "completed_arm_seeds": [],
        "invalid_reasons": [],
        "incomplete_reasons": [],
    }
    _atomic_json(state_path, state)

    all_records = []
    training_reports = []
    compute_records = []
    manipulation_reports = []
    primary_depths: dict[str, int] = {}
    adaptive_depths: dict[str, int] = {}
    try:
        training_task_values = training_tasks(split_registry["splits"]["train"])
        for seed in optimizer_config.paired_seeds:
            frozen_schedule = build_schedule(
                training_task_values, optimizer_config, seed=seed
            )
            for arm_id in ARM_IDS:
                model, training_report, compute = train_arm_seed(
                    root,
                    run_directory,
                    run_id=run_id,
                    arm_id=arm_id,
                    seed=seed,
                    manifest=manifest,
                    resume=resume,
                )
                training_reports.append(
                    {"arm_id": arm_id, "seed": seed, **training_report}
                )
                records, evaluation_report = evaluate_arm_seed(
                    root, run_directory, model, seed=seed, resume=resume
                )
                all_records.extend(records)
                primary_depths[arm_id] = evaluation_report["primary_macro_depth"]
                adaptive_depths[arm_id] = evaluation_report["adaptive_macro_depth"]
                complete_compute = dataclasses.replace(
                    compute,
                    status="COMPLETE",
                    estimated_inference_flops=evaluation_report[
                        "estimated_inference_flops"
                    ],
                    realized_inference_flops=evaluation_report[
                        "realized_inference_flops"
                    ],
                    evaluation_examples_planned=evaluation_report["records"],
                    evaluation_examples_completed=evaluation_report["records"],
                )
                complete_compute.validate()
                compute_records.append(complete_compute.to_dict())
                codec = ByteCodec(
                    maximum_input_tokens=training_config.maximum_input_tokens,
                    maximum_output_tokens=training_config.maximum_output_tokens,
                )
                manipulation_task = generate_family(
                    "ood", "algorithmic", split_registry["splits"]["ood"]
                )[0]
                if arm_id != "fixed-depth-v1":
                    manipulation_reports.append(
                        run_post_training_manipulations(
                            model,
                            codec,
                            manipulation_task,
                            schedule=frozen_schedule,
                        )
                    )
                state["completed_arm_seeds"].append(
                    {"arm_id": arm_id, "seed": seed}
                )
                _atomic_json(state_path, state)
        manipulations = {
            key: all(report[key] for report in manipulation_reports)
            for key in manipulation_reports[0]
        }
        metric_rows = aggregate_evaluation(all_records)
        training_compute_control = training_compute_normalized_control(
            root,
            run_directory,
            run_id=run_id,
            manifest=manifest,
            seeds=optimizer_config.paired_seeds,
            primary_depths=primary_depths,
            final_compute_records=compute_records,
            resume=resume,
        )
        audit = promotion_audit(
            metric_rows,
            seeds=optimizer_config.paired_seeds,
            primary_depths=primary_depths,
            adaptive_depths=adaptive_depths,
            manipulations=manipulations,
            training_compute_control=training_compute_control,
        )
        status = resolve_terminal_status(
            invalid_reasons=[],
            incomplete_reasons=[],
            decision=audit["decision"],
        )
        result = {
            **state,
            "status": status,
            "primary_depths": primary_depths,
            "adaptive_depths": adaptive_depths,
            "training_reports": training_reports,
            "compute_records": compute_records,
            "metric_rows": metric_rows,
            "promotion_audit": audit,
        }
        _atomic_json(state_path, result)
        return result
    except (KeyboardInterrupt, MemoryError) as error:
        state["incomplete_reasons"].append(type(error).__name__)
        state["status"] = "INCOMPLETE"
        _atomic_json(state_path, state)
        raise
    except (
        RecurrentCheckpointError,
        RecurrentComparisonError,
        RecurrentContractError,
        FloatingPointError,
        ValueError,
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
