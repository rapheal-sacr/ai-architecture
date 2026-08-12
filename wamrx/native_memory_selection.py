"""Fail-closed post-run selection for the E0.16 memory configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .native_memory_run import ARM_IDS


SELECTION_SCHEMA_VERSION = 1
SELECTION_ID = "wamrx-native-memory-selection-v1"
TERMINAL_STATUS = "COMPLETE_RETAIN_EXPLICIT_MULTIVIEW"
SELECTED_ARM = "explicit-multiview-v1"
BLOCKED_AUTHORIZATIONS = (
    "deterministic_session_cache",
    "learned_native_memory",
    "mixture_of_experts",
    "adapter_consolidation",
    "self_improvement",
)


class NativeMemorySelectionError(ValueError):
    """The selected memory configuration does not match terminal evidence."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeMemorySelectionError(f"{field} must be an object")
    return value


def validate_selection(
    selection: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    result_sha256: str,
) -> dict[str, Any]:
    if selection.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise NativeMemorySelectionError("unsupported native-memory selection schema")
    if selection.get("selection_id") != SELECTION_ID:
        raise NativeMemorySelectionError("unexpected native-memory selection identity")

    source = _mapping(selection.get("selected_from"), "selected_from")
    if source.get("experiment") != "E0.16" or result.get("experiment") != "E0.16":
        raise NativeMemorySelectionError("selection must be sourced from E0.16")
    if source.get("result_sha256") != result_sha256:
        raise NativeMemorySelectionError("E0.16 result hash does not match selection")
    if source.get("manifest") != result.get("manifest"):
        raise NativeMemorySelectionError("E0.16 frozen manifest does not match selection")
    if source.get("run_id") != result.get("run_id"):
        raise NativeMemorySelectionError("E0.16 run identity does not match selection")

    status = result.get("status")
    audit = _mapping(result.get("promotion_audit"), "promotion_audit")
    if (
        status != TERMINAL_STATUS
        or source.get("terminal_status") != status
        or audit.get("decision") != status
    ):
        raise NativeMemorySelectionError("terminal E0.16 decision is not explicit multiview")
    if result.get("invalid_reasons") or result.get("incomplete_reasons"):
        raise NativeMemorySelectionError("terminal E0.16 contains failure reasons")

    seeds = selection.get("registered_seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 5
        or len(set(seeds)) != 5
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or result.get("completed_seeds") != seeds
    ):
        raise NativeMemorySelectionError("E0.16 does not contain five exact paired seeds")

    evaluation_rows = result.get("evaluation_rows")
    if not isinstance(evaluation_rows, list):
        raise NativeMemorySelectionError("E0.16 evaluation rows are missing")
    expected_rows = len(seeds) * len(ARM_IDS) * 2 * 36
    if (
        selection.get("completed_evaluation_rows") != expected_rows
        or len(evaluation_rows) != expected_rows
    ):
        raise NativeMemorySelectionError("E0.16 evaluation row count is incomplete")
    keys = [
        (
            row.get("manifest_hash"),
            row.get("seed"),
            row.get("arm_id"),
            row.get("split"),
            row.get("task_id"),
        )
        for row in evaluation_rows
    ]
    if len(keys) != len(set(keys)):
        raise NativeMemorySelectionError("E0.16 evaluation rows contain duplicate keys")

    reports = result.get("training_reports")
    if not isinstance(reports, list) or {row.get("seed") for row in reports} != set(seeds):
        raise NativeMemorySelectionError("E0.16 gate training reports are incomplete")
    for report in reports:
        if (
            report.get("optimizer_updates") != 400
            or report.get("examples_seen") != 6400
            or report.get("core_hash_before") != report.get("core_hash_after")
            or report.get("core_hash_before")
            != _mapping(report.get("core_identity"), "core_identity").get(
                "model_parameter_hash"
            )
        ):
            raise NativeMemorySelectionError("E0.16 training or frozen-core record drifted")

    manipulations = _mapping(audit.get("manipulations"), "manipulations")
    required = selection.get("required_manipulations")
    if (
        not isinstance(required, list)
        or set(required) != set(manipulations)
        or not all(manipulations.get(key) is True for key in required)
    ):
        raise NativeMemorySelectionError("E0.16 manipulation audit is incomplete")
    gates = _mapping(audit.get("registered_gates"), "registered_gates")
    if (
        gates.get("structural_pass") is not True
        or gates.get("boundary_safety_pass") is not True
        or gates.get("primary_pass") is not False
        or gates.get("compute_pass") is not False
        or gates.get("learned_correction_pass") is not False
    ):
        raise NativeMemorySelectionError("E0.16 gates do not support explicit multiview")

    metrics = result.get("metric_rows")
    overall = [row for row in metrics if row.get("grouping") == "overall"]
    expected_pairs = {
        (arm_id, seed, split)
        for arm_id in ARM_IDS
        for seed in seeds
        for split in ("id", "ood")
    }
    actual_pairs = {
        (row.get("arm_id"), row.get("seed"), row.get("split")) for row in overall
    }
    if actual_pairs != expected_pairs or any(
        row.get("correct") != 4 or row.get("examples") != 36 for row in overall
    ):
        raise NativeMemorySelectionError("E0.16 accuracy tie does not match selection")

    selected = _mapping(selection.get("selected_configuration"), "selected_configuration")
    if (
        selected.get("arm_id") != SELECTED_ARM
        or selected.get("policy") != "retain-ledger-derived-explicit-views"
        or selected.get("native_session_slots") != 0
        or selected.get("learned_native_memory") is not False
    ):
        raise NativeMemorySelectionError("selection does not fail closed to explicit views")
    retired = selection.get("retired_general_memory_candidates")
    if not isinstance(retired, list) or not all(isinstance(item, Mapping) for item in retired):
        raise NativeMemorySelectionError("retired memory candidates must be objects")
    retired_map = {item.get("arm_id"): item.get("policy") for item in retired}
    if retired_map != {
        "deterministic-session-cache-v1": "preserve-as-negative-evidence-only",
        "minimal-gated-native-memory-v1": "preserve-as-negative-evidence-only",
    } or len(retired) != 2:
        raise NativeMemorySelectionError("stateful memory candidates are not retired")

    authorization = _mapping(selection.get("authorization"), "authorization")
    if (
        authorization.get("explicit_multiview_memory") is not True
        or authorization.get("metamemory_over_explicit_views") is not True
        or any(authorization.get(key) is not False for key in BLOCKED_AUTHORIZATIONS)
    ):
        raise NativeMemorySelectionError("selection authorizes an unselected capability")

    return {
        "selection_id": SELECTION_ID,
        "source_experiment": "E0.16",
        "source_result_sha256": result_sha256,
        "terminal_status": status,
        "selected_arm_id": SELECTED_ARM,
        "learned_native_memory": False,
        "next_memory_policy_scope": "explicit-ledger-derived-views-only",
    }


def validate_selection_files(root: Path) -> dict[str, Any]:
    selection_path = root / "contracts" / "wamrx_native_memory_selection_v1.json"
    selection = json.loads(selection_path.read_text())
    source = _mapping(selection.get("selected_from"), "selected_from")
    result_path = root / str(source.get("result_path"))
    result = json.loads(result_path.read_text())
    return validate_selection(
        selection,
        result,
        result_sha256=file_sha256(result_path),
    )
