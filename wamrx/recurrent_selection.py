"""Fail-closed post-run selection for the Milestone 3 reasoner core.

This module is deliberately standard-library-only.  It binds the selected core
to the exact terminal E0.14 result without importing MLX or re-running metrics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .recurrent_model_constants import RECURRENT_ARM_IDS


SELECTION_SCHEMA_VERSION = 1
STATUS_TO_GENERAL_CORE = {
    "COMPLETE_RETAIN_FIXED": "fixed-depth-v1",
    "COMPLETE_ADOPT_FLAT": "flat-recurrent-v1",
    "COMPLETE_ADOPT_HIERARCHY": "hierarchical-recurrent-v1",
}
BLOCKED_AUTHORIZATIONS = (
    "recurrent_general_core",
    "native_neural_memory",
    "mixture_of_experts",
    "adapter_consolidation",
    "self_improvement",
)


class RecurrentSelectionError(ValueError):
    """The post-run selection does not match its terminal evidence."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecurrentSelectionError(f"{field} must be an object")
    return value


def _pair_set(rows: Any, field: str) -> set[tuple[str, int]]:
    if not isinstance(rows, list):
        raise RecurrentSelectionError(f"{field} must be a list")
    pairs: list[tuple[str, int]] = []
    for row in rows:
        item = _mapping(row, field)
        arm_id = item.get("arm_id")
        seed = item.get("seed")
        if arm_id not in RECURRENT_ARM_IDS or isinstance(seed, bool) or not isinstance(seed, int):
            raise RecurrentSelectionError(f"{field} contains an invalid arm/seed")
        pairs.append((arm_id, seed))
    if len(pairs) != len(set(pairs)):
        raise RecurrentSelectionError(f"{field} contains duplicate arm/seed pairs")
    return set(pairs)


def validate_selection(
    selection: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    result_sha256: str,
) -> dict[str, Any]:
    """Validate a selection and return its stable consumer-facing decision."""

    if selection.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise RecurrentSelectionError("unsupported reasoner-selection schema")
    if selection.get("selection_id") != "wamrx-reasoner-selection-v1":
        raise RecurrentSelectionError("unexpected reasoner-selection identity")

    source = _mapping(selection.get("selected_from"), "selected_from")
    if source.get("experiment") != "E0.14" or result.get("experiment") != "E0.14":
        raise RecurrentSelectionError("selection must be sourced from E0.14")
    if source.get("result_sha256") != result_sha256:
        raise RecurrentSelectionError("E0.14 result hash does not match selection")
    if source.get("manifest") != result.get("manifest"):
        raise RecurrentSelectionError("E0.14 frozen manifest does not match selection")

    status = result.get("status")
    if source.get("terminal_status") != status:
        raise RecurrentSelectionError("terminal result status does not match selection")
    selected_core = _mapping(selection.get("selected_core"), "selected_core")
    expected_arm = STATUS_TO_GENERAL_CORE.get(status)
    if expected_arm is None or selected_core.get("arm_id") != expected_arm:
        raise RecurrentSelectionError("terminal status does not select this core")
    if result.get("invalid_reasons") or result.get("incomplete_reasons"):
        raise RecurrentSelectionError("terminal result contains failure reasons")

    audit = _mapping(result.get("promotion_audit"), "promotion_audit")
    if audit.get("decision") != status:
        raise RecurrentSelectionError("promotion audit and terminal status disagree")
    eligibility = _mapping(audit.get("recurrent_eligible"), "recurrent_eligible")
    if eligibility != {
        "flat-recurrent-v1": False,
        "hierarchical-recurrent-v1": False,
    }:
        raise RecurrentSelectionError("recurrent eligibility does not support fixed depth")

    seeds = selection.get("registered_seeds")
    if not isinstance(seeds, list) or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ):
        raise RecurrentSelectionError("registered_seeds must be integer seeds")
    if len(seeds) != len(set(seeds)) or len(seeds) != 5:
        raise RecurrentSelectionError("selection requires five distinct registered seeds")
    expected_pairs = {(arm_id, seed) for arm_id in RECURRENT_ARM_IDS for seed in seeds}
    if _pair_set(result.get("completed_arm_seeds"), "completed_arm_seeds") != expected_pairs:
        raise RecurrentSelectionError("E0.14 does not contain all registered arm/seed pairs")
    if _pair_set(result.get("training_reports"), "training_reports") != expected_pairs:
        raise RecurrentSelectionError("E0.14 training reports are incomplete")
    if _pair_set(result.get("compute_records"), "compute_records") != expected_pairs:
        raise RecurrentSelectionError("E0.14 compute records are incomplete")
    for record in result["compute_records"]:
        if record.get("status") != "COMPLETE":
            raise RecurrentSelectionError("E0.14 has a non-complete compute record")
        for completed, planned in (
            ("optimizer_updates_completed", "optimizer_updates_planned"),
            ("training_examples_seen", "training_examples_planned"),
            ("evaluation_examples_completed", "evaluation_examples_planned"),
        ):
            if record.get(completed) != record.get(planned):
                raise RecurrentSelectionError(f"E0.14 compute record mismatches {completed}")

    depths = _mapping(result.get("primary_depths"), "primary_depths")
    if depths.get(expected_arm) != selected_core.get("macro_depth"):
        raise RecurrentSelectionError("selected macro depth does not match E0.14")
    if selected_core.get("policy") != "retain-as-general-reasoner-core":
        raise RecurrentSelectionError("selected core lacks the retain policy")

    retired = selection.get("retired_general_core_candidates")
    if not isinstance(retired, list) or not all(
        isinstance(item, Mapping) for item in retired
    ):
        raise RecurrentSelectionError("retired candidates must be objects")
    retired_ids = [item.get("arm_id") for item in retired]
    if len(retired_ids) != len(set(retired_ids)):
        raise RecurrentSelectionError("retired candidates contain duplicate arms")
    retired_pairs = {item.get("arm_id"): item.get("policy") for item in retired}
    if retired_pairs != {
        "flat-recurrent-v1": "preserve-as-negative-evidence-only",
        "hierarchical-recurrent-v1": "preserve-as-negative-evidence-only",
    }:
        raise RecurrentSelectionError("recurrent candidates are not retired fail-closed")

    required_manipulations = selection.get("required_manipulations")
    manipulations = _mapping(audit.get("manipulations"), "manipulations")
    if (
        not isinstance(required_manipulations, list)
        or set(required_manipulations) != set(manipulations)
        or not all(manipulations.get(key) is True for key in required_manipulations)
    ):
        raise RecurrentSelectionError("registered manipulation audit is incomplete")

    authorization = _mapping(selection.get("authorization"), "authorization")
    if authorization.get("fixed_depth_general_core") is not True:
        raise RecurrentSelectionError("fixed-depth core is not authorized")
    if any(authorization.get(key) is not False for key in BLOCKED_AUTHORIZATIONS):
        raise RecurrentSelectionError("selection authorizes work outside E0.14")

    return {
        "selection_id": selection["selection_id"],
        "source_experiment": "E0.14",
        "source_result_sha256": result_sha256,
        "terminal_status": status,
        "selected_arm_id": expected_arm,
        "selected_macro_depth": selected_core["macro_depth"],
        "retired_general_core_candidates": sorted(retired_pairs),
    }


def validate_selection_files(root: Path) -> dict[str, Any]:
    selection_path = root / "contracts" / "wamrx_reasoner_selection_v1.json"
    selection = json.loads(selection_path.read_text())
    source = _mapping(selection.get("selected_from"), "selected_from")
    result_path = root / str(source.get("result_path"))
    result = json.loads(result_path.read_text())
    return validate_selection(
        selection,
        result,
        result_sha256=file_sha256(result_path),
    )
