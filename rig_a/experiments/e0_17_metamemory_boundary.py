"""E0.17 -- explicit-view metamemory and compression structural boundary.

This assay is deliberately non-neural. It executes the seven advisory policy
actions and falsifies authority, coverage, temporal, contradiction, unresolved,
rebuild, deletion, skill-candidate, capacity, and accounting paths before a
learned policy is implemented. No ID/OOD performance metric is read.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wamrx.canonical import sha256_json  # noqa: E402
from wamrx.events import Event, SpeechAct  # noqa: E402
from wamrx.metamemory import (  # noqa: E402
    CompressionArtifact,
    DeterministicMetamemoryPolicy,
    MAXIMUM_SOURCE_ITEMS,
    MetamemoryDecision,
    MetamemoryError,
    SkillCandidate,
    build_compression_artifact,
    build_skill_candidate,
    validate_compression_artifact,
    validate_decision,
    validate_skill_candidate,
    with_replaced_artifact,
)
from wamrx.metamemory_tasks import (  # noqa: E402
    FAMILIES,
    MetamemoryTask,
    generate_family,
    verify_frozen_splits,
)
from wamrx.native_memory_selection import validate_selection_files  # noqa: E402
from wamrx.store import AppendOnlyEventStore  # noqa: E402


CONTRACT_PATH = ROOT / "contracts" / "wamrx_metamemory_v1.json"
SPLIT_PATH = ROOT / "contracts" / "wamrx_metamemory_splits_v1.json"
RESULT_PATH = ROOT / "results" / "e0_17_metamemory_boundary.json"
VALID_AT = "2026-08-12T18:00:00+00:00"


def _event(item: dict[str, Any], index: int) -> Event:
    transaction = datetime(2026, 8, 12, 14, 0, tzinfo=timezone.utc) + timedelta(
        seconds=index
    )
    verified = bool(item["verified"])
    return Event.create(
        event_id=str(item["event_id"]),
        transaction_time=transaction.isoformat(),
        valid_from="2026-08-12T00:00:00+00:00",
        actor="metamemory-task-world",
        source_id=f"source:{item['event_id']}",
        verifier_id="verifier:metamemory-task" if verified else None,
        modality="structured-text",
        speech_act=SpeechAct.OBSERVED,
        payload={
            "region": item["region"],
            "kind": item["kind"],
            "content": item["content"],
        },
        confidence=1.0 if verified else None,
        verifier_class="executable" if verified else "unverified",
        provenance_witnesses=(f"external:metamemory:{item['event_id']}",)
        if verified
        else (),
    )


def _store(path: Path, tasks: tuple[MetamemoryTask, ...]) -> AppendOnlyEventStore:
    store = AppendOnlyEventStore(path)
    events = []
    for task in tasks:
        for item in task.items:
            events.append(_event(dict(item), len(events)))
    store.append_batch(events)
    return store


def _tombstone(target: str) -> Event:
    return Event.create(
        event_id=f"metamemory:tombstone:{target}",
        transaction_time="2026-08-12T17:30:00+00:00",
        valid_from="2026-08-12T00:00:00+00:00",
        actor="metamemory-verifier",
        source_id="verifier:metamemory",
        verifier_id="verifier:metamemory",
        modality="structured-control",
        speech_act=SpeechAct.TOMBSTONE,
        payload={"reason": "registered deletion control"},
        parent_ids=(target,),
        target_event_ids=(target,),
        verifier_class="executable",
        provenance_witnesses=("external:metamemory:deletion-control",),
    )


def _fires(error_type: type[BaseException], operation: Callable[[], Any]) -> bool:
    try:
        operation()
    except error_type:
        return True
    return False


def _dependency_report(contract: dict[str, Any]) -> dict[str, Any]:
    hashes = {}
    for name, registration in contract["depends_on"].items():
        path = ROOT / registration["path"]
        actual = sha256_json(json.loads(path.read_text()))
        if actual != registration["content_hash"]:
            raise ValueError(f"E0.17 dependency hash drifted: {name}")
        hashes[name] = actual
    selection = validate_selection_files(ROOT)
    if (
        selection["terminal_status"]
        != contract["depends_on"]["native_memory_selection"][
            "required_terminal_status"
        ]
        or selection["selected_arm_id"]
        != contract["depends_on"]["native_memory_selection"]["required_arm"]
    ):
        raise ValueError("E0.17 is not bound to the selected explicit-memory outcome")
    return {"hashes": hashes, "selection": selection}


def run(*, write_result: bool = True) -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text())
    registry = json.loads(SPLIT_PATH.read_text())
    dependency = _dependency_report(contract)
    split_report = verify_frozen_splits(registry)
    if not split_report["passed"]:
        raise ValueError("E0.17 frozen metamemory splits drifted")
    tasks = tuple(
        generate_family("train", family, registry["splits"]["train"])[0]
        for family in FAMILIES
    )

    with tempfile.TemporaryDirectory() as directory:
        store = _store(Path(directory) / "ledger.sqlite", tasks)
        ledger_before = store.count()
        policy = DeterministicMetamemoryPolicy()
        decisions = []
        for task in tasks:
            decision = policy.decide(task)
            validate_decision(store, task, decision, valid_at=VALID_AT)
            decisions.append(decision)

        compression_task = next(
            task for task in tasks if task.family == "redundant_history_compression"
        )
        compression = build_compression_artifact(
            store,
            compression_task,
            valid_at=VALID_AT,
        )
        validate_compression_artifact(
            store,
            compression_task,
            compression,
            valid_at=VALID_AT,
        )
        rebuilt = build_compression_artifact(
            store,
            compression_task,
            valid_at=VALID_AT,
        )
        policy.charge_artifact(compression)

        procedure_task = next(
            task for task in tasks if task.family == "procedure_repetition"
        )
        skill_candidate = build_skill_candidate(
            store,
            procedure_task,
            valid_at=VALID_AT,
        )
        validate_skill_candidate(
            store,
            procedure_task,
            skill_candidate,
            valid_at=VALID_AT,
        )
        policy.charge_skill_candidate(skill_candidate)
        ledger_after = store.count()
        compute = policy.compute_record()

        summarize_decision = next(
            decision for decision in decisions if decision.action == "summarize"
        )
        manipulations = {
            "M1_fabricated_support_rejected": _fires(
                MetamemoryError,
                lambda: validate_decision(
                    store,
                    compression_task,
                    dataclasses.replace(
                        summarize_decision,
                        supporting_event_ids=("summary:fabricated",),
                    ),
                    valid_at=VALID_AT,
                ),
            ),
            "M2_dropped_region_rejected": _fires(
                MetamemoryError,
                lambda: validate_compression_artifact(
                    store,
                    compression_task,
                    with_replaced_artifact(
                        compression,
                        region_manifest={
                            key: value
                            for key, value in compression.region_manifest.items()
                            if key != "rare-protected"
                        },
                    ),
                    valid_at=VALID_AT,
                ),
            ),
            "M3_dropped_contradiction_or_refutation_rejected": all(
                _fires(
                    MetamemoryError,
                    lambda field=field: validate_compression_artifact(
                        store,
                        compression_task,
                        with_replaced_artifact(compression, **{field: ()}),
                        valid_at=VALID_AT,
                    ),
                )
                for field in ("contradiction_item_ids", "refutation_item_ids")
            ),
            "M4_dropped_temporal_qualifier_rejected": _fires(
                MetamemoryError,
                lambda: validate_compression_artifact(
                    store,
                    compression_task,
                    with_replaced_artifact(compression, temporal_qualifiers={}),
                    valid_at=VALID_AT,
                ),
            ),
            "M5_hidden_unresolved_item_rejected": _fires(
                MetamemoryError,
                lambda: validate_compression_artifact(
                    store,
                    compression_task,
                    with_replaced_artifact(compression, unresolved_item_ids=()),
                    valid_at=VALID_AT,
                ),
            ),
            "M6_omitted_source_support_rejected": _fires(
                MetamemoryError,
                lambda: validate_compression_artifact(
                    store,
                    compression_task,
                    with_replaced_artifact(
                        compression,
                        supporting_event_ids=compression.supporting_event_ids[:-1],
                    ),
                    valid_at=VALID_AT,
                ),
            ),
            "M8_changed_rebuild_hash_rejected": _fires(
                MetamemoryError,
                lambda: validate_compression_artifact(
                    store,
                    compression_task,
                    with_replaced_artifact(compression, source_item_hash="0" * 64),
                    valid_at=VALID_AT,
                ),
            ),
            "M9_metamemory_ids_as_evidence_rejected": _fires(
                MetamemoryError,
                lambda: validate_compression_artifact(
                    store,
                    compression_task,
                    with_replaced_artifact(
                        compression,
                        supporting_event_ids=(compression.artifact_id,),
                    ),
                    valid_at=VALID_AT,
                ),
            ),
            "M10_no_direct_ledger_write_path": (
                ledger_before == ledger_after
                and all(not decision.direct_ledger_write_authorized for decision in decisions)
            ),
            "M11_unsupported_skill_candidate_rejected": _fires(
                MetamemoryError,
                lambda: validate_skill_candidate(
                    store,
                    procedure_task,
                    dataclasses.replace(
                        skill_candidate,
                        repetition_item_ids=skill_candidate.repetition_item_ids[:2],
                    ),
                    valid_at=VALID_AT,
                ),
            ),
            "M12_stale_ontology_rejected": _fires(
                MetamemoryError,
                lambda: validate_compression_artifact(
                    store,
                    compression_task,
                    with_replaced_artifact(compression, ontology_version="v0"),
                    valid_at=VALID_AT,
                ),
            ),
            "M13_capacity_caps_fail_closed": all(
                (
                    _fires(
                        MetamemoryError,
                        lambda: validate_compression_artifact(
                            store,
                            compression_task,
                            with_replaced_artifact(
                                compression,
                                summary_text="x" * 33000,
                            ),
                            valid_at=VALID_AT,
                        ),
                    ),
                    _fires(
                        MetamemoryError,
                        lambda: build_compression_artifact(
                            store,
                            dataclasses.replace(
                                compression_task,
                                items=tuple(
                                    {
                                        **dict(compression_task.items[0]),
                                        "item_id": f"overflow-item-{index:03d}",
                                    }
                                    for index in range(MAXIMUM_SOURCE_ITEMS + 1)
                                ),
                            ),
                            valid_at=VALID_AT,
                        ),
                    ),
                )
            ),
        }
        store.append(_tombstone(compression.supporting_event_ids[0]))
        manipulations["M7_tombstoned_support_disables_artifact"] = _fires(
            MetamemoryError,
            lambda: validate_compression_artifact(
                store,
                compression_task,
                compression,
                valid_at=VALID_AT,
            ),
        )

    checks = {
        "K1_dependencies_exact": bool(dependency["hashes"]),
        "K2_no_authoritative_write": ledger_before == ledger_after,
        "K3_ledger_only_evidence": all(
            not any(
                event_id.startswith(("memory:", "slot:", "summary:", "policy:"))
                for event_id in decision.supporting_event_ids
            )
            for decision in decisions
        ),
        "K4_protected_fields_preserved": (
            set(compression.region_manifest) == set(compression_task.protected_regions)
            and len(compression.contradiction_item_ids) == 1
            and len(compression.refutation_item_ids) == 1
            and len(compression.temporal_qualifiers) == 2
            and len(compression.unresolved_item_ids) == 1
            and len(compression.supporting_event_ids) == len(compression_task.items)
        ),
        "K5_clean_rebuild_exact": compression == rebuilt,
        "K6_deletion_control_fires": manipulations[
            "M7_tombstoned_support_disables_artifact"
        ],
        "K7_skill_candidate_witnessed": (
            len(skill_candidate.repetition_item_ids) >= 3
            and not skill_candidate.promotion_authority
        ),
        "K8_ignore_and_request_semantics": all(
            decision.action not in {"ignore", "request_evidence"}
            or not decision.supporting_event_ids
            for decision in decisions
        ),
        "K9_capacity_within_caps": (
            len(compression.source_item_ids) <= MAXIMUM_SOURCE_ITEMS
            and compute.serialized_artifact_bytes <= 32768
        ),
        "K10_accounting_complete": (
            set(compute.operation_counts) == set(contract["policy_actions"])
            and sum(compute.operation_counts.values()) == len(contract["policy_actions"])
            and compute.realized_policy_flops > 0
        ),
        "K11_splits_exact_and_disjoint": split_report["passed"],
        "K12_scope_remains_non_neural": contract["future_comparison"]["status"]
        == "NOT_RUN",
    }
    status = "PASS" if all(checks.values()) and all(manipulations.values()) else "FAIL"
    result = {
        "experiment": "E0.17",
        "status": status,
        "contract_id": contract["contract_id"],
        "contract_hash": sha256_json(contract),
        "dependency_report": dependency,
        "split_report": split_report,
        "fixture_task_ids": [task.task_id for task in tasks],
        "decisions": [decision.to_dict() for decision in decisions],
        "compression_artifact": compression.to_dict(),
        "compression_rebuild_exact": compression == rebuilt,
        "skill_candidate": skill_candidate.to_dict(),
        "compute": compute.to_dict(),
        "ledger_rows_before": ledger_before,
        "ledger_rows_after": ledger_after,
        "checks": checks,
        "manipulations": manipulations,
        "accuracy_metrics_read": 0,
        "learned_policy_status": "NOT_RUN",
        "future_comparison_status": contract["future_comparison"]["status"],
        "scope_limit": "Structural explicit-view metamemory boundary only. No learned policy, ID/OOD performance, native memory, core change, MoE, consolidation, or promotion claim is made.",
    }
    if write_result:
        RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(write_result=not args.check_only), indent=2, sort_keys=True))
