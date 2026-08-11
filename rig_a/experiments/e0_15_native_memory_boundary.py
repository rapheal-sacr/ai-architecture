"""E0.15 -- authority-limited native working-memory structural falsification.

CLAIM UNDER TEST
    A fixed-capacity, session-bound working-memory interface can expose future
    learned remember/update/merge/forget decisions without acquiring evidence
    authority, surviving reset, or hiding deletion, version, capacity, and
    accounting failures.

SCOPE
    This is a non-neural boundary assay.  It does not instantiate or train the
    minimal learned memory arm, alter the selected reasoner, or read accuracy.

REGISTERED KILL CRITERIA
    K1  the contract, selected-core anchor, statuses, or thresholds are ambiguous.
    K2  any frozen task hash changes or any OOD axis overlaps training.
    K3  checkpoint or durable-candidate schemas omit a boundary field.
    K4  a compliant remember/update/merge/forget fixture fails or escapes accounting.
    K5  memory performs an authoritative write or a candidate lacks observed support.
    K6  tombstoned support remains usable by a slot, answer, or stale read.
    K7  reset, expiry, old-epoch, or cross-user access reveals residual state.
    K8  slot or serialized-byte capacity can be exceeded.
    K9  a checkpoint restores across owner/session/task/epoch/model/ontology identity.
    K10 any of the eleven registered negative controls fails to fire.
    K11 the structural result authorizes neural memory or reports accuracy.

MANIPULATION CHECKS
    M1  unsupported latent state attempts a durable write.
    M2  evidence is omitted after being encoded into memory.
    M3  tombstoned evidence remains in an old state/read.
    M4  an old epoch and a different user attempt access after reset.
    M5  semantically similar facts are forced onto one exact key.
    M6  fixed slot capacity is exceeded.
    M7  a checkpoint is restored under a stale model or ontology.
    M8  an unverified poisoned observation is offered as evidence.
    M9  automatic decay attempts to delete rare protected state.
    M10 checkpoint creation is interrupted.
    M11 memory state is cited as independent evidence.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wamrx.canonical import sha256_json  # noqa: E402
from wamrx.contracts import load_contracts  # noqa: E402
from wamrx.events import Event, SpeechAct  # noqa: E402
from wamrx.native_memory import (  # noqa: E402
    AccessViolation,
    AuthorityViolation,
    CapacityExceeded,
    EvidenceViolation,
    IncompatibleCheckpoint,
    InterruptedCheckpoint,
    MemoryAccess,
    MemoryCollision,
    MemoryEvidenceBundle,
    MemorySlot,
    NativeMemoryConfig,
    ProtectedRetentionViolation,
    REGISTERED_FAILURE_STATUSES,
    REGISTERED_OPERATIONS,
    SessionExpired,
    SessionMemory,
    checkpoint_sha256,
)
from wamrx.native_memory_tasks import (  # noqa: E402
    FAMILIES,
    generate_family,
    load_split_registry,
    verify_frozen_splits,
)
from wamrx.recurrent_selection import file_sha256, validate_selection_files  # noqa: E402
from wamrx.store import AppendOnlyEventStore  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "contracts" / "wamrx_native_memory_v1.json"
SPLIT_PATH = ROOT / "contracts" / "wamrx_native_memory_splits_v1.json"
CORE_MANIFEST_PATH = ROOT / "contracts" / "wamrx_native_memory_core_checkpoints_v1.json"
RESULT_PATH = ROOT / "results" / "e0_15_native_memory_boundary.json"
VALID_AT = "2026-08-11T18:00:00+00:00"
NOW = "2026-08-11T18:05:00+00:00"
EXPIRES = "2026-08-11T19:00:00+00:00"


def observed_event(
    event_id: str,
    value: str,
    region: str,
    minute: int,
    *,
    verifier_class: str = "executable",
) -> Event:
    verified = verifier_class != "unverified"
    return Event.create(
        event_id=event_id,
        transaction_time=f"2026-08-11T17:{minute:02d}:00+00:00",
        valid_from="2026-08-11T00:00:00+00:00",
        actor="native-memory-world",
        source_id=f"source:{region}",
        verifier_id="verifier:native-memory" if verified else None,
        modality="structured-text",
        speech_act=SpeechAct.OBSERVED,
        payload={"key": event_id, "value": value, "region": region},
        confidence=1.0 if verified else None,
        verifier_class=verifier_class,
        provenance_witnesses=(f"external:world:{event_id}",) if verified else (),
    )


def control_event(event_id: str, target: str, minute: int) -> Event:
    return Event.create(
        event_id=event_id,
        transaction_time=f"2026-08-11T18:{minute:02d}:00+00:00",
        valid_from="2026-08-11T00:00:00+00:00",
        actor="native-memory-verifier",
        source_id="verifier:native-memory",
        verifier_id="verifier:native-memory",
        modality="structured-control",
        speech_act=SpeechAct.TOMBSTONE,
        payload={"reason": "registered native-memory deletion control"},
        parent_ids=(target,),
        target_event_ids=(target,),
        verifier_class="executable",
        provenance_witnesses=("external:world:native-memory-control",),
    )


def config(
    *,
    session_id: str = "session-a",
    task_id: str = "task-a",
    owner_id: str = "owner-a",
    maximum_slots: int = 4,
    base_weight_version: str = "fixed-depth-v1:e0.14",
    ontology_version: str = "v1",
    expires_at: str = EXPIRES,
) -> NativeMemoryConfig:
    return NativeMemoryConfig(
        session_id=session_id,
        task_id=task_id,
        owner_id=owner_id,
        maximum_slots=maximum_slots,
        maximum_serialized_bytes=65_536,
        expires_at=expires_at,
        base_weight_version=base_weight_version,
        ontology_version=ontology_version,
    )


def raises(error: type[BaseException], function) -> bool:
    try:
        function()
    except error:
        return True
    return False


def schema_has(path: pathlib.Path, required: set[str]) -> bool:
    schema = json.loads(path.read_text())
    return required <= set(schema.get("required", ()))


def _fresh_bundle(store: AppendOnlyEventStore, *event_ids: str) -> MemoryEvidenceBundle:
    return MemoryEvidenceBundle.create(
        store,
        event_ids=event_ids,
        valid_at=VALID_AT,
        ontology_version="v1",
    )


def run() -> dict:
    contract = json.loads(CONTRACT_PATH.read_text())
    mechanisms = load_contracts(CONTRACT_PATH)
    selection = validate_selection_files(ROOT)
    split_registry = load_split_registry(SPLIT_PATH)
    split_report = verify_frozen_splits(split_registry)
    core_manifest = json.loads(CORE_MANIFEST_PATH.read_text())
    e0_14 = json.loads((ROOT / "results" / "e0_14_recurrent_model_comparison.json").read_text())
    fixed_reports = {
        int(row["seed"]): row["checkpoint"]
        for row in e0_14["training_reports"]
        if row["arm_id"] == "fixed-depth-v1"
    }
    frozen_core_matches_e0_14 = (
        core_manifest["source_result_sha256"] == selection["source_result_sha256"]
        and core_manifest["arm_id"] == "fixed-depth-v1"
        and core_manifest["macro_depth"] == 4
        and set(fixed_reports) == {row["seed"] for row in core_manifest["checkpoints"]}
        and all(
            {
                key: fixed_reports[row["seed"]][key]
                for key in ("bytes", "file_sha256", "metadata_hash", "state_hash")
            }
            == {key: row[key] for key in ("bytes", "file_sha256", "metadata_hash", "state_hash")}
            for row in core_manifest["checkpoints"]
        )
    )
    core_availability = []
    for row in core_manifest["checkpoints"]:
        relocated = ROOT / row["relocation_path"]
        source = pathlib.Path(row["source_path"])
        located = relocated if relocated.exists() else source if source.exists() else None
        core_availability.append(
            {
                "seed": row["seed"],
                "available": located is not None,
                "located_path": str(located) if located is not None else None,
                "hash_matches": (
                    located is not None
                    and located.stat().st_size == row["bytes"]
                    and file_sha256(located) == row["file_sha256"]
                ),
            }
        )

    with tempfile.TemporaryDirectory(prefix="wamrx-e0-15-") as directory:
        temp = pathlib.Path(directory)
        store = AppendOnlyEventStore(temp / "authority.sqlite")
        store.append_batch(
            [
                observed_event("fact-a", "alpha", "finance", 0),
                observed_event("fact-b", "beta", "operations", 1),
                observed_event("fact-corrected", "alpha-2", "finance", 2),
                observed_event("fact-rare", "tail", "rare-protected", 3),
            ]
        )
        initial_ledger_count = store.count()
        all_bundle = _fresh_bundle(
            store, "fact-a", "fact-b", "fact-corrected", "fact-rare"
        )

        memory = SessionMemory(config())
        access = memory.access
        slot_a = memory.remember(
            store,
            access=access,
            now=NOW,
            bundle=all_bundle,
            content_key="project-location",
            value="alpha",
            support_event_ids=("fact-a",),
            protected_regions=("finance",),
        )
        slot_b = memory.remember(
            store,
            access=access,
            now=NOW,
            bundle=all_bundle,
            content_key="project-owner",
            value="beta",
            support_event_ids=("fact-b",),
            protected_regions=("operations",),
        )
        slot_a = memory.update(
            store,
            access=access,
            now=NOW,
            bundle=all_bundle,
            slot_id=slot_a.slot_id,
            value="alpha-2",
            support_event_ids=("fact-corrected",),
        )
        merged = memory.merge(
            store,
            access=access,
            now=NOW,
            bundle=all_bundle,
            slot_ids=(slot_a.slot_id, slot_b.slot_id),
            content_key="project-summary",
            value={"location": "alpha-2", "owner": "beta"},
        )
        memory.forget(
            access=access,
            now=NOW,
            slot_id=merged.slot_id,
            reason="explicit-user",
        )
        rare = memory.remember(
            store,
            access=access,
            now=NOW,
            bundle=all_bundle,
            content_key="rare-protected-fact",
            value="tail",
            support_event_ids=("fact-rare",),
            protected_regions=("rare-protected",),
        )
        read = memory.prepare_model_call(
            store, access=access, now=NOW, bundle=all_bundle
        )
        manifest = memory.answer_manifest(read, slot_ids=(rare.slot_id,))
        candidate = memory.durable_candidate(
            store,
            access=access,
            now=NOW,
            read=read,
            proposal_id="candidate-rare-v1",
            payload={"claim": "rare fact is tail"},
            slot_ids=(rare.slot_id,),
        )
        candidate_ledger_count = store.count()
        ledger_unchanged = candidate_ledger_count == initial_ledger_count

        checkpoint = memory.checkpoint(access=access, now=NOW)
        restored = SessionMemory.restore(
            checkpoint,
            expected_config=memory.config,
            expected_access=access,
        )
        decoded_checkpoint = json.loads(checkpoint.decode("utf-8"))
        checkpoint_roundtrip = (
            restored.state_hash == memory.state_hash
            and restored.payload_state_hash == memory.payload_state_hash
            and decoded_checkpoint["checkpoint_sha256"]
            == sha256_json(decoded_checkpoint["checkpoint"])
        )

        # M1: forge a read containing a latent-only slot, then ask for a durable output.
        latent_slot = MemorySlot(
            slot_id="memory:latent-durable",
            content_key="latent",
            value="unsupported",
            support_event_ids=("memory:latent-root",),
            superseded_support_event_ids=(),
            protected_regions=("operations",),
            created_turn=0,
            updated_turn=0,
        )
        latent_read = dataclasses.replace(
            read, active_slots=(*read.active_slots, latent_slot)
        )
        unsupported_durable_rejected = raises(
            AuthorityViolation,
            lambda: memory.durable_candidate(
                store,
                access=access,
                now=NOW,
                read=latent_read,
                proposal_id="latent-write",
                payload={"claim": "unsupported"},
                slot_ids=(latent_slot.slot_id,),
            ),
        )

        # M2: active support has to be present in the newly built call bundle.
        omitted_bundle = _fresh_bundle(store, "fact-a", "fact-b")
        omitted_evidence_rejected = raises(
            EvidenceViolation,
            lambda: memory.prepare_model_call(
                store, access=access, now=NOW, bundle=omitted_bundle
            ),
        )

        # M5 and M11 fire before any mutation.
        collision_detected = raises(
            MemoryCollision,
            lambda: memory.remember(
                store,
                access=access,
                now=NOW,
                bundle=all_bundle,
                content_key="rare-protected-fact",
                value="similar but distinct",
                support_event_ids=("fact-b",),
                protected_regions=("operations",),
            ),
        )
        memory_as_evidence_rejected = raises(
            AuthorityViolation,
            lambda: memory.remember(
                store,
                access=access,
                now=NOW,
                bundle=all_bundle,
                content_key="latent-root-attempt",
                value="unsupported",
                support_event_ids=(rare.slot_id,),
                protected_regions=("operations",),
            ),
        )
        protected_decay_rejected = raises(
            ProtectedRetentionViolation,
            lambda: memory.forget(
                access=access,
                now=NOW,
                slot_id=rare.slot_id,
                reason="decay",
            ),
        )
        interrupted_checkpoint_rejected = raises(
            InterruptedCheckpoint,
            lambda: memory.checkpoint(
                access=access, now=NOW, interrupt_after_slots=1
            ),
        )

        # M3: deletion changes the frontier, disables the affected slot, and rejects old reads.
        store.append(control_event("delete-fact-rare", "fact-rare", 10))
        post_delete_bundle = _fresh_bundle(store, "fact-a", "fact-b", "fact-corrected")
        post_delete_read = memory.prepare_model_call(
            store, access=access, now=NOW, bundle=post_delete_bundle
        )
        tombstone_disabled = (
            rare.slot_id in post_delete_read.disabled_slot_ids
            and rare.slot_id not in {slot.slot_id for slot in post_delete_read.active_slots}
        )
        stale_read_rejected = raises(
            EvidenceViolation,
            lambda: memory.durable_candidate(
                store,
                access=access,
                now=NOW,
                read=read,
                proposal_id="stale-after-delete",
                payload={"claim": "tail"},
                slot_ids=(rare.slot_id,),
            ),
        )

        # M4: reset clears payload, rotates epoch, and rejects both old and cross-user tokens.
        reset_memory = SessionMemory(config(session_id="reset-session"))
        reset_access = reset_memory.access
        reset_memory.remember(
            store,
            access=reset_access,
            now=NOW,
            bundle=post_delete_bundle,
            content_key="reset-secret",
            value="must disappear",
            support_event_ids=("fact-a",),
            protected_regions=("finance",),
        )
        reset_receipt = reset_memory.reset(access=reset_access, now=NOW)
        old_epoch_rejected = raises(
            AccessViolation,
            lambda: reset_memory.prepare_model_call(
                store, access=reset_access, now=NOW, bundle=post_delete_bundle
            ),
        )
        cross_user = dataclasses.replace(
            reset_receipt.new_access, owner_id="owner-other"
        )
        cross_user_rejected = raises(
            AccessViolation,
            lambda: reset_memory.prepare_model_call(
                store, access=cross_user, now=NOW, bundle=post_delete_bundle
            ),
        )
        reset_zero = (
            not reset_memory.slots
            and reset_memory.payload_state_hash == sha256_json([])
            and reset_receipt.empty_payload_hash == sha256_json([])
        )

        expiry_memory = SessionMemory(
            config(
                session_id="expiry-session",
                expires_at="2026-08-11T18:06:00+00:00",
            )
        )
        expiry_access = expiry_memory.access
        expiry_memory.remember(
            store,
            access=expiry_access,
            now=NOW,
            bundle=post_delete_bundle,
            content_key="expiring",
            value="gone",
            support_event_ids=("fact-a",),
            protected_regions=("finance",),
        )
        expiry_rejected = raises(
            SessionExpired,
            lambda: expiry_memory.prepare_model_call(
                store,
                access=expiry_access,
                now="2026-08-11T18:06:00+00:00",
                bundle=post_delete_bundle,
            ),
        ) and not expiry_memory.slots

        # M6: a fixed two-slot memory cannot allocate a third slot.
        cap_memory = SessionMemory(config(session_id="capacity-session", maximum_slots=2))
        cap_access = cap_memory.access
        for key, event_id in (("one", "fact-a"), ("two", "fact-b")):
            cap_memory.remember(
                store,
                access=cap_access,
                now=NOW,
                bundle=post_delete_bundle,
                content_key=key,
                value=key,
                support_event_ids=(event_id,),
                protected_regions=("finance",),
            )
        capacity_rejected = raises(
            CapacityExceeded,
            lambda: cap_memory.remember(
                store,
                access=cap_access,
                now=NOW,
                bundle=post_delete_bundle,
                content_key="three",
                value="three",
                support_event_ids=("fact-corrected",),
                protected_regions=("finance",),
            ),
        ) and len(cap_memory.slots) == 2

        # M7: identity includes both model and ontology versions.
        stale_model_rejected = raises(
            IncompatibleCheckpoint,
            lambda: SessionMemory.restore(
                checkpoint,
                expected_config=config(base_weight_version="fixed-depth-v2"),
                expected_access=access,
            ),
        )
        stale_ontology_rejected = raises(
            IncompatibleCheckpoint,
            lambda: SessionMemory.restore(
                checkpoint,
                expected_config=config(ontology_version="v2"),
                expected_access=access,
            ),
        )

        # M8: unverified observations cannot enter an evidence bundle.
        poison_store = AppendOnlyEventStore(temp / "poison.sqlite")
        poison_store.append(
            observed_event(
                "poisoned-observation",
                "adversarial",
                "finance",
                20,
                verifier_class="unverified",
            )
        )
        poisoned_observation_rejected = raises(
            EvidenceViolation,
            lambda: _fresh_bundle(poison_store, "poisoned-observation"),
        )

        manipulations = {
            "M1_unsupported_latent_durable_write_rejected": unsupported_durable_rejected,
            "M2_omitted_reinjected_evidence_rejected": omitted_evidence_rejected,
            "M3_tombstoned_old_state_and_read_disabled": tombstone_disabled and stale_read_rejected,
            "M4_reset_old_epoch_and_cross_user_rejected": reset_zero and old_epoch_rejected and cross_user_rejected,
            "M5_similar_fact_key_collision_detected": collision_detected,
            "M6_fixed_capacity_overflow_rejected": capacity_rejected,
            "M7_stale_model_and_ontology_checkpoint_rejected": stale_model_rejected and stale_ontology_rejected,
            "M8_poisoned_unverified_observation_rejected": poisoned_observation_rejected,
            "M9_protected_rare_decay_rejected": protected_decay_rejected,
            "M10_interrupted_checkpoint_rejected": interrupted_checkpoint_rejected,
            "M11_memory_as_independent_evidence_rejected": memory_as_evidence_rejected,
        }

        checkpoint_schema_complete = schema_has(
            ROOT / contract["executable_schemas"]["checkpoint"],
            {"checkpoint", "checkpoint_sha256"},
        )
        candidate_schema_complete = schema_has(
            ROOT / contract["executable_schemas"]["durable_candidate"],
            {
                "proposal_id",
                "session_id",
                "task_id",
                "owner_id",
                "payload",
                "support",
                "ledger_frontier_sequence",
                "ledger_frontier_hash",
                "authority",
                "direct_ledger_write_authorized",
            },
        )
        task_counts = {
            split: sum(
                len(generate_family(split, family, split_registry["splits"][split]))
                for family in FAMILIES
            )
            for split in ("train", "id", "ood")
        }
        compute = memory.compute_record()
        required_operation_counts = {
            operation: compute.operation_counts.get(operation, 0)
            for operation in REGISTERED_OPERATIONS
        }
        anchor = contract["depends_on"]
        contract_complete = (
            len(mechanisms) == 5
            and set(contract["failure_statuses"]) == set(REGISTERED_FAILURE_STATUSES)
            and len(contract["registered_negative_controls"]) == 11
            and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in contract["structural_thresholds"].values()
            )
            and anchor["selected_core_id"] == selection["selected_arm_id"]
            and anchor["selected_macro_depth"] == selection["selected_macro_depth"]
            and anchor["e0_14_result_sha256"] == selection["source_result_sha256"]
            and sha256_json(core_manifest)
            == contract["frozen_core_weights"]["content_hash"]
            and frozen_core_matches_e0_14
        )
        checks = {
            "K1_contract_status_threshold_and_selection_anchor_complete": contract_complete,
            "K2_frozen_splits_exact_and_ood_disjoint": (
                split_report["passed"]
                and sha256_json(split_registry)
                == contract["frozen_split_registry"]["content_hash"]
            ),
            "K3_checkpoint_and_candidate_schemas_complete": checkpoint_schema_complete and candidate_schema_complete,
            "K4_compliant_operations_and_exact_accounting_pass": (
                all(value >= 1 for value in required_operation_counts.values())
                and compute.realized_memory_flops > 0
                and compute.occupied_slots <= compute.maximum_slots
                and compute.serialized_slot_bytes <= compute.maximum_serialized_bytes
            ),
            "K5_candidate_only_boundary_and_observed_support_hold": (
                ledger_unchanged
                and candidate.authority == "candidate-only"
                and candidate.direct_ledger_write_authorized is False
                and candidate.support.supporting_event_ids == ("fact-rare",)
                and manifest.supporting_event_ids == ("fact-rare",)
            ),
            "K6_tombstone_immediately_disables_slot_answer_and_old_read": tombstone_disabled and stale_read_rejected,
            "K7_reset_expiry_epoch_and_cross_user_leave_no_residual": (
                reset_zero and old_epoch_rejected and cross_user_rejected and expiry_rejected
            ),
            "K8_fixed_slot_and_byte_capacity_enforced": (
                capacity_rejected
                and compute.serialized_slot_bytes <= compute.maximum_serialized_bytes
            ),
            "K9_checkpoint_roundtrip_exact_and_identity_bound": (
                checkpoint_roundtrip and stale_model_rejected and stale_ontology_rejected
            ),
            "K10_all_eleven_negative_controls_fire": all(manipulations.values()),
            "K11_neural_comparison_remains_not_run": True,
        }

        result = {
            "experiment": "E0.15",
            "status": "PASS" if all(checks.values()) else "FAIL",
            "verdict": "PASS" if all(checks.values()) else "FAIL",
            "neural_comparison_status": "NOT_RUN",
            "accuracy_metrics_read": 0,
            "selected_core": {
                "arm_id": selection["selected_arm_id"],
                "macro_depth": selection["selected_macro_depth"],
                "selection_commit": anchor["selection_commit"],
                "selection_tag": anchor["selection_tag"],
                "e0_14_result_sha256": selection["source_result_sha256"],
            },
            "frozen_core_weights": {
                "manifest_id": core_manifest["manifest_id"],
                "manifest_hash": sha256_json(core_manifest),
                "checkpoints_match_e0_14": frozen_core_matches_e0_14,
                "artifact_availability": core_availability,
                "availability_status": (
                    "READY"
                    if all(row["hash_matches"] for row in core_availability)
                    else "NOT_AVAILABLE"
                ),
            },
            "contract": {
                "path": str(CONTRACT_PATH.relative_to(ROOT)),
                "hash": sha256_json(contract),
                "mechanism_ids": [item.mechanism_id for item in mechanisms],
                "comparison_arms": [item["arm_id"] for item in contract["comparison_arms"]],
                "failure_statuses": list(contract["failure_statuses"]),
            },
            "frozen_splits": {
                "registry_id": split_report["registry_id"],
                "registry_hash": sha256_json(split_registry),
                "task_counts": task_counts,
                "family_count": len(FAMILIES),
                "hash_mismatches": split_report["hash_mismatches"],
                "ood_axis_disjoint": split_report["ood_axis_disjoint"],
            },
            "checks": checks,
            "manipulations": manipulations,
            "compliant_fixture": {
                "registered_operations": list(REGISTERED_OPERATIONS),
                "operation_counts": required_operation_counts,
                "ledger_count_before_memory": initial_ledger_count,
                "ledger_count_after_candidate": candidate_ledger_count,
                "candidate": candidate.to_dict(),
                "checkpoint_sha256": checkpoint_sha256(checkpoint),
                "checkpoint_bytes": len(checkpoint),
                "compute": compute.to_dict(),
            },
            "scope_limit": "Non-neural structural boundary only. E0.16 remains NOT_RUN; no learned memory, model-weight, depth, accuracy, MoE, consolidation, or promotion claim is made.",
        }

    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
