from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from rig_a.experiments.e0_15_native_memory_boundary import (
    EXPIRES,
    NOW,
    VALID_AT,
    config,
    control_event,
    observed_event,
    run,
)
from tools.check_native_memory_core_artifacts import check as check_core_artifacts
from wamrx.canonical import sha256_json
from wamrx.native_memory import (
    AccessViolation,
    AuthorityViolation,
    CapacityExceeded,
    EvidenceViolation,
    IncompatibleCheckpoint,
    MemoryCollision,
    MemoryEvidenceBundle,
    NativeMemoryConfig,
    ProtectedRetentionViolation,
    SessionExpired,
    SessionMemory,
)
from wamrx.native_memory_tasks import (
    FAMILIES,
    generate_family,
    load_split_registry,
    verify_frozen_splits,
)
from wamrx.store import AppendOnlyEventStore


ROOT = Path(__file__).resolve().parents[1]


class NativeMemoryBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = AppendOnlyEventStore(Path(self.temp.name) / "ledger.sqlite")
        self.store.append_batch(
            [
                observed_event("fact-a", "alpha", "finance", 0),
                observed_event("fact-b", "beta", "operations", 1),
                observed_event("fact-c", "gamma", "rare-protected", 2),
            ]
        )
        self.bundle = self.evidence("fact-a", "fact-b", "fact-c")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def evidence(self, *event_ids: str) -> MemoryEvidenceBundle:
        return MemoryEvidenceBundle.create(
            self.store,
            event_ids=event_ids,
            valid_at=VALID_AT,
            ontology_version="v1",
        )

    def remember(
        self,
        memory: SessionMemory,
        key: str,
        event_id: str,
        *,
        region: str = "finance",
    ):
        return memory.remember(
            self.store,
            access=memory.access,
            now=NOW,
            bundle=self.bundle,
            content_key=key,
            value=f"value:{key}",
            support_event_ids=(event_id,),
            protected_regions=(region,),
        )

    def test_model_call_reinjects_support_and_tombstone_disables_slot(self) -> None:
        memory = SessionMemory(config())
        slot = self.remember(memory, "key-a", "fact-a")
        with self.assertRaises(EvidenceViolation):
            memory.prepare_model_call(
                self.store,
                access=memory.access,
                now=NOW,
                bundle=self.evidence("fact-b"),
            )
        old_read = memory.prepare_model_call(
            self.store, access=memory.access, now=NOW, bundle=self.bundle
        )
        self.store.append(control_event("delete-a", "fact-a", 10))
        current = memory.prepare_model_call(
            self.store,
            access=memory.access,
            now=NOW,
            bundle=self.evidence("fact-b", "fact-c"),
        )
        self.assertIn(slot.slot_id, current.disabled_slot_ids)
        with self.assertRaises(EvidenceViolation):
            memory.durable_candidate(
                self.store,
                access=memory.access,
                now=NOW,
                read=old_read,
                proposal_id="stale",
                payload={"claim": "alpha"},
                slot_ids=(slot.slot_id,),
            )

    def test_update_merge_and_candidate_preserve_ledger_only_support(self) -> None:
        memory = SessionMemory(config())
        a = self.remember(memory, "key-a", "fact-a")
        b = self.remember(memory, "key-b", "fact-b", region="operations")
        a = memory.update(
            self.store,
            access=memory.access,
            now=NOW,
            bundle=self.bundle,
            slot_id=a.slot_id,
            value="gamma",
            support_event_ids=("fact-c",),
        )
        self.assertEqual(a.superseded_support_event_ids, ("fact-a",))
        merged = memory.merge(
            self.store,
            access=memory.access,
            now=NOW,
            bundle=self.bundle,
            slot_ids=(a.slot_id, b.slot_id),
            content_key="summary",
            value={"a": "gamma", "b": "beta"},
        )
        read = memory.prepare_model_call(
            self.store, access=memory.access, now=NOW, bundle=self.bundle
        )
        before = self.store.count()
        proposal = memory.durable_candidate(
            self.store,
            access=memory.access,
            now=NOW,
            read=read,
            proposal_id="proposal",
            payload={"summary": True},
            slot_ids=(merged.slot_id,),
        )
        self.assertEqual(self.store.count(), before)
        self.assertEqual(
            proposal.support.supporting_event_ids, ("fact-b", "fact-c")
        )
        self.assertFalse(proposal.direct_ledger_write_authorized)
        self.assertFalse(
            any(item.startswith("memory:") for item in proposal.support.candidate_event_ids)
        )

    def test_collision_slot_capacity_and_byte_capacity_fail_closed(self) -> None:
        memory = SessionMemory(config(maximum_slots=1))
        self.remember(memory, "key-a", "fact-a")
        with self.assertRaises(MemoryCollision):
            self.remember(memory, "key-a", "fact-b")
        with self.assertRaises(CapacityExceeded):
            self.remember(memory, "key-b", "fact-b")
        self.assertEqual(len(memory.slots), 1)

        tiny = SessionMemory(
            NativeMemoryConfig(
                session_id="tiny",
                task_id="task",
                owner_id="owner",
                maximum_slots=4,
                maximum_serialized_bytes=8,
                expires_at=EXPIRES,
                base_weight_version="fixed-depth-v1:e0.14",
                ontology_version="v1",
            )
        )
        with self.assertRaises(CapacityExceeded):
            self.remember(tiny, "too-large", "fact-a")
        self.assertEqual(tiny.compute_record().occupied_slots, 0)

    def test_reset_cross_identity_and_expiry_clear_state(self) -> None:
        memory = SessionMemory(config())
        old_access = memory.access
        self.remember(memory, "secret", "fact-a")
        receipt = memory.reset(access=old_access, now=NOW)
        self.assertEqual(memory.payload_state_hash, sha256_json([]))
        self.assertFalse(memory.slots)
        with self.assertRaises(AccessViolation):
            memory.prepare_model_call(
                self.store, access=old_access, now=NOW, bundle=self.bundle
            )
        other = dataclasses.replace(receipt.new_access, owner_id="other")
        with self.assertRaises(AccessViolation):
            memory.prepare_model_call(
                self.store, access=other, now=NOW, bundle=self.bundle
            )

        expiring = SessionMemory(
            config(session_id="expires", expires_at="2026-08-11T18:06:00+00:00")
        )
        self.remember(expiring, "expiring", "fact-a")
        with self.assertRaises(SessionExpired):
            expiring.prepare_model_call(
                self.store,
                access=expiring.access,
                now="2026-08-11T18:06:00+00:00",
                bundle=self.bundle,
            )
        self.assertFalse(expiring.slots)

    def test_checkpoint_is_hash_version_and_session_bound(self) -> None:
        memory = SessionMemory(config())
        self.remember(memory, "key-a", "fact-a")
        access = memory.access
        checkpoint = memory.checkpoint(access=access, now=NOW)
        restored = SessionMemory.restore(
            checkpoint, expected_config=memory.config, expected_access=access
        )
        self.assertEqual(restored.state_hash, memory.state_hash)

        with self.assertRaises(IncompatibleCheckpoint):
            SessionMemory.restore(
                checkpoint,
                expected_config=config(base_weight_version="fixed-depth-v2"),
                expected_access=access,
            )
        envelope = json.loads(checkpoint.decode("utf-8"))
        envelope["checkpoint"]["task_id"] = "tampered"
        tampered = json.dumps(envelope).encode("utf-8")
        with self.assertRaises(IncompatibleCheckpoint):
            SessionMemory.restore(
                tampered, expected_config=memory.config, expected_access=access
            )

    def test_protected_decay_and_memory_evidence_are_rejected(self) -> None:
        memory = SessionMemory(config())
        rare = self.remember(
            memory, "rare", "fact-c", region="rare-protected"
        )
        with self.assertRaises(ProtectedRetentionViolation):
            memory.forget(
                access=memory.access,
                now=NOW,
                slot_id=rare.slot_id,
                reason="decay",
            )
        with self.assertRaises(AuthorityViolation):
            memory.remember(
                self.store,
                access=memory.access,
                now=NOW,
                bundle=self.bundle,
                content_key="latent",
                value="unsupported",
                support_event_ids=(rare.slot_id,),
                protected_regions=("operations",),
            )


class NativeMemoryRegistrationTests(unittest.TestCase):
    def test_frozen_core_checkpoint_identities_match_e0_14(self) -> None:
        report = check_core_artifacts()
        self.assertFalse(report["identity_errors"])
        self.assertIn(report["status"], {"READY", "NOT_AVAILABLE"})
        self.assertTrue(
            all(row["identity_matches_e0_14"] for row in report["artifacts"])
        )

    def test_frozen_splits_are_exact_and_ood_disjoint(self) -> None:
        registry = load_split_registry(
            ROOT / "contracts" / "wamrx_native_memory_splits_v1.json"
        )
        report = verify_frozen_splits(registry)
        self.assertTrue(report["passed"])
        self.assertTrue(all(report["ood_axis_disjoint"].values()))
        self.assertEqual(
            sum(
                len(generate_family("train", family, registry["splits"]["train"]))
                for family in FAMILIES
            ),
            72,
        )

    def test_registered_e0_15_passes_without_neural_metrics(self) -> None:
        result = run(write_result=False)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["neural_comparison_status"], "NOT_RUN")
        self.assertEqual(result["accuracy_metrics_read"], 0)
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(len(result["manipulations"]), 11)
        self.assertTrue(all(result["manipulations"].values()))


if __name__ == "__main__":
    unittest.main()
