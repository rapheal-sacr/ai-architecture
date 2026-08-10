from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rig_a.experiments.e0_3_no_compounding import run as run_e0_3
from rig_a.experiments.e0_4_grounding_audit import run as run_e0_4
from wamrx.artifacts import (
    ArtifactCompatibilityPolicy,
    ArtifactEnvelope,
    ArtifactStamp,
    InvalidArtifactError,
    SupportManifest,
)
from wamrx.events import Event, SpeechAct
from wamrx.resolver import resolve
from wamrx.store import AppendOnlyEventStore

VALID_AT = "2026-08-10T20:00:00+00:00"


def observed(event_id: str, *, transaction_time: str) -> Event:
    return Event.create(
        event_id=event_id,
        transaction_time=transaction_time,
        valid_from="2026-08-10T00:00:00-04:00",
        actor="test",
        source_id="source:test",
        verifier_id="verifier:test",
        modality="text",
        speech_act=SpeechAct.OBSERVED,
        payload={"text": event_id, "region": "test"},
        verifier_class="grounded",
        provenance_witnesses=(f"external:test:{event_id}",),
    )


def verify(event_id: str, target: str, *, transaction_time: str) -> Event:
    return Event.create(
        event_id=event_id,
        transaction_time=transaction_time,
        valid_from="2026-08-10T00:00:00+00:00",
        actor="test",
        source_id="source:test",
        verifier_id="verifier:test",
        modality="control",
        speech_act=SpeechAct.VERIFIED,
        payload={"reason": "test"},
        parent_ids=(target,),
        target_event_ids=(target,),
        verifier_class="executable",
        provenance_witnesses=(f"event:{target}",),
    )


class GroundingGateTests(unittest.TestCase):
    def test_registered_e0_3_and_e0_4(self) -> None:
        e0_3 = run_e0_3()
        e0_4 = run_e0_4()
        self.assertEqual(e0_3["verdict"], "PASS")
        self.assertTrue(all(e0_3["checks"].values()))
        self.assertEqual(e0_4["verdict"], "PASS")
        self.assertEqual(e0_4["grounding_coverage"], 1.0)


class TemporalOrderTests(unittest.TestCase):
    def test_mixed_offsets_normalize_before_hashing(self) -> None:
        eastern = observed(
            "same", transaction_time="2026-08-10T08:00:00-04:00"
        )
        utc = observed(
            "same", transaction_time="2026-08-10T12:00:00+00:00"
        )
        self.assertEqual(eastern.transaction_time, "2026-08-10T12:00:00+00:00")
        self.assertEqual(eastern.canonical, utc.canonical)

    def test_backdated_append_does_not_reorder_transaction_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = AppendOnlyEventStore(Path(directory) / "ledger.sqlite")
            root = observed(
                "root", transaction_time="2026-08-10T12:00:00+00:00"
            )
            first = verify(
                "verify-first",
                "root",
                transaction_time="2026-08-10T14:00:00+00:00",
            )
            backdated = verify(
                "verify-backdated",
                "root",
                transaction_time="2026-08-10T10:00:00+00:00",
            )
            store.append_batch([root, first])
            store.append(backdated)
            record = resolve(store, valid_at=VALID_AT).record("root")
            self.assertEqual(
                record.control_event_ids,
                ("verify-first", "verify-backdated"),
            )


class CompatibilityPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = AppendOnlyEventStore(Path(self.temp.name) / "ledger.sqlite")
        self.store.append(
            observed("support", transaction_time="2026-08-10T12:00:00+00:00")
        )
        self.content = {"value": 1}
        self.stamp = ArtifactStamp.create(
            artifact_id="artifact-v1",
            artifact_type="test",
            content=self.content,
            store=self.store,
            base_weight_version="none",
            component_versions={"compiler": "v1"},
            ontology_version="ontology-v1",
            verifier_version="verifier-v1",
            build_config={"mode": "test"},
        )
        self.envelope = ArtifactEnvelope(
            content=self.content,
            stamp=self.stamp,
            support=SupportManifest.create(
                supporting_event_ids=["support"],
                candidate_event_ids=["support"],
            ),
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_read_requires_policy(self) -> None:
        with self.assertRaises(InvalidArtifactError):
            self.envelope.validate(
                self.store,
                compatibility_policy=None,
                valid_at=VALID_AT,
            )

    def test_unregistered_mismatches_fail_closed(self) -> None:
        component_v2 = ArtifactCompatibilityPolicy(
            active_base_weight_version="none",
            active_component_versions={"compiler": "v2"},
            active_ontology_version="ontology-v1",
            active_verifier_version="verifier-v1",
        )
        with self.assertRaises(InvalidArtifactError):
            self.envelope.validate(
                self.store,
                compatibility_policy=component_v2,
                valid_at=VALID_AT,
            )
        ontology_v2 = ArtifactCompatibilityPolicy(
            active_base_weight_version="none",
            active_component_versions={"compiler": "v1"},
            active_ontology_version="ontology-v2",
            active_verifier_version="verifier-v1",
        )
        with self.assertRaises(InvalidArtifactError):
            self.envelope.validate(
                self.store,
                compatibility_policy=ontology_v2,
                valid_at=VALID_AT,
            )

    def test_registered_predecessors_pass(self) -> None:
        policy = ArtifactCompatibilityPolicy(
            active_base_weight_version="none",
            active_component_versions={"compiler": "v2"},
            active_ontology_version="ontology-v2",
            active_verifier_version="verifier-v2",
            compatible_component_predecessors={"compiler": ("v1",)},
            compatible_ontology_predecessors=("ontology-v1",),
            compatible_verifier_predecessors=("verifier-v1",),
        )
        self.envelope.validate(
            self.store,
            compatibility_policy=policy,
            valid_at=VALID_AT,
        )


if __name__ == "__main__":
    unittest.main()
