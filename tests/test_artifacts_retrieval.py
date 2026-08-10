from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from wamrx.artifacts import (
    ArtifactCompatibilityPolicy,
    ArtifactEnvelope,
    ArtifactStamp,
    InvalidArtifactError,
    SupportManifest,
)
from wamrx.events import Event, SpeechAct
from wamrx.retrieval import HybridRetrievalIndex
from wamrx.store import AppendOnlyEventStore, EventConflictError

VALID_AT = "2026-08-09T12:00:00+00:00"


def fact(event_id: str, text: str, region: str, kind: str) -> Event:
    return Event.create(
        event_id=event_id,
        transaction_time="2026-08-09T10:00:00+00:00",
        valid_from="2026-08-09T00:00:00+00:00",
        actor="test",
        source_id="source",
        modality="text",
        speech_act=SpeechAct.ASSERTED,
        payload={
            "text": text,
            "region": region,
            "metadata": {"kind": kind},
        },
        verifier_class="grounded",
        verifier_id="test-verifier",
        provenance_witnesses=(f"external:test:{event_id}",),
    )


def tombstone(target: str) -> Event:
    return Event.create(
        event_id=f"tombstone-{target}",
        transaction_time="2026-08-09T11:00:00+00:00",
        valid_from="2026-08-09T00:00:00+00:00",
        actor="test",
        source_id="verifier",
        modality="structured-control",
        speech_act=SpeechAct.TOMBSTONE,
        target_event_ids=(target,),
        parent_ids=(target,),
        verifier_class="executable",
        verifier_id="test-verifier",
        provenance_witnesses=("external:test:tombstone",),
    )


class ArtifactAndRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = AppendOnlyEventStore(Path(self.temp.name) / "ledger.sqlite")
        self.store.append_batch(
            [
                fact("alpha", "alpha cobalt project", "common", "project"),
                fact("beta", "beta saffron recipe", "rare", "recipe"),
                fact("gamma", "gamma cobalt recipe", "common", "recipe"),
            ]
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_complete_journal_and_metadata_filter(self) -> None:
        index = HybridRetrievalIndex.build(
            self.store, artifact_id="index", valid_at=VALID_AT
        )
        result = index.search(
            "cobalt",
            query_id="filtered",
            valid_at=VALID_AT,
            top_k=2,
            metadata_filter={"kind": "recipe"},
        )
        self.assertEqual([hit.event_id for hit in result.hits], ["gamma"])
        journal = self.store.retrieval_records()[0]
        self.assertEqual(len(journal["candidates"]), 3)
        alpha = next(item for item in journal["candidates"] if item["event_id"] == "alpha")
        self.assertIn("metadata:kind", alpha["filter_reasons"])
        self.assertIn("embedding_score", alpha)
        self.assertEqual(alpha["supporting_event_ids"], ["alpha"])
        self.assertIn("decision_frontier", journal)
        self.assertIn("top_k_boundary_margin", journal)

    def test_duplicate_query_id_cannot_change_audit_record(self) -> None:
        index = HybridRetrievalIndex.build(
            self.store, artifact_id="index", valid_at=VALID_AT
        )
        index.search("alpha", query_id="same", valid_at=VALID_AT)
        with self.assertRaises(EventConflictError):
            index.search("beta", query_id="same", valid_at=VALID_AT)

    def test_tombstone_disables_stale_index_and_answer_manifest(self) -> None:
        index = HybridRetrievalIndex.build(
            self.store, artifact_id="index", valid_at=VALID_AT
        )
        before = index.search("alpha cobalt", query_id="before", valid_at=VALID_AT, top_k=1)
        self.assertEqual(before.hits[0].event_id, "alpha")
        answer = {"answer": "alpha cobalt project"}
        stamp = ArtifactStamp.create(
            artifact_id="answer",
            artifact_type="answer",
            content=answer,
            store=self.store,
            base_weight_version="none",
            component_versions={"retrieval": "v1"},
            ontology_version="v1",
            verifier_version="test-v1",
            build_config={"query_id": "before"},
        )
        envelope = ArtifactEnvelope(
            content=answer,
            stamp=stamp,
            support=index.selection_manifest("before"),
        )
        policy = ArtifactCompatibilityPolicy.exact_for_stamp(stamp)
        envelope.validate(
            self.store, compatibility_policy=policy, valid_at=VALID_AT
        )
        self.store.append(tombstone("alpha"))
        after = index.search("alpha cobalt", query_id="after", valid_at=VALID_AT, top_k=3)
        self.assertNotIn("alpha", [hit.event_id for hit in after.hits])
        after_journal = self.store.retrieval_records()[-1]
        alpha_candidate = next(
            item for item in after_journal["candidates"] if item["event_id"] == "alpha"
        )
        self.assertEqual(alpha_candidate["resolved_status"], "tombstone")
        self.assertEqual(alpha_candidate["control_event_ids"], ["tombstone-alpha"])
        with self.assertRaises(InvalidArtifactError):
            envelope.validate(
                self.store, compatibility_policy=policy, valid_at=VALID_AT
            )

    def test_missing_lineage_and_content_tamper_fail_closed(self) -> None:
        content = {"summary": "alpha"}
        stamp = ArtifactStamp.create(
            artifact_id="summary",
            artifact_type="summary",
            content=content,
            store=self.store,
            base_weight_version="none",
            component_versions={"compiler": "v1"},
            ontology_version="v1",
            verifier_version="test-v1",
            build_config={"mode": "exact"},
        )
        good = SupportManifest.create(
            supporting_event_ids=["alpha"], candidate_event_ids=["alpha", "beta"]
        )
        policy = ArtifactCompatibilityPolicy.exact_for_stamp(stamp)
        ArtifactEnvelope(content, stamp, good).validate(
            self.store, compatibility_policy=policy, valid_at=VALID_AT
        )
        with self.assertRaises(InvalidArtifactError):
            ArtifactEnvelope({"summary": "tampered"}, stamp, good).validate(
                self.store, compatibility_policy=policy, valid_at=VALID_AT
            )
        missing = SupportManifest.create(
            supporting_event_ids=["ghost"], candidate_event_ids=["ghost"]
        )
        with self.assertRaises(InvalidArtifactError):
            ArtifactEnvelope(content, stamp, missing).validate(
                self.store, compatibility_policy=policy, valid_at=VALID_AT
            )


if __name__ == "__main__":
    unittest.main()
