"""E0.10 -- WAM-RX milestone-1 authoritative memory kernel.

CLAIM UNDER TEST
    A small, non-neural implementation can reconstruct, correct, audit, and
    delete memory while preserving regional compile adequacy and exact lineage.

PRE-REGISTERED KILL CRITERIA
    K1  repeated replay produces more than one resolved snapshot hash.
    K2  interruption at any insert boundary exposes a partial event batch.
    K3  the ledger accepts mutation of an existing event row.
    K4  an incomplete or hash-mismatched artifact stamp reads.
    K5  an artifact can cite evidence absent from its recorded frontier.
    K6  an all-support answer remains usable after support is invalidated.
    K7  a pooled-pass/tail-fail compiler is not rejected, or the compliant
        compiler fails coverage, correction, or contradiction preservation.
    K8  retired evidence remains in a repaired artifact.
    K9  surviving support becomes unavailable after tombstone or repair.
    K10 repaired and clean-from-scratch index content hashes differ.
    K11 a retrieval journal omits an indexed rival from its candidate set.
    K12 a journal omits scores, filters, versions, selections, margins, or its
        append-only guarantee.
    K13 a previously selected tombstoned item remains selectable through the
        stale index before asynchronous repair.

MANIPULATION CHECKS
    M1  the biased compiler excludes the only rare event and MUST fail the rare
        region while pooled coverage remains >= 0.90.
    M2  the pre-deletion query MUST select the item later tombstoned.
    M3  corrupting artifact content MUST make its envelope unreadable.

The registered thresholds live in contracts/wamrx_milestone1.json.  This file
does not tune them after observing the run.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sqlite3
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wamrx.artifacts import (  # noqa: E402
    ArtifactCompatibilityPolicy,
    ArtifactEnvelope,
    ArtifactStamp,
    InvalidArtifactError,
    SupportManifest,
)
from wamrx.contracts import load_contracts  # noqa: E402
from wamrx.evaluation import QueryProbe, measure_compile_adequacy  # noqa: E402
from wamrx.events import Event, SpeechAct  # noqa: E402
from wamrx.resolver import replay, resolve  # noqa: E402
from wamrx.retrieval import HybridRetrievalIndex  # noqa: E402
from wamrx.store import (  # noqa: E402
    AppendOnlyEventStore,
    SimulatedWriteInterruption,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
VALID_AT = "2026-08-09T12:00:00+00:00"


def memory_event(
    event_id: str,
    text: str,
    region: str,
    minute: int,
    *,
    metadata: dict[str, str] | None = None,
) -> Event:
    timestamp = f"2026-08-09T10:{minute:02d}:00+00:00"
    return Event.create(
        event_id=event_id,
        transaction_time=timestamp,
        valid_from="2026-08-09T00:00:00+00:00",
        actor="synthetic-world",
        source_id=f"source:{region}",
        modality="text",
        speech_act=SpeechAct.OBSERVED,
        payload={"text": text, "region": region, "metadata": metadata or {}},
        confidence=1.0,
        verifier_class="executable",
        verifier_id="verifier:synthetic-world",
        provenance_witnesses=(f"external:world:{event_id}",),
    )


def control_event(
    event_id: str, act: SpeechAct, target: str, minute: int
) -> Event:
    timestamp = f"2026-08-09T11:{minute:02d}:00+00:00"
    return Event.create(
        event_id=event_id,
        transaction_time=timestamp,
        valid_from="2026-08-09T00:00:00+00:00",
        actor="synthetic-verifier",
        source_id="verifier:ground-truth",
        modality="structured-control",
        speech_act=act,
        payload={"reason": f"synthetic {act.value}"},
        target_event_ids=(target,),
        parent_ids=(target,),
        verifier_class="executable",
        verifier_id="verifier:ground-truth",
        provenance_witnesses=("external:world:control-manifest",),
    )


def world_events() -> list[Event]:
    events = [
        memory_event(
            f"common-{index}",
            f"common archive beacon token{index}",
            "common",
            index,
            metadata={"kind": "archive"},
        )
        for index in range(9)
    ]
    events.extend(
        [
            memory_event(
                "rare-0",
                "rare quasar obsidian signal",
                "rare",
                9,
                metadata={"kind": "rare-observation"},
            ),
            memory_event(
                "location-old",
                "project office location montreal",
                "common",
                10,
                metadata={"kind": "location"},
            ),
            memory_event(
                "location-new",
                "project office location toronto",
                "common",
                11,
                metadata={"kind": "location"},
            ),
            memory_event(
                "delete-me",
                "temporary cobalt launch phrase",
                "common",
                12,
                metadata={"kind": "temporary"},
            ),
            memory_event(
                "delete-me-too",
                "temporary amber launch phrase",
                "common",
                13,
                metadata={"kind": "temporary"},
            ),
            control_event("refute-location-old", SpeechAct.REFUTED, "location-old", 0),
            control_event("verify-location-new", SpeechAct.VERIFIED, "location-new", 1),
        ]
    )
    return events


def check_atomic_recovery(root: pathlib.Path) -> dict:
    batch = [
        memory_event("atomic-a", "atomic alpha", "common", 20),
        memory_event("atomic-b", "atomic beta", "common", 21),
        memory_event("atomic-c", "atomic gamma", "rare", 22),
    ]
    rows_after_interruption = {}
    for boundary in range(len(batch) + 1):
        path = root / f"atomic-{boundary}.sqlite"
        store = AppendOnlyEventStore(path)
        try:
            store.append_batch(batch, interrupt_after=boundary)
        except SimulatedWriteInterruption:
            pass
        rows_after_interruption[str(boundary)] = store.count()
        store.verify_integrity()
    return {
        "rows_after_interruption": rows_after_interruption,
        "passed": all(value == 0 for value in rows_after_interruption.values()),
    }


def check_append_only(store: AppendOnlyEventStore) -> bool:
    statements = (
        "UPDATE events SET event_json = event_json WHERE event_id = 'common-0'",
        "DELETE FROM events WHERE event_id = 'common-0'",
    )
    for statement in statements:
        connection = sqlite3.connect(store.path)
        try:
            connection.execute(statement)
            connection.commit()
            return False
        except sqlite3.DatabaseError:
            connection.rollback()
        finally:
            connection.close()
    return True


def check_journal_append_only(store: AppendOnlyEventStore) -> bool:
    statements = (
        "UPDATE retrieval_journal SET record_json = record_json WHERE query_id = 'delete-before'",
        "DELETE FROM retrieval_journal WHERE query_id = 'delete-before'",
    )
    for statement in statements:
        connection = sqlite3.connect(store.path)
        try:
            connection.execute(statement)
            connection.commit()
            return False
        except sqlite3.DatabaseError:
            connection.rollback()
        finally:
            connection.close()
    return True


def run() -> dict:
    contracts = load_contracts(ROOT / "contracts" / "wamrx_milestone1.json")
    with tempfile.TemporaryDirectory(prefix="wamrx-e0-10-") as directory:
        temp = pathlib.Path(directory)
        store = AppendOnlyEventStore(temp / "world.sqlite")
        events = world_events()
        store.append_batch(events)
        store.verify_integrity()

        replay_hashes = {
            resolve(store, valid_at=VALID_AT).snapshot_hash for _ in range(5)
        }
        replay_hashes.add(replay(store.events(), valid_at=VALID_AT).snapshot_hash)
        atomic = check_atomic_recovery(temp)
        append_only = check_append_only(store)

        active_ids = resolve(store, valid_at=VALID_AT).usable_event_ids()
        biased_ids = active_ids - {"rare-0"}
        biased = HybridRetrievalIndex.build(
            store,
            artifact_id="biased-index-v1",
            valid_at=VALID_AT,
            include_event_ids=biased_ids,
        )
        complete = HybridRetrievalIndex.build(
            store, artifact_id="complete-index-v1", valid_at=VALID_AT
        )
        probes = (
            QueryProbe(
                probe_id="common",
                region="common",
                query="common archive beacon token0",
                expected_event_ids=("common-0",),
            ),
            QueryProbe(
                probe_id="rare",
                region="rare",
                query="rare quasar obsidian signal",
                expected_event_ids=("rare-0",),
            ),
        )
        biased_report = measure_compile_adequacy(
            store,
            biased,
            valid_at=VALID_AT,
            protected_regions=("common", "rare"),
            probes=probes,
            minimum_regional_coverage=1.0,
            maximum_regional_query_distortion=0.0,
        )
        complete_report = measure_compile_adequacy(
            store,
            complete,
            valid_at=VALID_AT,
            protected_regions=("common", "rare"),
            probes=probes,
            minimum_regional_coverage=1.0,
            maximum_regional_query_distortion=0.0,
        )

        corrected_location = complete.search(
            "project office location montreal",
            query_id="corrected-location",
            valid_at=VALID_AT,
            top_k=2,
        )

        before_delete = complete.search(
            "temporary cobalt launch phrase",
            query_id="delete-before",
            valid_at=VALID_AT,
            top_k=1,
        )
        before_delete_too = complete.search(
            "temporary amber launch phrase",
            query_id="delete-too-before",
            valid_at=VALID_AT,
            top_k=1,
        )
        answer_manifest = complete.selection_manifest("delete-before")
        answer_content = {"answer": "temporary cobalt launch phrase"}
        answer_stamp = ArtifactStamp.create(
            artifact_id="answer-before-delete",
            artifact_type="retrieval-answer",
            content=answer_content,
            store=store,
            base_weight_version="none",
            component_versions={"retrieval": "wamrx-hybrid-v1"},
            ontology_version="v1",
            verifier_version="executable-v1",
            build_config={"query_id": "delete-before"},
        )
        answer_envelope = ArtifactEnvelope(
            content=answer_content,
            stamp=answer_stamp,
            support=answer_manifest,
        )
        answer_policy = ArtifactCompatibilityPolicy.exact_for_stamp(answer_stamp)
        answer_envelope.validate(
            store,
            compatibility_policy=answer_policy,
            valid_at=VALID_AT,
        )
        tombstone = control_event(
            "tombstone-delete-me", SpeechAct.TOMBSTONE, "delete-me", 2
        )
        tombstone_too = control_event(
            "tombstone-delete-me-too", SpeechAct.TOMBSTONE, "delete-me-too", 3
        )
        store.append_batch([tombstone, tombstone_too])
        after_delete = complete.search(
            "temporary cobalt launch phrase",
            query_id="delete-after-stale-index",
            valid_at=VALID_AT,
            top_k=1,
        )
        after_delete_too = complete.search(
            "temporary amber launch phrase",
            query_id="delete-too-after-stale-index",
            valid_at=VALID_AT,
            top_k=1,
        )
        surviving = complete.search(
            "rare quasar obsidian signal",
            query_id="survivor-after-delete",
            valid_at=VALID_AT,
            top_k=1,
        )

        answer_after_delete_rejected = False
        try:
            answer_envelope.validate(
                store,
                compatibility_policy=answer_policy,
                valid_at=VALID_AT,
            )
        except InvalidArtifactError:
            answer_after_delete_rejected = True

        corrupted_content_rejected = False
        try:
            ArtifactEnvelope(
                content={"corrupt": True},
                stamp=complete.envelope.stamp,
                support=complete.envelope.support,
            ).validate(
                store,
                compatibility_policy=complete.compatibility_policy,
                valid_at=VALID_AT,
                check_support=False,
            )
        except InvalidArtifactError:
            corrupted_content_rejected = True

        missing_lineage_rejected = False
        try:
            ArtifactEnvelope(
                content=answer_content,
                stamp=answer_stamp,
                support=SupportManifest.create(
                    supporting_event_ids=["missing-event"],
                    candidate_event_ids=["missing-event"],
                ),
            ).validate(
                store,
                compatibility_policy=answer_policy,
                valid_at=VALID_AT,
            )
        except InvalidArtifactError:
            missing_lineage_rejected = True

        incomplete_stamp_rejected = False
        try:
            ArtifactEnvelope(
                content=answer_content,
                stamp=dataclasses.replace(answer_stamp, verifier_version=""),
                support=answer_manifest,
            ).validate(
                store,
                compatibility_policy=answer_policy,
                valid_at=VALID_AT,
            )
        except InvalidArtifactError:
            incomplete_stamp_rejected = True

        missing_frontier_rejected = False
        try:
            ArtifactEnvelope(
                content=answer_content,
                stamp=dataclasses.replace(
                    answer_stamp, ledger_frontier_hash="f" * 64
                ),
                support=answer_manifest,
            ).validate(
                store,
                compatibility_policy=answer_policy,
                valid_at=VALID_AT,
            )
        except InvalidArtifactError:
            missing_frontier_rejected = True

        repaired = HybridRetrievalIndex.build(
            store, artifact_id="repaired-index-v2", valid_at=VALID_AT
        )
        clean_store = AppendOnlyEventStore(temp / "clean.sqlite")
        clean_store.append_batch(store.events())
        clean = HybridRetrievalIndex.build(
            clean_store, artifact_id="clean-index-v2", valid_at=VALID_AT
        )
        repair_equivalent = repaired.content_hash == clean.content_hash

        journals = store.retrieval_records()
        journal_keys = {
            "query_id",
            "query",
            "query_version",
            "valid_at",
            "filters",
            "index_artifact_id",
            "index_content_hash",
            "index_frontier",
            "decision_frontier",
            "component_versions",
            "candidates",
            "selected_event_ids",
            "top_k",
            "top_k_boundary_margin",
        }
        candidate_keys = {
            "event_id",
            "supporting_event_ids",
            "control_event_ids",
            "resolved_status",
            "filter_passed",
            "filter_reasons",
            "lexical_score",
            "embedding_score",
            "reranker_score",
            "final_score",
            "selected",
        }
        journal_complete = all(
            journal_keys <= set(record)
            and all(candidate_keys <= set(candidate) for candidate in record["candidates"])
            for record in journals
        )
        delete_after_record = next(
            item for item in journals if item["query_id"] == "delete-after-stale-index"
        )
        full_candidate_journal = len(delete_after_record["candidates"]) == len(
            complete.documents
        )

        checks = {
            "K1_deterministic_replay": len(replay_hashes) == 1,
            "K2_atomic_recovery": atomic["passed"],
            "K3_append_only_enforced": append_only,
            "K4_artifact_stamp_fail_closed": (
                corrupted_content_rejected and incomplete_stamp_rejected
            ),
            "K5_missing_lineage_and_frontier_rejected": (
                missing_lineage_rejected and missing_frontier_rejected
            ),
            "K6_invalidated_support_rejected": answer_after_delete_rejected,
            "K7_regional_compiler_and_corrections_pass": (
                biased_report.pooled_coverage >= 0.90
                and biased_report.regional_coverage["rare"] == 0.0
                and not biased_report.passed
                and complete_report.passed
                and complete_report.contradiction_preservation == 1.0
                and corrected_location.hits
                and corrected_location.hits[0].event_id == "location-new"
                and all(
                    hit.event_id != "location-old" for hit in corrected_location.hits
                )
            ),
            "K8_retired_evidence_absent_from_repair": (
                {"delete-me", "delete-me-too"}.isdisjoint(repaired.document_ids())
                and repaired.disabled_records.get("delete-me") == "tombstone"
                and repaired.disabled_records.get("delete-me-too") == "tombstone"
            ),
            "K9_survivor_available": (
                surviving.hits and surviving.hits[0].event_id == "rare-0"
            ),
            "K10_clean_rebuild_equivalent": repair_equivalent,
            "K11_complete_candidate_set": full_candidate_journal,
            "K12_selection_journal_complete_and_immutable": (
                journal_complete and check_journal_append_only(store)
            ),
            "K13_tombstone_immediate_disable": (
                before_delete.hits
                and before_delete.hits[0].event_id == "delete-me"
                and before_delete_too.hits
                and before_delete_too.hits[0].event_id == "delete-me-too"
                and all(hit.event_id != "delete-me" for hit in after_delete.hits)
                and all(
                    hit.event_id != "delete-me-too" for hit in after_delete_too.hits
                )
            ),
        }
        return {
            "experiment": "E0.10",
            "claim": "milestone-1 memory is reconstructable, correctable, auditable, and deletable",
            "contract_count": len(contracts),
            "event_count_before_delete": len(events),
            "event_count_after_delete": store.count(),
            "replay_hash": next(iter(replay_hashes)),
            "atomic_recovery": atomic,
            "biased_compiler": biased_report.to_dict(),
            "complete_compiler": complete_report.to_dict(),
            "retrieval_journal_count": len(journals),
            "checks": checks,
            "verdict": "PASS" if all(checks.values()) else "FAIL",
        }


def main() -> int:
    result = run()
    path = ROOT / "results" / "e0_10_wamrx_memory_kernel.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
