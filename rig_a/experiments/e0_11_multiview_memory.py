"""E0.11 -- provenance-linked analytic and belief/constraint memory.

CLAIM UNDER TEST
    Typed temporal analytics and an explicit belief/constraint graph add
    executable aggregation, comparison, contradiction, and multi-constraint
    operations without becoming independent sources of truth.

BASELINE SCOPE
    The retrieval baseline succeeds only if the registered exact answer appears
    in one retrieved hit.  It tests the retrieval API alone, not retrieval plus
    a language model or query-specific arithmetic.  This narrow interpretation
    is frozen in contracts/wamrx_multiview_memory.json.

REGISTERED KILL CRITERIA
    K1  malformed structured payloads compile or any row/edge lacks ledger support.
    K2  corrected/retracted analytic rows affect exact aggregates or comparisons.
    K3  uncertain candidate fields lose confidence or witness identity.
    K4  a refuted competing claim disappears from contradiction history.
    K5  missing/conflicting evidence is classified as satisfied.
    K6  a generated next step appears as independent supporting evidence.
    K7  multiview memory misses any of four registered exact tasks.
    K8  task gain over the answer-in-hit retrieval baseline is less than two.
    K9  evidence-lineage coverage is below 1.0 in any protected region.
    K10 a stale analytic row or graph edge remains usable after tombstone.
    K11 pre-deletion analytic or graph answer manifests survive support deletion.
    K12 repaired analytic/graph artifacts differ from clean-from-scratch builds.
    K13 ontology identity is absent from either artifact stamp.
    K14 an analytic query journal is incomplete, mutable, or misses a registered query.

MANIPULATION CHECKS
    M1  summing raw event payloads MUST include the retracted 999 outlier and be wrong.
    M2  an active-only graph MUST drop the resolved contradiction history.
    M3  deleting the operations-region lineage MUST make its regional coverage fail.
    M4  registering a query that never runs MUST make journal-coverage audit fail.
    M5  a malformed analytic schema MUST fail compilation.
    M6  an active v2 ontology MUST reject a v1 artifact without a migration rule.
    M7  contradictory live claims MUST keep a constraint unresolved.
    M8  a generated next-step string MUST fail if laundered into artifact support.
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wamrx.analytic import AnalyticCompileError, AnalyticMemory  # noqa: E402
from wamrx.artifacts import (  # noqa: E402
    ArtifactCompatibilityPolicy,
    ArtifactEnvelope,
    ArtifactStamp,
    InvalidArtifactError,
    SupportManifest,
)
from wamrx.belief_graph import (  # noqa: E402
    BeliefGraph,
    ConstraintRequirement,
)
from wamrx.contracts import load_contracts  # noqa: E402
from wamrx.canonical import sha256_json  # noqa: E402
from wamrx.events import Event, SpeechAct  # noqa: E402
from wamrx.retrieval import HybridRetrievalIndex  # noqa: E402
from wamrx.store import AppendOnlyEventStore  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
VALID_AT = "2026-08-10T12:00:00+00:00"


def content_event(
    event_id: str,
    text: str,
    region: str,
    minute: int,
    *,
    analytic: dict | None = None,
    claim: dict | None = None,
    verifier_class: str = "executable",
) -> Event:
    payload = {"text": text, "region": region}
    if analytic is not None:
        payload["analytic"] = analytic
    if claim is not None:
        payload["claim"] = claim
    return Event.create(
        event_id=event_id,
        transaction_time=f"2026-08-10T09:{minute:02d}:00+00:00",
        valid_from="2026-08-10T00:00:00+00:00",
        actor="multiview-world",
        source_id=f"source:{region}",
        verifier_id="verifier:multiview-ground-truth",
        modality="structured-text",
        speech_act=SpeechAct.OBSERVED,
        payload=payload,
        confidence=1.0,
        verifier_class=verifier_class,
        provenance_witnesses=(f"external:world:{event_id}",),
    )


def control_event(
    event_id: str,
    act: SpeechAct,
    target: str,
    minute: int,
) -> Event:
    return Event.create(
        event_id=event_id,
        transaction_time=f"2026-08-10T10:{minute:02d}:00+00:00",
        valid_from="2026-08-10T00:00:00+00:00",
        actor="multiview-verifier",
        source_id="verifier:multiview-ground-truth",
        verifier_id="verifier:multiview-ground-truth",
        modality="structured-control",
        speech_act=act,
        payload={"reason": f"registered {act.value}"},
        parent_ids=(target,),
        target_event_ids=(target,),
        verifier_class="executable",
        provenance_witnesses=("external:world:control-manifest",),
    )


def expense(
    event_id: str,
    month: str,
    amount: float,
    minute: int,
    *,
    candidate_vendor: bool = False,
) -> Event:
    analytic = {
        "record_type": "expense",
        "entity": "account:research",
        "effective_at": f"2026-{month}-15T12:00:00+00:00",
        "dimensions": {"category": "compute", "currency": "CAD"},
        "measures": {"amount": amount},
        "fields": {"month": month},
    }
    if candidate_vendor:
        analytic["candidate_fields"] = {
            "vendor": {
                "value": "ComputeCo",
                "confidence": 0.6,
                "witness_event_ids": [event_id],
            }
        }
    return content_event(
        event_id,
        f"expense for {month} was {amount:.0f} CAD",
        "finance",
        minute,
        analytic=analytic,
    )


def claim_event(
    event_id: str,
    subject: str,
    relation: str,
    value,
    minute: int,
    text: str,
) -> Event:
    return content_event(
        event_id,
        text,
        "operations",
        minute,
        claim={
            "subject": subject,
            "relation": relation,
            "object": value,
            "conditions": ["current"],
        },
    )


def initial_events() -> list[Event]:
    events = [
        expense("expense-jan", "01", 100, 0, candidate_vendor=True),
        expense("expense-feb-old", "02", 999, 1),
        expense("expense-feb-new", "02", 120, 2),
        expense("expense-mar", "03", 140, 3),
        control_event(
            "retract-expense-feb-old",
            SpeechAct.RETRACTED,
            "expense-feb-old",
            0,
        ),
        control_event(
            "verify-expense-feb-new",
            SpeechAct.VERIFIED,
            "expense-feb-new",
            1,
        ),
        claim_event(
            "location-old",
            "project:wamrx",
            "office_location",
            "Montreal",
            10,
            "WAM-RX project office location was Montreal",
        ),
        claim_event(
            "location-new",
            "project:wamrx",
            "office_location",
            "Toronto",
            11,
            "WAM-RX project office location is Toronto",
        ),
        control_event("refute-location-old", SpeechAct.REFUTED, "location-old", 2),
        control_event("verify-location-new", SpeechAct.VERIFIED, "location-new", 3),
        claim_event(
            "a-security", "candidate-a", "security_verified", True, 20,
            "candidate-a security is verified",
        ),
        claim_event(
            "a-latency", "candidate-a", "latency_ms", 80, 21,
            "candidate-a latency is 80 milliseconds",
        ),
        claim_event(
            "a-cost", "candidate-a", "monthly_cost", 120, 22,
            "candidate-a monthly cost is 120",
        ),
        claim_event(
            "b-security", "candidate-b", "security_verified", False, 23,
            "candidate-b security is not verified",
        ),
        claim_event(
            "b-latency", "candidate-b", "latency_ms", 50, 24,
            "candidate-b latency is 50 milliseconds",
        ),
        claim_event(
            "b-cost", "candidate-b", "monthly_cost", 80, 25,
            "candidate-b monthly cost is 80",
        ),
        claim_event(
            "c-security", "candidate-c", "security_verified", True, 26,
            "candidate-c security is verified",
        ),
        claim_event(
            "c-cost", "candidate-c", "monthly_cost", 90, 27,
            "candidate-c monthly cost is 90",
        ),
    ]
    return events


REQUIREMENTS = (
    ConstraintRequirement(
        relation="security_verified",
        operator="equals",
        expected=True,
        next_step="verify candidate security",
    ),
    ConstraintRequirement(
        relation="latency_ms",
        operator="lte",
        expected=100,
        next_step="benchmark candidate latency",
    ),
    ConstraintRequirement(
        relation="monthly_cost",
        operator="lte",
        expected=100,
        next_step="obtain candidate cost",
    ),
)


def answer_in_hit(
    retrieval: HybridRetrievalIndex,
    *,
    query_id: str,
    query: str,
    expected_phrase: str,
) -> bool:
    result = retrieval.search(
        query,
        query_id=query_id,
        valid_at=VALID_AT,
        top_k=4,
    )
    expected = expected_phrase.lower()
    return any(expected in hit.text.lower() for hit in result.hits)


def regional_lineage_coverage(
    expected: dict[str, set[str]], observed: set[str]
) -> dict[str, float]:
    return {
        region: len(event_ids & observed) / len(event_ids) if event_ids else 1.0
        for region, event_ids in expected.items()
    }


def answer_envelope(
    store: AppendOnlyEventStore,
    *,
    artifact_id: str,
    content: dict,
    support,
    component: str,
) -> ArtifactEnvelope:
    stamp = ArtifactStamp.create(
        artifact_id=artifact_id,
        artifact_type="multiview-answer",
        content=content,
        store=store,
        base_weight_version="none",
        component_versions={component: "v1"},
        ontology_version="v1",
        verifier_version="e0.11-v1",
        build_config={"registered_experiment": "E0.11"},
    )
    return ArtifactEnvelope(content=content, stamp=stamp, support=support)


def analytic_journal_is_immutable(store: AppendOnlyEventStore) -> bool:
    statements = (
        "UPDATE analytic_query_journal SET record_json = record_json",
        "DELETE FROM analytic_query_journal",
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


def malformed_schema_is_rejected(path: pathlib.Path) -> bool:
    store = AppendOnlyEventStore(path)
    store.append(
        content_event(
            "malformed-analytic",
            "malformed analytic control",
            "finance",
            40,
            analytic={
                "record_type": "expense",
                "entity": "account:bad",
                "effective_at": "2026-01-01T00:00:00+00:00",
                "measures": {"amount": "not-a-number"},
            },
        )
    )
    try:
        AnalyticMemory.build(
            store,
            artifact_id="malformed-analytic-control",
            valid_at=VALID_AT,
        )
    except AnalyticCompileError:
        return True
    return False


def live_contradiction_is_unresolved(path: pathlib.Path) -> bool:
    store = AppendOnlyEventStore(path)
    store.append_batch(
        [
            *initial_events(),
            claim_event(
                "c-latency-good",
                "candidate-c",
                "latency_ms",
                95,
                40,
                "candidate-c latency is 95 milliseconds",
            ),
            claim_event(
                "c-latency-conflict",
                "candidate-c",
                "latency_ms",
                150,
                41,
                "candidate-c latency is 150 milliseconds",
            ),
        ]
    )
    graph = BeliefGraph.build(
        store,
        artifact_id="live-contradiction-control",
        valid_at=VALID_AT,
    )
    result = graph.evaluate("candidate-c", REQUIREMENTS, valid_at=VALID_AT)
    return (
        result.status == "unresolved"
        and result.conflicting_constraints == ("latency_ms",)
    )


def derived_plan_laundering_is_rejected(
    store: AppendOnlyEventStore,
    plan: str,
) -> bool:
    content = {"generated_next_step": plan}
    stamp = ArtifactStamp.create(
        artifact_id="derived-plan-laundering-control",
        artifact_type="negative-control",
        content=content,
        store=store,
        base_weight_version="none",
        component_versions={"belief-graph": "v1"},
        ontology_version="v1",
        verifier_version="e0.11-v1",
        build_config={"negative_control": "derived-evidence-laundering"},
    )
    envelope = ArtifactEnvelope(
        content=content,
        stamp=stamp,
        support=SupportManifest.create(
            supporting_event_ids=[plan],
            candidate_event_ids=[plan],
        ),
    )
    try:
        envelope.validate(
            store,
            compatibility_policy=ArtifactCompatibilityPolicy.exact_for_stamp(stamp),
            valid_at=VALID_AT,
        )
    except InvalidArtifactError:
        return True
    return False


def run() -> dict:
    contracts = load_contracts(ROOT / "contracts" / "wamrx_multiview_memory.json")
    with tempfile.TemporaryDirectory(prefix="wamrx-e0-11-") as directory:
        temp = pathlib.Path(directory)
        store = AppendOnlyEventStore(temp / "world.sqlite")
        events = initial_events()
        store.append_batch(events)

        analytic = AnalyticMemory.build(
            store, artifact_id="analytic-pre-delete", valid_at=VALID_AT
        )
        graph_before_latency = BeliefGraph.build(
            store, artifact_id="graph-before-latency", valid_at=VALID_AT
        )
        unresolved = graph_before_latency.select_subjects(
            ("candidate-a", "candidate-b", "candidate-c"),
            REQUIREMENTS,
            valid_at=VALID_AT,
        )
        c_unresolved = next(
            item for item in unresolved.evaluations if item.subject == "candidate-c"
        )

        latency_c = claim_event(
            "c-latency",
            "candidate-c",
            "latency_ms",
            95,
            28,
            "candidate-c latency is 95 milliseconds",
        )
        store.append(latency_c)
        graph = BeliefGraph.build(
            store, artifact_id="graph-complete", valid_at=VALID_AT
        )
        retrieval = HybridRetrievalIndex.build(
            store, artifact_id="retrieval-complete", valid_at=VALID_AT
        )

        total = analytic.aggregate(
            "amount",
            query_id="analytic-total-pre-delete",
            operation="sum",
            valid_at=VALID_AT,
            record_type="expense",
            entity="account:research",
        )
        comparison = analytic.compare_windows(
            "amount",
            query_id="analytic-comparison-pre-delete",
            valid_at=VALID_AT,
            left=("2026-01-01T00:00:00+00:00", "2026-02-01T00:00:00+00:00"),
            right=("2026-03-01T00:00:00+00:00", "2026-04-01T00:00:00+00:00"),
            record_type="expense",
            entity="account:research",
        )
        contradictions = graph.contradictions(valid_at=VALID_AT)
        location = next(
            item
            for item in contradictions
            if item.subject == "project:wamrx" and item.relation == "office_location"
        )
        selection = graph.select_subjects(
            ("candidate-a", "candidate-b", "candidate-c"),
            REQUIREMENTS,
            valid_at=VALID_AT,
        )
        c_evaluation = next(
            item for item in selection.evaluations if item.subject == "candidate-c"
        )

        exact_tasks = {
            "aggregation": total.value == 360.0,
            "temporal_comparison": comparison.value == {
                "left": 100.0,
                "right": 140.0,
                "delta": 40.0,
            },
            "contradiction": (
                location.status == "resolved"
                and {edge.object for edge in location.edges} == {"Montreal", "Toronto"}
                and [
                    edge.object
                    for edge in location.edges
                    if edge.active
                ]
                == ["Toronto"]
            ),
            "multi_constraint": selection.eligible_subjects == ("candidate-c",),
        }
        baseline_tasks = {
            "aggregation": answer_in_hit(
                retrieval,
                query_id="baseline-aggregation",
                query="total expense January February March",
                expected_phrase="total expense was 360",
            ),
            "temporal_comparison": answer_in_hit(
                retrieval,
                query_id="baseline-comparison",
                query="expense change January to March",
                expected_phrase="increase by 40",
            ),
            "contradiction": answer_in_hit(
                retrieval,
                query_id="baseline-location",
                query="current WAM-RX project office location",
                expected_phrase="Toronto",
            ),
            "multi_constraint": answer_in_hit(
                retrieval,
                query_id="baseline-constraint",
                query="eligible candidate security latency cost",
                expected_phrase="eligible candidate is candidate-c",
            ),
        }

        observed_support = set(total.support.supporting_event_ids)
        observed_support.update(comparison.support.supporting_event_ids)
        observed_support.update(
            event_id for edge in location.edges for event_id in edge.support.supporting_event_ids
        )
        observed_support.update(c_evaluation.support.supporting_event_ids)
        expected_support = {
            "finance": {"expense-jan", "expense-feb-new", "expense-mar"},
            "operations": {
                "location-old",
                "location-new",
                "c-security",
                "c-latency",
                "c-cost",
            },
        }
        regional_coverage = regional_lineage_coverage(expected_support, observed_support)
        tail_blind_coverage = regional_lineage_coverage(
            expected_support, observed_support - expected_support["operations"]
        )

        raw_sum = sum(
            float(event.payload["analytic"]["measures"]["amount"])
            for event in events
            if "analytic" in event.payload
        )
        active_only_has_history = any(
            len(
                {
                    json.dumps(edge.object, sort_keys=True)
                    for edge in graph.claims_for(
                        valid_at=VALID_AT,
                        subject="project:wamrx",
                        relation="office_location",
                        active_only=True,
                    )
                }
            )
            > 1
            for _ in (0,)
        )

        candidate_vendor = next(
            row.candidate_fields["vendor"]
            for row in analytic.rows
            if row.event_id == "expense-jan"
        )
        generated_plans_are_evidence = any(
            plan in c_unresolved.support.candidate_event_ids
            for plan in c_unresolved.next_steps
        )

        analytic_answer = answer_envelope(
            store,
            artifact_id="analytic-answer-before-delete",
            content={"total": total.value},
            support=total.support,
            component="analytic",
        )
        graph_answer = answer_envelope(
            store,
            artifact_id="graph-answer-before-delete",
            content={"eligible": list(selection.eligible_subjects)},
            support=c_evaluation.support,
            component="belief-graph",
        )
        analytic_policy = ArtifactCompatibilityPolicy.exact_for_stamp(
            analytic_answer.stamp
        )
        graph_policy = ArtifactCompatibilityPolicy.exact_for_stamp(graph_answer.stamp)
        analytic_answer.validate(
            store, compatibility_policy=analytic_policy, valid_at=VALID_AT
        )
        graph_answer.validate(
            store, compatibility_policy=graph_policy, valid_at=VALID_AT
        )

        store.append_batch(
            [
                control_event(
                    "tombstone-expense-mar",
                    SpeechAct.TOMBSTONE,
                    "expense-mar",
                    4,
                ),
                control_event(
                    "tombstone-c-latency",
                    SpeechAct.TOMBSTONE,
                    "c-latency",
                    5,
                ),
            ]
        )
        stale_total = analytic.aggregate(
            "amount",
            query_id="analytic-total-post-delete",
            operation="sum",
            valid_at=VALID_AT,
            record_type="expense",
            entity="account:research",
        )
        stale_selection = graph.select_subjects(
            ("candidate-c",), REQUIREMENTS, valid_at=VALID_AT
        )
        stale_c = stale_selection.evaluations[0]

        answer_rejections = []
        for envelope, policy in (
            (analytic_answer, analytic_policy),
            (graph_answer, graph_policy),
        ):
            try:
                envelope.validate(
                    store,
                    compatibility_policy=policy,
                    valid_at=VALID_AT,
                )
                answer_rejections.append(False)
            except InvalidArtifactError:
                answer_rejections.append(True)

        repaired_analytic = AnalyticMemory.build(
            store, artifact_id="analytic-repaired", valid_at=VALID_AT
        )
        repaired_graph = BeliefGraph.build(
            store, artifact_id="graph-repaired", valid_at=VALID_AT
        )
        clean_store = AppendOnlyEventStore(temp / "clean.sqlite")
        clean_store.append_batch(store.events())
        clean_analytic = AnalyticMemory.build(
            clean_store, artifact_id="analytic-clean", valid_at=VALID_AT
        )
        clean_graph = BeliefGraph.build(
            clean_store, artifact_id="graph-clean", valid_at=VALID_AT
        )

        alternate_ontology = AnalyticMemory.build(
            store,
            artifact_id="analytic-ontology-v2",
            valid_at=VALID_AT,
            ontology_version="v2",
        )

        stale_ontology_rejected = False
        try:
            repaired_analytic.envelope.validate(
                store,
                compatibility_policy=ArtifactCompatibilityPolicy.exact_for_stamp(
                    alternate_ontology.envelope.stamp
                ),
                valid_at=VALID_AT,
                check_support=False,
            )
        except InvalidArtifactError:
            stale_ontology_rejected = True

        malformed_schema_rejected = malformed_schema_is_rejected(
            temp / "malformed.sqlite"
        )
        live_contradiction_unresolved = live_contradiction_is_unresolved(
            temp / "contradiction.sqlite"
        )
        derived_laundering_rejected = derived_plan_laundering_is_rejected(
            store,
            c_unresolved.next_steps[0],
        )

        analytic_journals = store.analytic_query_records()
        registered_query_ids = {
            "analytic-total-pre-delete",
            "analytic-comparison-pre-delete",
            "analytic-total-post-delete",
        }
        journal_query_ids = {record["query_id"] for record in analytic_journals}
        required_journal_keys = {
            "query_id",
            "operation",
            "valid_at",
            "parameters",
            "artifact_id",
            "artifact_content_hash",
            "artifact_frontier",
            "decision_frontier",
            "component_versions",
            "ontology_version",
            "candidates",
            "selected_row_ids",
            "support",
            "result",
            "result_hash",
        }
        analytic_source_ids = {
            "expense-jan",
            "expense-feb-old",
            "expense-feb-new",
            "expense-mar",
        }
        journal_records_complete = all(
            required_journal_keys <= set(record)
            and record["result_hash"] == sha256_json(record["result"])
            and analytic_source_ids
            <= {candidate["event_id"] for candidate in record["candidates"]}
            for record in analytic_journals
        )
        missing_real_queries = registered_query_ids - journal_query_ids
        unjournaled_control_missing = (
            registered_query_ids | {"deliberately-unjournaled-control"}
        ) - journal_query_ids

        multiview_score = sum(exact_tasks.values())
        baseline_score = sum(baseline_tasks.values())
        checks = {
            "K1_structured_schema_and_lineage_enforced": (
                malformed_schema_rejected
                and all(
                    row.support.supporting_event_ids == (row.event_id,)
                    for row in analytic.rows
                )
                and all(
                    edge.support.supporting_event_ids == (edge.event_id,)
                    for edge in graph.edges
                )
            ),
            "K2_corrections_and_temporal_math_exact": exact_tasks["aggregation"]
            and exact_tasks["temporal_comparison"],
            "K3_candidate_field_witness_preserved": (
                candidate_vendor.confidence == 0.6
                and candidate_vendor.witness_event_ids == ("expense-jan",)
            ),
            "K4_contradiction_history_preserved": exact_tasks["contradiction"],
            "K5_missing_evidence_not_satisfied": (
                c_unresolved.status == "unresolved"
                and c_unresolved.missing_evidence == ("latency_ms",)
                and c_unresolved.next_steps == ("benchmark candidate latency",)
            ),
            "K6_generated_plan_not_evidence": not generated_plans_are_evidence,
            "K7_all_multiview_tasks_exact": multiview_score == 4,
            "K8_gain_over_named_baseline": multiview_score - baseline_score >= 2,
            "K9_worst_region_lineage_complete": min(regional_coverage.values()) == 1.0,
            "K10_immediate_support_disable": (
                stale_total.value == 220.0
                and stale_c.status == "unresolved"
                and stale_c.missing_evidence == ("latency_ms",)
            ),
            "K11_durable_answers_invalidated": all(answer_rejections),
            "K12_clean_rebuild_equivalent": (
                repaired_analytic.content_hash == clean_analytic.content_hash
                and repaired_graph.content_hash == clean_graph.content_hash
            ),
            "K13_ontology_identity_stamped": (
                repaired_analytic.envelope.stamp.ontology_version == "v1"
                and alternate_ontology.envelope.stamp.ontology_version == "v2"
                and repaired_analytic.envelope.stamp.to_dict()
                != alternate_ontology.envelope.stamp.to_dict()
            ),
            "K14_analytic_query_journals_complete": (
                not missing_real_queries
                and journal_records_complete
                and analytic_journal_is_immutable(store)
            ),
        }
        manipulations = {
            "M1_raw_payload_sum_is_wrong": raw_sum == 1359.0 and raw_sum != total.value,
            "M2_active_only_drops_history": (
                not active_only_has_history and len(location.edges) == 2
            ),
            "M3_tail_blind_lineage_fails": tail_blind_coverage["operations"] == 0.0,
            "M4_unjournaled_query_detected": unjournaled_control_missing
            == {"deliberately-unjournaled-control"},
            "M5_malformed_schema_rejected": malformed_schema_rejected,
            "M6_unregistered_stale_ontology_rejected": stale_ontology_rejected,
            "M7_live_contradiction_blocks_satisfaction": (
                live_contradiction_unresolved
            ),
            "M8_derived_evidence_laundering_rejected": (
                derived_laundering_rejected
            ),
        }
        return {
            "experiment": "E0.11",
            "claim": "analytic and belief-graph views add exact multiview memory operations",
            "contract_count": len(contracts),
            "exact_tasks": exact_tasks,
            "retrieval_answer_in_hit_baseline": baseline_tasks,
            "multiview_score": multiview_score,
            "baseline_score": baseline_score,
            "regional_lineage_coverage": regional_coverage,
            "tail_blind_control_coverage": tail_blind_coverage,
            "pre_delete_total": total.value,
            "post_delete_stale_view_total": stale_total.value,
            "analytic_query_journal_count": len(analytic_journals),
            "checks": checks,
            "manipulation_checks": manipulations,
            "verdict": (
                "PASS"
                if all(checks.values()) and all(manipulations.values())
                else "FAIL"
            ),
        }


def main() -> int:
    result = run()
    path = ROOT / "results" / "e0_11_multiview_memory.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
