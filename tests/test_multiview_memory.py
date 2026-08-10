from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rig_a.experiments.e0_11_multiview_memory import (
    REQUIREMENTS,
    VALID_AT,
    claim_event,
    content_event,
    initial_events,
    run,
)
from wamrx.analytic import AnalyticCompileError, AnalyticMemory
from wamrx.belief_graph import BeliefGraph
from wamrx.store import AppendOnlyEventStore, EventConflictError


class AnalyticMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = AppendOnlyEventStore(Path(self.temp.name) / "ledger.sqlite")
        self.store.append_batch(initial_events())
        self.analytic = AnalyticMemory.build(
            self.store, artifact_id="analytic-test", valid_at=VALID_AT
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_filter_aggregation_grouping_trend_and_rank_are_journaled(self) -> None:
        filtered = self.analytic.filter(
            query_id="filter",
            valid_at=VALID_AT,
            record_type="expense",
            entity="account:research",
        )
        self.assertEqual(len(filtered.rows), 3)
        total = self.analytic.aggregate(
            "amount",
            query_id="sum",
            operation="sum",
            valid_at=VALID_AT,
            record_type="expense",
        )
        self.assertEqual(total.value, 360.0)
        grouped = self.analytic.group_aggregate(
            "amount",
            query_id="group",
            group_by="category",
            operation="sum",
            valid_at=VALID_AT,
            record_type="expense",
        )
        self.assertEqual(grouped.value, {"compute": 360.0})
        trend = self.analytic.trend(
            "amount",
            query_id="trend",
            valid_at=VALID_AT,
            record_type="expense",
        )
        self.assertEqual(trend.value["direction"], "up")
        ranked = self.analytic.rank_groups(
            "amount",
            query_id="rank",
            group_by="category",
            valid_at=VALID_AT,
            record_type="expense",
        )
        self.assertEqual(ranked.value, [("compute", 360.0)])
        journals = self.store.analytic_query_records()
        self.assertEqual(len(journals), 5)
        self.assertTrue(
            all(len(record["candidates"]) == 4 for record in journals)
        )

    def test_duplicate_query_id_cannot_change_record(self) -> None:
        self.analytic.aggregate(
            "amount",
            query_id="same",
            operation="sum",
            valid_at=VALID_AT,
            record_type="expense",
        )
        with self.assertRaises(EventConflictError):
            self.analytic.aggregate(
                "amount",
                query_id="same",
                operation="mean",
                valid_at=VALID_AT,
                record_type="expense",
            )

    def test_malformed_measure_fails_compilation(self) -> None:
        store = AppendOnlyEventStore(Path(self.temp.name) / "malformed.sqlite")
        store.append(
            content_event(
                "bad-analytic",
                "bad analytic row",
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
        with self.assertRaises(AnalyticCompileError):
            AnalyticMemory.build(
                store, artifact_id="bad", valid_at=VALID_AT
            )

    def test_effective_times_normalize_before_order_sensitive_queries(self) -> None:
        store = AppendOnlyEventStore(Path(self.temp.name) / "offsets.sqlite")
        first = {
            "record_type": "reading",
            "entity": "sensor:1",
            "effective_at": "2026-01-01T08:00:00-04:00",
            "measures": {"value": 1},
        }
        later = {
            "record_type": "reading",
            "entity": "sensor:1",
            "effective_at": "2026-01-01T13:00:00+00:00",
            "measures": {"value": 2},
        }
        store.append_batch(
            [
                content_event(
                    "offset-first", "first", "finance", 42, analytic=first
                ),
                content_event(
                    "offset-later", "later", "finance", 43, analytic=later
                ),
            ]
        )
        analytic = AnalyticMemory.build(
            store, artifact_id="offsets", valid_at=VALID_AT
        )
        self.assertEqual(
            [row.effective_at for row in analytic.rows],
            [
                "2026-01-01T12:00:00+00:00",
                "2026-01-01T13:00:00+00:00",
            ],
        )
        trend = analytic.trend(
            "value",
            query_id="offset-trend",
            valid_at=VALID_AT,
            record_type="reading",
        )
        self.assertEqual(trend.value["direction"], "up")


class BeliefGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = AppendOnlyEventStore(Path(self.temp.name) / "graph.sqlite")
        self.store.append_batch(initial_events())

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_resolved_contradiction_is_preserved(self) -> None:
        graph = BeliefGraph.build(
            self.store, artifact_id="graph", valid_at=VALID_AT
        )
        contradiction = next(
            item
            for item in graph.contradictions(valid_at=VALID_AT)
            if item.relation == "office_location"
        )
        self.assertEqual(contradiction.status, "resolved")
        self.assertEqual(
            {edge.object for edge in contradiction.edges},
            {"Montreal", "Toronto"},
        )

    def test_missing_and_conflicting_constraints_are_not_satisfied(self) -> None:
        graph = BeliefGraph.build(
            self.store, artifact_id="graph-missing", valid_at=VALID_AT
        )
        missing = graph.evaluate("candidate-c", REQUIREMENTS, valid_at=VALID_AT)
        self.assertEqual(missing.status, "unresolved")
        self.assertEqual(missing.missing_evidence, ("latency_ms",))

        self.store.append_batch(
            [
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
        conflicting_graph = BeliefGraph.build(
            self.store, artifact_id="graph-conflict", valid_at=VALID_AT
        )
        conflict = conflicting_graph.evaluate(
            "candidate-c", REQUIREMENTS, valid_at=VALID_AT
        )
        self.assertEqual(conflict.status, "unresolved")
        self.assertEqual(conflict.conflicting_constraints, ("latency_ms",))
        self.assertNotIn("benchmark candidate latency", conflict.support.candidate_event_ids)


class MultiviewExperimentTests(unittest.TestCase):
    def test_registered_e0_11(self) -> None:
        result = run()
        self.assertEqual(result["verdict"], "PASS")
        self.assertTrue(all(result["checks"].values()))
        self.assertTrue(all(result["manipulation_checks"].values()))
        self.assertEqual(result["multiview_score"], 4)
        self.assertEqual(result["baseline_score"], 1)


if __name__ == "__main__":
    unittest.main()
