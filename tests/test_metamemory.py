from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import tempfile
import unittest

from rig_a.experiments.e0_17_metamemory_boundary import (
    VALID_AT,
    _store,
    _tombstone,
    run,
)
from wamrx.metamemory import (
    DeterministicMetamemoryPolicy,
    MetamemoryCapacityExceeded,
    MetamemoryError,
    build_compression_artifact,
    build_skill_candidate,
    validate_compression_artifact,
    validate_decision,
    validate_skill_candidate,
    with_replaced_artifact,
)
from wamrx.metamemory_tasks import FAMILIES, generate_family, verify_frozen_splits


ROOT = Path(__file__).resolve().parents[1]


class MetamemoryBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = json.loads(
            (ROOT / "contracts" / "wamrx_metamemory_splits_v1.json").read_text()
        )
        cls.tasks = tuple(
            generate_family("train", family, cls.registry["splits"]["train"])[0]
            for family in FAMILIES
        )

    def test_frozen_splits_are_exact_and_ood_disjoint(self) -> None:
        report = verify_frozen_splits(self.registry)
        self.assertTrue(report["passed"])
        for split, expected in (("train", 56), ("id", 28), ("ood", 28)):
            tasks = [
                task
                for family in FAMILIES
                for task in generate_family(split, family, self.registry["splits"][split])
            ]
            self.assertEqual(len(tasks), expected)

    def test_all_actions_are_candidate_only_and_accounted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = _store(Path(directory) / "ledger.sqlite", self.tasks)
            before = store.count()
            policy = DeterministicMetamemoryPolicy()
            decisions = [policy.decide(task) for task in self.tasks]
            for task, decision in zip(self.tasks, decisions):
                validate_decision(store, task, decision, valid_at=VALID_AT)
                self.assertFalse(decision.direct_ledger_write_authorized)
            self.assertEqual(store.count(), before)
            self.assertEqual(
                set(policy.compute_record().operation_counts),
                {"ignore", "stage", "link", "retrieve", "summarize", "structure", "request_evidence"},
            )

    def test_compression_preserves_protected_fields_and_deletes_stale(self) -> None:
        task = next(
            item for item in self.tasks if item.family == "redundant_history_compression"
        )
        with tempfile.TemporaryDirectory() as directory:
            store = _store(Path(directory) / "ledger.sqlite", (task,))
            artifact = build_compression_artifact(store, task, valid_at=VALID_AT)
            validate_compression_artifact(store, task, artifact, valid_at=VALID_AT)
            rebuilt = build_compression_artifact(store, task, valid_at=VALID_AT)
            self.assertEqual(artifact, rebuilt)
            with self.assertRaises(MetamemoryError):
                validate_compression_artifact(
                    store,
                    task,
                    with_replaced_artifact(artifact, unresolved_item_ids=()),
                    valid_at=VALID_AT,
                )
            store.append(_tombstone(artifact.supporting_event_ids[0]))
            with self.assertRaises(MetamemoryError):
                validate_compression_artifact(store, task, artifact, valid_at=VALID_AT)

    def test_skill_candidate_never_promotes_itself(self) -> None:
        task = next(item for item in self.tasks if item.family == "procedure_repetition")
        with tempfile.TemporaryDirectory() as directory:
            store = _store(Path(directory) / "ledger.sqlite", (task,))
            candidate = build_skill_candidate(store, task, valid_at=VALID_AT)
            validate_skill_candidate(store, task, candidate, valid_at=VALID_AT)
            self.assertFalse(candidate.promotion_authority)
            with self.assertRaises(MetamemoryError):
                validate_skill_candidate(
                    store,
                    task,
                    dataclasses.replace(candidate, promotion_authority=True),
                    valid_at=VALID_AT,
                )

    def test_serialized_capacity_fails_closed(self) -> None:
        task = next(
            item for item in self.tasks if item.family == "redundant_history_compression"
        )
        with tempfile.TemporaryDirectory() as directory:
            store = _store(Path(directory) / "ledger.sqlite", (task,))
            artifact = build_compression_artifact(store, task, valid_at=VALID_AT)
            with self.assertRaises(MetamemoryCapacityExceeded):
                validate_compression_artifact(
                    store,
                    task,
                    with_replaced_artifact(artifact, summary_text="x" * 33000),
                    valid_at=VALID_AT,
                )

    def test_registered_e0_17_passes_without_metrics(self) -> None:
        result = run(write_result=False)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["accuracy_metrics_read"], 0)
        self.assertEqual(result["learned_policy_status"], "NOT_RUN")
        self.assertTrue(all(result["checks"].values()))
        self.assertTrue(all(result["manipulations"].values()))


if __name__ == "__main__":
    unittest.main()
