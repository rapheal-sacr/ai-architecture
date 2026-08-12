from __future__ import annotations

import json
from pathlib import Path
import unittest

from wamrx.native_memory_run import (
    OPERATION_ORDER,
    build_gate_schedule,
    gate_schedule_hash,
    split_gate_examples,
    validate_run_registration,
)


ROOT = Path(__file__).resolve().parents[1]


class NativeMemoryRunRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(
            (ROOT / "contracts" / "wamrx_native_memory_run_v1.json").read_text()
        )
        cls.registry = json.loads(
            (ROOT / "contracts" / "wamrx_native_memory_splits_v1.json").read_text()
        )

    def test_registered_examples_and_schedules_are_exact(self) -> None:
        report = validate_run_registration(ROOT)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["accuracy_metrics_read"], 0)
        self.assertEqual(report["gate_parameter_count"], 1892)
        self.assertEqual(report["examples"]["train"]["count"], 137)

    def test_all_four_operations_have_frozen_training_examples(self) -> None:
        examples = split_gate_examples(self.registry, "train")
        self.assertEqual(
            {example.target_operation for example in examples},
            set(OPERATION_ORDER),
        )
        self.assertTrue(
            all("expected" not in example.input for example in examples)
        )

    def test_schedule_is_balanced_per_batch_and_repeatable(self) -> None:
        examples = split_gate_examples(self.registry, "train")
        training = self.contract["gate_training"]
        kwargs = {
            "seed": 41001,
            "updates": training["optimizer_updates"],
            "examples_per_operation": training["examples_per_operation_per_batch"],
            "schedule_seed_offset": training["schedule_seed_offset"],
        }
        first = build_gate_schedule(examples, **kwargs)
        second = build_gate_schedule(examples, **kwargs)
        self.assertEqual(gate_schedule_hash(first), gate_schedule_hash(second))
        by_id = {example.example_id: example for example in examples}
        for batch in first:
            counts = {operation: 0 for operation in OPERATION_ORDER}
            for example_id in batch.example_ids:
                counts[by_id[example_id].target_operation] += 1
            self.assertEqual(set(counts.values()), {4})


if __name__ == "__main__":
    unittest.main()
