from __future__ import annotations

import json
from pathlib import Path
import unittest

try:
    import mlx.core as mx
except (ImportError, OSError):  # pragma: no cover - optional dependency boundary.
    mx = None

if mx is not None:
    from wamrx.native_memory_model import (
        GATE_PARAMETER_COUNT,
        MinimalOperationGate,
        load_frozen_core,
        model_parameter_hash,
        run_gate_preflight,
    )
    from wamrx.native_memory_comparison import (
        build_run_manifest,
        promotion_audit,
    )
    from wamrx.native_memory_run import (
        ARM_IDS,
        build_gate_schedule,
        split_gate_examples,
    )
    from wamrx.recurrent_model import parameter_count


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(mx is None, "MLX runtime is optional")
class NativeMemoryModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(
            (ROOT / "contracts" / "wamrx_native_memory_run_v1.json").read_text()
        )
        cls.registry = json.loads(
            (ROOT / "contracts" / "wamrx_native_memory_splits_v1.json").read_text()
        )

    def test_gate_has_exact_registered_parameter_count(self) -> None:
        mx.random.seed(112001)
        gate = MinimalOperationGate()
        self.assertEqual(parameter_count(gate), GATE_PARAMETER_COUNT)
        decisions = gate.decide(mx.zeros((3, 472)))
        self.assertEqual(len(decisions), 3)

    def test_registered_core_loads_without_mutation(self) -> None:
        core, identity = load_frozen_core(ROOT, 41001)
        self.assertEqual(identity.parameter_count, 8388608)
        self.assertEqual(identity.macro_depth, 4)
        self.assertEqual(model_parameter_hash(core), identity.model_parameter_hash)

    def test_two_update_preflight_resumes_bitwise_without_metrics(self) -> None:
        examples = split_gate_examples(self.registry, "train")
        training = self.contract["gate_training"]
        schedule = build_gate_schedule(
            examples,
            seed=41001,
            updates=training["optimizer_updates"],
            examples_per_operation=training["examples_per_operation_per_batch"],
            schedule_seed_offset=training["schedule_seed_offset"],
        )
        report = run_gate_preflight(
            ROOT,
            seed=41001,
            examples=examples,
            schedule=schedule,
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["accuracy_metrics_read"], 0)
        self.assertEqual(
            report["continuous_state_hash"], report["resumed_state_hash"]
        )
        self.assertEqual(report["core_hash_before"], report["core_hash_after"])

    def test_terminal_result_manifest_matches_frozen_implementation(self) -> None:
        result = json.loads(
            (ROOT / "results" / "e0_16_native_memory_comparison.json").read_text()
        )
        self.assertEqual(result["status"], "COMPLETE_RETAIN_EXPLICIT_MULTIVIEW")
        self.assertEqual(result["manifest"], build_run_manifest(ROOT))

    @staticmethod
    def _synthetic_audit_inputs(*, learned_wins: bool = False):
        seeds = (41001, 41002, 41003, 41004, 41005)
        metrics = []
        scores = {
            "explicit-multiview-v1": 0.5,
            "deterministic-session-cache-v1": 0.7,
            "minimal-gated-native-memory-v1": 0.9 if learned_wins else 0.4,
        }
        for arm_id in ARM_IDS:
            for seed in seeds:
                for split in ("id", "ood"):
                    for grouping, label in (
                        ("overall", "all"),
                        ("family", "correction"),
                        ("region", "operations"),
                    ):
                        score = scores[arm_id]
                        metrics.append(
                            {
                                "arm_id": arm_id,
                                "seed": seed,
                                "split": split,
                                "grouping": grouping,
                                "label": label,
                                "accuracy": score,
                                "compute_normalized_score": score,
                            }
                        )
        rows = []
        for arm_id in ARM_IDS:
            for seed in seeds:
                rows.extend(
                    [
                        {
                            "arm_id": arm_id,
                            "seed": seed,
                            "family": "correction",
                            "correct": learned_wins or arm_id != "minimal-gated-native-memory-v1",
                            "total_inference_flops": 100,
                            "unsupported_durable_writes": 0,
                        },
                        {
                            "arm_id": arm_id,
                            "seed": seed,
                            "family": "reset_task_switch",
                            "correct": True,
                            "prediction": None,
                            "pre_reset_values": ["old"],
                            "total_inference_flops": 100,
                            "unsupported_durable_writes": 0,
                        },
                    ]
                )
        probes = [
            {
                "capacity_saturation_curve": {
                    arm_id: [
                        {"occupancy": value, "storage_bytes": 0}
                        for value in (0, 4, 8, 12, 16)
                    ]
                    for arm_id in ARM_IDS
                },
                "post_invalidation_performance": {
                    arm_id: {
                        "had_slot": True,
                        "disabled_after_tombstone": True,
                        "stale_value_emitted": False,
                    }
                    for arm_id in ARM_IDS
                },
            }
            for _ in seeds
        ]
        return seeds, metrics, rows, probes

    def test_promotion_audit_enforces_cache_noninferiority(self) -> None:
        seeds, metrics, rows, probes = self._synthetic_audit_inputs()
        audit = promotion_audit(
            metrics,
            rows,
            seeds=seeds,
            manipulations={"M1": True, "M2": True},
            secondary_probes=probes,
        )
        self.assertEqual(audit["decision"], "COMPLETE_RETAIN_DETERMINISTIC_CACHE")
        self.assertTrue(audit["registered_gates"]["cache_id_pass"])
        self.assertTrue(audit["registered_gates"]["cache_protected_pass"])

    def test_failed_registered_manipulation_is_invalid(self) -> None:
        seeds, metrics, rows, probes = self._synthetic_audit_inputs()
        audit = promotion_audit(
            metrics,
            rows,
            seeds=seeds,
            manipulations={"M1": True, "M2": False},
            secondary_probes=probes,
        )
        self.assertEqual(audit["decision"], "INVALID")

    def test_learned_promotion_requires_correction_and_compute_gates(self) -> None:
        seeds, metrics, rows, probes = self._synthetic_audit_inputs(
            learned_wins=True
        )
        audit = promotion_audit(
            metrics,
            rows,
            seeds=seeds,
            manipulations={"M1": True, "M2": True},
            secondary_probes=probes,
        )
        self.assertEqual(audit["decision"], "COMPLETE_ADOPT_LEARNED_MEMORY")
        self.assertTrue(audit["registered_gates"]["learned_correction_pass"])

if __name__ == "__main__":
    unittest.main()
