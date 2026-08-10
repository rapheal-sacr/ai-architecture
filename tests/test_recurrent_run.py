from __future__ import annotations

import pathlib
import json
import unittest

from wamrx.recurrent_run import (
    RecurrentRunError,
    RunComputeRecord,
    holm_adjusted_tests,
    paired_student_t,
    resolve_terminal_status,
    student_t_quantile,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


class RecurrentRunContractTests(unittest.TestCase):
    def test_registered_student_t_critical_values(self) -> None:
        self.assertAlmostEqual(student_t_quantile(0.95, 4), 2.1318468, places=6)
        self.assertAlmostEqual(student_t_quantile(0.975, 4), 2.7764451, places=6)

    def test_paired_unit_and_holm_are_deterministic(self) -> None:
        positive = [0.08, 0.07, 0.09, 0.06, 0.10]
        test = paired_student_t(positive, null_margin=0.0, alpha=0.05)
        self.assertEqual(test["pairs"], 5)
        self.assertGreater(test["one_sided_lower_bound"], 0.0)
        json.dumps(paired_student_t([0.1] * 5, null_margin=0.0, alpha=0.05), allow_nan=False)
        family = holm_adjusted_tests(
            {
                "algorithmic": (positive, 0.0),
                "structured": ([0.04, 0.05, 0.03, 0.06, 0.05], 0.0),
                "multiview": ([-0.01, 0.0, 0.01, 0.0, -0.01], 0.0),
            }
        )
        self.assertEqual(
            family["procedure"], "Holm step-down one-sided paired Student-t"
        )
        self.assertFalse(
            family["hypotheses"]["multiview"]["holm_rejects_null"]
        )

    def test_run_compute_record_separates_estimated_and_realized(self) -> None:
        record = RunComputeRecord(
            run_id="preflight-v1",
            arm_id="flat-recurrent-v1",
            seed=41001,
            status="INCOMPLETE",
            parameter_count=8_372_224,
            schedule_hash="a" * 64,
            estimated_training_flops=1000,
            realized_training_flops=500,
            estimated_inference_flops=200,
            realized_inference_flops=120,
            training_examples_planned=320,
            training_examples_seen=160,
            optimizer_updates_planned=20,
            optimizer_updates_completed=10,
            evaluation_examples_planned=0,
            evaluation_examples_completed=0,
        )
        self.assertEqual(record.to_dict()["realized_training_flops"], 500)
        with self.assertRaises(RecurrentRunError):
            RunComputeRecord(
                **{
                    **record.__dict__,
                    "realized_training_flops": 1001,
                }
            ).validate()

    def test_status_precedence_never_silently_omits_failure(self) -> None:
        self.assertEqual(
            resolve_terminal_status(
                invalid_reasons=["NaN"],
                incomplete_reasons=["missing seed"],
                decision="COMPLETE_ADOPT_FLAT",
            ),
            "INVALID",
        )
        self.assertEqual(
            resolve_terminal_status(
                invalid_reasons=[], incomplete_reasons=["interrupted"], decision=None
            ),
            "INCOMPLETE",
        )
        self.assertEqual(
            resolve_terminal_status(
                invalid_reasons=[], incomplete_reasons=[], decision=None
            ),
            "NOT_RUN",
        )

    def test_run_contract_and_schema_exist(self) -> None:
        self.assertTrue((ROOT / "contracts" / "wamrx_recurrent_run_v1.json").is_file())
        self.assertTrue((ROOT / "schemas" / "wamrx_recurrent_run.schema.json").is_file())


if __name__ == "__main__":
    unittest.main()
