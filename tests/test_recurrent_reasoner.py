from __future__ import annotations

import dataclasses
import json
import unittest

from rig_a.experiments.e0_12_recurrent_reasoner_assay import (
    CONTRACT_PATH,
    SPLIT_PATH,
    run,
    valid_fixture,
)
from wamrx.recurrent import (
    RecurrentContractError,
    UnresolvedConstraintState,
    audit_comparison,
    audit_depth_protocol,
    decide_halt,
)
from wamrx.recurrent_tasks import (
    FAMILIES,
    generate_family,
    load_split_registry,
    verify_frozen_splits,
)


class FrozenTaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = load_split_registry(SPLIT_PATH)

    def test_all_frozen_hashes_and_ood_axes_match(self) -> None:
        report = verify_frozen_splits(self.registry)
        self.assertTrue(report["passed"])
        self.assertFalse(report["hash_mismatches"])
        self.assertTrue(all(report["ood_axis_disjoint"].values()))
        self.assertEqual(
            report["maximum_required_reasoning_steps"]["algorithmic"],
            {"train": 4, "id": 4, "ood": 12},
        )
        self.assertLessEqual(
            max(
                maximum
                for family in FAMILIES
                for maximum in report["maximum_required_reasoning_steps"][
                    family
                ].values()
            ),
            12,
        )

    def test_generation_is_repeatable_and_split_ids_are_disjoint(self) -> None:
        all_ids: dict[str, set[str]] = {}
        for split, spec in self.registry["splits"].items():
            generated = {
                task.task_id
                for family in FAMILIES
                for task in generate_family(split, family, spec)
            }
            repeated = {
                task.task_id
                for family in FAMILIES
                for task in generate_family(split, family, spec)
            }
            self.assertEqual(generated, repeated)
            all_ids[split] = generated
        self.assertTrue(all_ids["train"].isdisjoint(all_ids["id"]))
        self.assertTrue(all_ids["train"].isdisjoint(all_ids["ood"]))
        self.assertEqual(
            {split: len(ids) for split, ids in all_ids.items()},
            {"train": 576, "id": 192, "ood": 192},
        )


class InterfaceAndAccountingTests(unittest.TestCase):
    def test_valid_fixture_binds_output_trace_evidence_and_compute(self) -> None:
        bundle, budget, trace, output, compute = valid_fixture()
        bundle.validate(required_regions=("finance", "operations"))
        trace.validate()
        output.validate(trace)
        compute.validate(budget)
        self.assertEqual(output.trace_hash, trace.trace_hash)
        self.assertEqual(output.evidence_bundle_hash, bundle.bundle_hash)

    def test_reinjection_state_chain_and_budget_fail_closed(self) -> None:
        _, budget, trace, _, compute = valid_fixture()
        no_reinjection = dataclasses.replace(
            trace,
            steps=(
                dataclasses.replace(
                    trace.steps[0], injected_evidence_bundle_hash="wrong"
                ),
                trace.steps[1],
            ),
        )
        with self.assertRaises(RecurrentContractError):
            no_reinjection.validate()
        broken_state = dataclasses.replace(
            trace,
            steps=(
                trace.steps[0],
                dataclasses.replace(trace.steps[1], high_state_in_hash="wrong"),
            ),
        )
        with self.assertRaises(RecurrentContractError):
            broken_state.validate()
        with self.assertRaises(RecurrentContractError):
            dataclasses.replace(
                compute, retrieval_calls=budget.maximum_retrieval_calls + 1
            ).validate(budget)

    def test_both_halting_paths_are_active(self) -> None:
        clear = UnresolvedConstraintState()
        conflict = UnresolvedConstraintState(
            conflicting_evidence=("latency",), answer_instability=0.3
        )
        self.assertEqual(
            decide_halt(
                "stop",
                clear,
                budget_exhausted=False,
                maximum_answer_instability=0.02,
            ),
            "resolved",
        )
        self.assertEqual(
            decide_halt(
                "stop",
                conflict,
                budget_exhausted=False,
                maximum_answer_instability=0.02,
            ),
            "continue",
        )
        self.assertEqual(
            decide_halt(
                "continue",
                clear,
                budget_exhausted=False,
                maximum_answer_instability=0.02,
            ),
            "continue",
        )
        self.assertEqual(
            decide_halt(
                "continue",
                conflict,
                budget_exhausted=True,
                maximum_answer_instability=0.02,
            ),
            "unresolved_budget",
        )


class RecurrentContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = json.loads(CONTRACT_PATH.read_text())

    def test_nominal_comparison_matches_and_confound_fails(self) -> None:
        thresholds = self.contract["promotion_thresholds"]
        nominal = audit_comparison(
            self.contract["comparison_arms"],
            maximum_parameter_spread=thresholds[
                "maximum_parameter_count_relative_spread"
            ],
            maximum_flop_budget_spread=thresholds[
                "maximum_inference_flop_budget_relative_spread"
            ],
        )
        self.assertTrue(nominal["passed"])
        confounded = [dict(arm) for arm in self.contract["comparison_arms"]]
        confounded[0]["encoder_id"] = "different-encoder"
        audit = audit_comparison(
            confounded,
            maximum_parameter_spread=thresholds[
                "maximum_parameter_count_relative_spread"
            ],
            maximum_flop_budget_spread=thresholds[
                "maximum_inference_flop_budget_relative_spread"
            ],
        )
        self.assertFalse(audit["passed"])
        self.assertIn("encoder_id", audit["mismatches"])

    def test_fixed_loop_protocol_is_detected(self) -> None:
        protocol = self.contract["depth_protocol"]
        self.assertTrue(audit_depth_protocol(protocol)["passed"])
        fixed = json.loads(json.dumps(protocol))
        fixed["training_macro_depth_distribution"]["support"] = [4]
        self.assertFalse(audit_depth_protocol(fixed)["passed"])

    def test_registered_e0_12_is_pre_model_and_complete(self) -> None:
        result = run()
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["model_comparison_status"], "NOT_RUN")
        self.assertTrue(all(result["checks"].values()))
        self.assertTrue(all(result["manipulation_checks"].values()))
        self.assertEqual(result["task_counts"]["total"], 960)


if __name__ == "__main__":
    unittest.main()
