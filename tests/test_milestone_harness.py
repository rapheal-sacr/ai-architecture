from __future__ import annotations

import unittest

from rig_a.experiments.e0_10_wamrx_memory_kernel import run
from wamrx.contracts import load_contracts


class MilestoneHarnessTests(unittest.TestCase):
    def test_frozen_contract_registry(self) -> None:
        contracts = load_contracts("contracts/wamrx_milestone1.json")
        self.assertEqual(len(contracts), 4)
        self.assertEqual(len({item.mechanism_id for item in contracts}), 4)

    def test_registered_milestone_experiment(self) -> None:
        result = run()
        self.assertEqual(result["verdict"], "PASS")
        self.assertTrue(all(result["checks"].values()))
        self.assertGreaterEqual(result["biased_compiler"]["pooled_coverage"], 0.90)
        self.assertEqual(result["biased_compiler"]["worst_region_coverage"], 0.0)
        self.assertEqual(result["complete_compiler"]["worst_region_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
