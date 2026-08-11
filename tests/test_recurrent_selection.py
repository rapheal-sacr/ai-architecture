from __future__ import annotations

import copy
import json
import pathlib
import unittest

from wamrx.recurrent_selection import (
    RecurrentSelectionError,
    validate_selection,
    validate_selection_files,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
SELECTION_PATH = ROOT / "contracts" / "wamrx_reasoner_selection_v1.json"
RESULT_PATH = ROOT / "results" / "e0_14_recurrent_model_comparison.json"


class RecurrentSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.selection = json.loads(SELECTION_PATH.read_text())
        cls.result = json.loads(RESULT_PATH.read_text())
        cls.bound_hash = cls.selection["selected_from"]["result_sha256"]

    def validate(self, selection: dict, result: dict) -> dict:
        return validate_selection(selection, result, result_sha256=self.bound_hash)

    def test_checked_in_selection_is_hash_bound_and_valid(self) -> None:
        report = validate_selection_files(ROOT)
        self.assertEqual(report["terminal_status"], "COMPLETE_RETAIN_FIXED")
        self.assertEqual(report["selected_arm_id"], "fixed-depth-v1")
        self.assertEqual(report["selected_macro_depth"], 4)

    def test_terminal_status_drift_cannot_select_fixed_depth(self) -> None:
        result = copy.deepcopy(self.result)
        result["status"] = "COMPLETE_ADOPT_FLAT"
        result["promotion_audit"]["decision"] = "COMPLETE_ADOPT_FLAT"
        with self.assertRaises(RecurrentSelectionError):
            self.validate(self.selection, result)

    def test_missing_arm_seed_pair_fails_closed(self) -> None:
        result = copy.deepcopy(self.result)
        result["completed_arm_seeds"].pop()
        with self.assertRaises(RecurrentSelectionError):
            self.validate(self.selection, result)

    def test_failed_manipulation_fails_closed(self) -> None:
        result = copy.deepcopy(self.result)
        result["promotion_audit"]["manipulations"][
            "M8_missing_protected_region_fails_closed"
        ] = False
        with self.assertRaises(RecurrentSelectionError):
            self.validate(self.selection, result)

    def test_selection_cannot_authorize_blocked_architecture(self) -> None:
        selection = copy.deepcopy(self.selection)
        selection["authorization"]["native_neural_memory"] = True
        with self.assertRaises(RecurrentSelectionError):
            self.validate(selection, self.result)

    def test_duplicate_retired_candidate_fails_closed(self) -> None:
        selection = copy.deepcopy(self.selection)
        selection["retired_general_core_candidates"].append(
            copy.deepcopy(selection["retired_general_core_candidates"][0])
        )
        with self.assertRaises(RecurrentSelectionError):
            self.validate(selection, self.result)


if __name__ == "__main__":
    unittest.main()
