from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from wamrx.native_memory_selection import (
    NativeMemorySelectionError,
    validate_selection,
    validate_selection_files,
)


ROOT = Path(__file__).resolve().parents[1]
SELECTION_PATH = ROOT / "contracts" / "wamrx_native_memory_selection_v1.json"
RESULT_PATH = ROOT / "results" / "e0_16_native_memory_comparison.json"


class NativeMemorySelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.selection = json.loads(SELECTION_PATH.read_text())
        cls.result = json.loads(RESULT_PATH.read_text())
        cls.bound_hash = cls.selection["selected_from"]["result_sha256"]

    def validate(self, selection: dict, result: dict) -> dict:
        return validate_selection(selection, result, result_sha256=self.bound_hash)

    def test_checked_in_selection_is_hash_bound_and_valid(self) -> None:
        report = validate_selection_files(ROOT)
        self.assertEqual(report["terminal_status"], "COMPLETE_RETAIN_EXPLICIT_MULTIVIEW")
        self.assertEqual(report["selected_arm_id"], "explicit-multiview-v1")
        self.assertFalse(report["learned_native_memory"])

    def test_terminal_status_drift_fails_closed(self) -> None:
        result = copy.deepcopy(self.result)
        result["status"] = "COMPLETE_ADOPT_LEARNED_MEMORY"
        result["promotion_audit"]["decision"] = "COMPLETE_ADOPT_LEARNED_MEMORY"
        with self.assertRaises(NativeMemorySelectionError):
            self.validate(self.selection, result)

    def test_missing_evaluation_row_fails_closed(self) -> None:
        result = copy.deepcopy(self.result)
        result["evaluation_rows"].pop()
        with self.assertRaises(NativeMemorySelectionError):
            self.validate(self.selection, result)

    def test_failed_manipulation_fails_closed(self) -> None:
        result = copy.deepcopy(self.result)
        result["promotion_audit"]["manipulations"][
            "M4_tombstone_disables_before_readout"
        ] = False
        with self.assertRaises(NativeMemorySelectionError):
            self.validate(self.selection, result)

    def test_selection_cannot_authorize_learned_memory(self) -> None:
        selection = copy.deepcopy(self.selection)
        selection["authorization"]["learned_native_memory"] = True
        with self.assertRaises(NativeMemorySelectionError):
            self.validate(selection, self.result)


if __name__ == "__main__":
    unittest.main()
